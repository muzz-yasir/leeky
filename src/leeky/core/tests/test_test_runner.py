"""Tests for test runner module."""

import pytest
import asyncio
from datetime import datetime
from typing import List, Optional
from ..test_runner import TestRunner, CompletionEngine
from ..prompt_manager import PromptManager
from ..types import (
    TextSource,
    PromptTemplate,
    PromptResult,
    BatchResult,
    LeekyError
)

class MockEngine(CompletionEngine):
    """Mock completion engine for testing."""
    
    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses or ["Mock response"]
        self.current_response = 0
        self.calls = []
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Return mock response."""
        self.calls.append((prompt, kwargs))
        response = self.responses[self.current_response]
        self.current_response = (self.current_response + 1) % len(self.responses)
        return response

@pytest.fixture
def prompt_manager():
    """Create a prompt manager with test templates."""
    manager = PromptManager()
    manager.register_template(
        name="test_template",
        template="Test prompt: {text}",
        parameters={},
        metadata={"test": True}
    )
    return manager

@pytest.fixture
def mock_engine():
    """Create a mock completion engine."""
    return MockEngine(["Response 1", "Response 2", "Response 3"])

@pytest.fixture
def test_runner(prompt_manager, mock_engine):
    """Create a test runner instance."""
    config = {
        "batch_size": 2,
        "max_retries": 2,
        "timeout": 1.0,
        "sampling_strategy": "all",
        "sampling_parameters": {}
    }
    return TestRunner(prompt_manager, mock_engine, config)

@pytest.fixture
def test_texts():
    """Create test text sources."""
    return [
        TextSource(
            content=f"Test content {i}",
            source_id=f"test_{i}",
            metadata={},
            timestamp=datetime.now()
        )
        for i in range(3)
    ]

@pytest.mark.asyncio
async def test_run_single(test_runner, test_texts):
    """Test running a single prompt test."""
    template = test_runner.prompt_manager.get_template("test_template")
    result = await test_runner.run_single(test_texts[0], template)
    
    assert isinstance(result, PromptResult)
    assert result.prompt_template == template
    assert result.input_text == test_texts[0]
    assert result.output_text == "Response 1"
    assert isinstance(result.timestamp, datetime)
    assert result.execution_time > 0

@pytest.mark.asyncio
async def test_run_batch(test_runner, test_texts):
    """Test running a batch of prompt tests."""
    templates = test_runner.prompt_manager.get_all_templates()
    result = await test_runner.run_batch(test_texts, templates)
    
    assert isinstance(result, BatchResult)
    assert len(result.prompt_results) == 3  # One per text
    assert isinstance(result.start_time, datetime)
    assert isinstance(result.end_time, datetime)
    assert result.start_time <= result.end_time

@pytest.mark.asyncio
async def test_run_batch_with_sampling(test_runner, test_texts):
    """Test batch running with sampling."""
    test_runner.sampling_strategy = "random"
    test_runner.sampling_parameters = {"size": 2}
    
    templates = test_runner.prompt_manager.get_all_templates()
    result = await test_runner.run_batch(test_texts, templates)
    
    assert len(result.prompt_results) == 2  # Sampled size

@pytest.mark.asyncio
async def test_retry_on_failure():
    """Test retry behavior on failure."""
    class FailingEngine(CompletionEngine):
        def __init__(self, success_on_retry: int):
            self.attempts = 0
            self.success_on_retry = success_on_retry
        
        async def complete(self, prompt: str, **kwargs) -> str:
            self.attempts += 1
            if self.attempts < self.success_on_retry:
                raise Exception("Simulated failure")
            return "Success after retry"
    
    engine = FailingEngine(success_on_retry=2)
    manager = PromptManager()
    manager.register_template(name="test", template="{text}")
    
    runner = TestRunner(
        manager,
        engine,
        {
            "batch_size": 1,
            "max_retries": 3,
            "timeout": 1.0
        }
    )
    
    text = TextSource("test", "test_id", {}, datetime.now())
    template = manager.get_template("test")
    
    result = await runner.run_single(text, template)
    assert result.output_text == "Success after retry"
    assert engine.attempts == 2

@pytest.mark.asyncio
async def test_timeout():
    """Test timeout handling."""
    class SlowEngine(CompletionEngine):
        async def complete(self, prompt: str, **kwargs) -> str:
            await asyncio.sleep(0.5)
            return "Slow response"
    
    engine = SlowEngine()
    manager = PromptManager()
    manager.register_template(name="test", template="{text}")
    
    runner = TestRunner(
        manager,
        engine,
        {
            "batch_size": 1,
            "max_retries": 1,
            "timeout": 0.1  # Short timeout
        }
    )
    
    text = TextSource("test", "test_id", {}, datetime.now())
    template = manager.get_template("test")
    
    with pytest.raises(LeekyError, match="Test execution timed out"):
        await runner.run_single(text, template)

def test_sampling_strategies(test_runner, test_texts):
    """Test different sampling strategies."""
    # Test random sampling
    sampled = test_runner._sample_texts(
        test_texts,
        "random",
        {"size": 2}
    )
    assert len(sampled) == 2
    assert all(text in test_texts for text in sampled)
    
    # Test first N sampling
    sampled = test_runner._sample_texts(
        test_texts,
        "first",
        {"size": 2}
    )
    assert len(sampled) == 2
    assert sampled == test_texts[:2]
    
    # Test all sampling
    sampled = test_runner._sample_texts(
        test_texts,
        "all",
        {}
    )
    assert len(sampled) == len(test_texts)
    assert sampled == test_texts
    
    # Test invalid strategy
    with pytest.raises(LeekyError):
        test_runner._sample_texts(test_texts, "invalid", {})

@pytest.mark.asyncio
async def test_concurrent_batch_processing(test_runner, test_texts):
    """Test concurrent processing of batches."""
    # Create a larger set of texts
    texts = test_texts * 3  # 9 texts total
    test_runner.batch_size = 3  # 3 texts per batch
    
    templates = test_runner.prompt_manager.get_all_templates()
    result = await test_runner.run_batch(texts, templates)
    
    assert len(result.prompt_results) == 9
    # Verify responses are distributed correctly
    responses = [r.output_text for r in result.prompt_results]
    assert responses.count("Response 1") == 3
    assert responses.count("Response 2") == 3
    assert responses.count("Response 3") == 3
