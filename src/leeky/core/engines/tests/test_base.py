"""Base test class for completion engines."""

import pytest
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

from ..base_engine import CompletionEngine


class MockEngine(CompletionEngine):
    """Mock engine for testing base functionality."""
    
    def __init__(self):
        self._name = "mock-engine"
        self._max_tokens = 1000
        
    async def complete(self, prompt: str, **kwargs) -> str:
        return "mock completion"
        

    @property
    def name(self) -> str:
        return self._name
        
    @property
    def max_tokens(self) -> int:
        return self._max_tokens
        

    @property
    def default_params(self) -> dict:
        return {
            "temperature": 0.7,
            "max_tokens": None
        }


class BaseEngineTest:
    """Base test class for completion engines."""
    
    @pytest.fixture
    def engine(self) -> CompletionEngine:
        """Return engine instance for testing.
        
        Must be implemented by subclasses to return their specific engine.
        """
        raise NotImplementedError
    
    @pytest.mark.asyncio
    async def test_complete_basic(self, engine):
        """Test basic completion functionality."""
        prompt = "Test prompt"
        response = await engine.complete(prompt)
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_complete_with_params(self, engine):
        """Test completion with custom parameters."""
        prompt = "Test prompt"
        response = await engine.complete(
            prompt,
            temperature=0.5,
            max_tokens=50
        )
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_name(self, engine):
        """Test engine name property."""
        assert isinstance(engine.name, str)
        assert len(engine.name) > 0
    
    def test_max_tokens(self, engine):
        """Test max tokens property."""
        assert isinstance(engine.max_tokens, int)
        assert engine.max_tokens > 0
    
    def test_default_params(self, engine):
        """Test default parameters property."""
        params = engine.default_params
        assert isinstance(params, dict)
        assert "temperature" in params
        assert isinstance(params["temperature"], (int, float))


class TestMockEngine(BaseEngineTest):
    """Test the mock engine implementation."""
    
    @pytest.fixture
    def engine(self):
        """Return mock engine instance."""
        return MockEngine()
    
    def test_mock_specific_behavior(self, engine):
        """Test mock-specific functionality."""
        assert engine.name == "mock-engine"
        assert engine.max_tokens == 1000
