"""Tests for evaluator module."""

import pytest
from datetime import datetime
from typing import List, Sequence
from leeky.core.evaluator import Evaluator, BaseMetric, TokenMatchMetric, LCSMetric
from leeky.core.types import (
    TextSource,
    PromptTemplate,
    PromptResult,
    BatchResult,
    MetricResult,
    EvaluationMetric,
    EvaluationError
)

class MockMetric(BaseMetric):
    """Mock metric for testing."""
    
    def __init__(self, name: str = "mock_metric"):
        self.metric = EvaluationMetric(
            name=name,
            description="Mock metric for testing",
            version="1.0.0"
        )
    
    def evaluate(self, result: PromptResult) -> MetricResult:
        """Return mock evaluation result."""
        return MetricResult(
            metric=self.metric,
            score=0.5,
            details={
                "mock": True,
                "prompt_template": result.prompt_template
            },
            timestamp=datetime.now()
        )
    
    def evaluate_batch(self, results: Sequence[PromptResult]) -> List[MetricResult]:
        """Return mock batch evaluation results."""
        return [self.evaluate(result) for result in results]

@pytest.fixture
def evaluator():
    """Create an evaluator instance with mock metrics."""
    return Evaluator([
        MockMetric("metric1"),
        MockMetric("metric2")
    ])

@pytest.fixture
def test_results():
    """Create test prompt results."""
    template = PromptTemplate(
        template="Test template",
        name="test",
        version="1.0.0",
        parameters={},
        metadata={},
        created_at=datetime.now()
    )
    
    text = TextSource(
        content="Test content",
        source_id="test",
        metadata={},
        timestamp=datetime.now()
    )
    
    return [
        PromptResult(
            prompt_template=template,
            input_text=text,
            output_text=f"Output {i}",
            metadata={},
            timestamp=datetime.now(),
            execution_time=0.1
        )
        for i in range(3)
    ]

@pytest.fixture
def batch_result(test_results):
    """Create a test batch result."""
    return BatchResult(
        prompt_results=test_results,
        metadata={},
        start_time=datetime.now(),
        end_time=datetime.now()
    )

def test_evaluate_single(evaluator, test_results):
    """Test evaluating a single result."""
    result = evaluator.evaluate_single(test_results[0])
    
    assert len(result) == 2  # Two metrics
    assert "metric1" in result
    assert "metric2" in result
    assert all(isinstance(r, MetricResult) for r in result.values())

def test_evaluate_batch(evaluator, batch_result):
    """Test evaluating a batch of results."""
    results = evaluator.evaluate_batch(batch_result)
    
    assert len(results) == 2  # Two metrics
    assert "metric1" in results
    assert "metric2" in results
    assert all(isinstance(r, list) for r in results.values())
    assert all(len(r) == 3 for r in results.values())  # Three results each

def test_rank_prompts(evaluator, batch_result):
    """Test ranking prompts based on evaluation results."""
    evaluation = evaluator.evaluate_batch(batch_result)
    rankings = evaluator.rank_prompts(evaluation)
    
    assert len(rankings) == 1  # One template
    assert "template_name" in rankings[0]
    assert "overall_score" in rankings[0]
    assert "metric_scores" in rankings[0]
    assert "rank" in rankings[0]
    assert rankings[0]["rank"] == 1

def test_rank_prompts_with_weights(evaluator, batch_result):
    """Test ranking prompts with custom weights."""
    evaluation = evaluator.evaluate_batch(batch_result)
    weights = {"metric1": 0.8, "metric2": 0.2}
    rankings = evaluator.rank_prompts(evaluation, weights)
    
    assert len(rankings) == 1
    assert rankings[0]["overall_score"] == 0.5  # Both metrics return 0.5

def test_get_statistics(evaluator, batch_result):
    """Test calculating statistics from evaluation results."""
    evaluation = evaluator.evaluate_batch(batch_result)
    stats = evaluator.get_statistics(evaluation)
    
    assert len(stats) == 2  # Two metrics
    for metric_stats in stats.values():
        assert "mean" in metric_stats
        assert "median" in metric_stats
        assert "std_dev" in metric_stats
        assert "min" in metric_stats
        assert "max" in metric_stats
        assert metric_stats["mean"] == 0.5  # Mock metric always returns 0.5

def test_lcs_metric():
    """Test the LCS metric implementation."""
    metric = LCSMetric()
    
    template = PromptTemplate(
        template="Test",
        name="test",
        version="1.0",
        parameters={},
        metadata={},
        created_at=datetime.now()
    )
    
    # Create a text source with completion portion
    text = TextSource(
        content="The quick brown fox jumps over the lazy dog",
        source_id="test",
        metadata={},
        timestamp=datetime.now()
    )
    text.completion_portion = "jumps over the lazy dog"
    
    # Test exact match
    result = PromptResult(
        prompt_template=template,
        input_text=text,
        output_text="jumps over the lazy dog",
        metadata={},
        timestamp=datetime.now(),
        execution_time=0.1
    )
    evaluation = metric.evaluate(result)
    assert evaluation.score == 1.0
    assert evaluation.details["lcs_length"] == len(text.completion_portion.split())
    # Compare normalized words
    expected_words = [w.strip('.,!?()[]{}""\'\'').lower() for w in text.completion_portion.split()]
    actual_words = [w.strip('.,!?()[]{}""\'\'').lower() for w in evaluation.details["lcs"].split()]
    assert actual_words == expected_words
    
    # Test partial match
    result = PromptResult(
        prompt_template=template,
        input_text=text,
        output_text="jumps quickly over a lazy dog",
        metadata={},
        timestamp=datetime.now(),
        execution_time=0.1
    )
    evaluation = metric.evaluate(result)
    assert 0 < evaluation.score < 1.0
    assert evaluation.details["lcs"] == "jumps over lazy dog"
    
    # Test no match
    result = PromptResult(
        prompt_template=template,
        input_text=text,
        output_text="something completely different here",
        metadata={},
        timestamp=datetime.now(),
        execution_time=0.1
    )
    evaluation = metric.evaluate(result)
    assert evaluation.score == 0.0
    assert evaluation.details["lcs_length"] == 0
    
    # Test empty completion
    text_empty = TextSource(
        content="Empty test",
        source_id="test",
        metadata={},
        timestamp=datetime.now()
    )
    text_empty.completion_portion = ""
    
    result = PromptResult(
        prompt_template=template,
        input_text=text_empty,
        output_text="any output",
        metadata={},
        timestamp=datetime.now(),
        execution_time=0.1
    )
    evaluation = metric.evaluate(result)
    assert evaluation.score == 0.0
    assert "error" in evaluation.details


def test_token_match_metric():
    """Test the token match metric implementation."""
    metric = TokenMatchMetric()
    
    template = PromptTemplate(
        template="Test",
        name="test",
        version="1.0",
        parameters={},
        metadata={},
        created_at=datetime.now()
    )
    
    text = TextSource(
        content="The quick brown fox",
        source_id="test",
        metadata={},
        timestamp=datetime.now()
    )
    
    # Test exact match
    result = PromptResult(
        prompt_template=template,
        input_text=text,
        output_text="The quick brown fox",
        metadata={},
        timestamp=datetime.now(),
        execution_time=0.1
    )
    evaluation = metric.evaluate(result)
    assert evaluation.score == 1.0
    
    # Test partial match
    result = PromptResult(
        prompt_template=template,
        input_text=text,
        output_text="The quick fox runs",
        metadata={},
        timestamp=datetime.now(),
        execution_time=0.1
    )
    evaluation = metric.evaluate(result)
    assert 0 < evaluation.score < 1.0
    
    # Test no match
    result = PromptResult(
        prompt_template=template,
        input_text=text,
        output_text="Something completely different",
        metadata={},
        timestamp=datetime.now(),
        execution_time=0.1
    )
    evaluation = metric.evaluate(result)
    assert evaluation.score == 0.0

def test_empty_evaluation(evaluator):
    """Test handling empty evaluation results."""
    with pytest.raises(EvaluationError):
        evaluator.rank_prompts({})

def test_invalid_weights(evaluator, batch_result):
    """Test handling invalid weights."""
    evaluation = evaluator.evaluate_batch(batch_result)
    with pytest.raises(EvaluationError):
        # Weight for nonexistent metric
        evaluator.rank_prompts(evaluation, {"invalid_metric": 1.0})

def test_multiple_templates(evaluator):
    """Test evaluating results from multiple templates."""
    # Create results with different templates
    template1 = PromptTemplate(
        template="Template 1",
        name="test1",
        version="1.0",
        parameters={},
        metadata={},
        created_at=datetime.now()
    )
    
    template2 = PromptTemplate(
        template="Template 2",
        name="test2",
        version="1.0",
        parameters={},
        metadata={},
        created_at=datetime.now()
    )
    
    text = TextSource(
        content="Test content",
        source_id="test",
        metadata={},
        timestamp=datetime.now()
    )
    
    results = [
        PromptResult(
            prompt_template=template1,
            input_text=text,
            output_text="Output 1",
            metadata={},
            timestamp=datetime.now(),
            execution_time=0.1
        ),
        PromptResult(
            prompt_template=template2,
            input_text=text,
            output_text="Output 2",
            metadata={},
            timestamp=datetime.now(),
            execution_time=0.1
        )
    ]
    
    batch = BatchResult(
        prompt_results=results,
        metadata={},
        start_time=datetime.now(),
        end_time=datetime.now()
    )
    
    evaluation = evaluator.evaluate_batch(batch)
    rankings = evaluator.rank_prompts(evaluation)
    
    assert len(rankings) == 2  # Two templates
    assert rankings[0]["rank"] == 1
    assert rankings[1]["rank"] == 2
