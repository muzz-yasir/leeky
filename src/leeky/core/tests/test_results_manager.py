"""Tests for results manager module."""

import pytest
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
from ..results_manager import ResultsManager
from ..types import (
    TextSource,
    PromptTemplate,
    PromptResult,
    BatchResult,
    MetricResult,
    EvaluationMetric,
    DataFormat,
    LeekyError
)

@pytest.fixture
def results_manager(tmp_path):
    """Create a results manager instance for testing."""
    return ResultsManager(tmp_path / "results")

@pytest.fixture
def test_batch_result():
    """Create a test batch result."""
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
    
    results = [
        PromptResult(
            prompt_template=template,
            input_text=text,
            output_text=f"Output {i}",
            metadata={"test_case": f"case{i}"},
            timestamp=datetime.now(),
            execution_time=0.1
        )
        for i in range(3)
    ]
    
    return BatchResult(
        prompt_results=results,
        metadata={"test_batch": True},
        start_time=datetime.now(),
        end_time=datetime.now()
    )

@pytest.fixture
def test_metrics():
    """Create test metric results."""
    metric = EvaluationMetric(
        name="test_metric",
        description="Test metric",
        version="1.0.0"
    )
    
    return {
        "test_metric": [
            MetricResult(
                metric=metric,
                score=0.5,
                details={"test": True},
                timestamp=datetime.now()
            )
            for _ in range(3)
        ]
    }

def test_save_load_json(results_manager, test_batch_result):
    """Test saving and loading results in JSON format."""
    # Save results
    file_path = results_manager.save_batch_result(
        test_batch_result,
        format=DataFormat.JSON
    )
    assert file_path.exists()
    
    # Verify JSON content
    with open(file_path, 'r') as f:
        data = json.load(f)
        assert "metadata" in data
        assert "results" in data
        assert len(data["results"]) == 3
    
    # Load results
    loaded = results_manager.load_batch_result(file_path)
    assert isinstance(loaded, BatchResult)
    assert len(loaded.prompt_results) == len(test_batch_result.prompt_results)
    
    # Verify content
    original = test_batch_result.prompt_results[0]
    loaded_result = loaded.prompt_results[0]
    assert loaded_result.output_text == original.output_text
    assert loaded_result.execution_time == original.execution_time

def test_save_load_csv(results_manager, test_batch_result):
    """Test saving and loading results in CSV format."""
    # Save results
    file_path = results_manager.save_batch_result(
        test_batch_result,
        format=DataFormat.CSV
    )
    assert file_path.exists()
    
    # Verify CSV content
    df = pd.read_csv(file_path)
    assert len(df) == len(test_batch_result.prompt_results)
    assert "template_name" in df.columns
    assert "output_text" in df.columns
    assert "execution_time" in df.columns
    
    # Load results
    loaded = results_manager.load_batch_result(file_path)
    assert isinstance(loaded, BatchResult)
    assert len(loaded.prompt_results) == len(test_batch_result.prompt_results)

def test_generate_report(results_manager, test_batch_result, test_metrics, tmp_path):
    """Test report generation."""
    output_path = tmp_path / "report.pdf"
    
    results_manager.generate_report(
        test_batch_result,
        test_metrics,
        output_path,
        format=DataFormat.PDF
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

def test_analyze_results(results_manager, test_batch_result, test_metrics):
    """Test results analysis."""
    analysis = results_manager.analyze_results(test_batch_result, test_metrics)
    
    assert "summary" in analysis
    assert "metrics" in analysis
    assert "total_tests" in analysis["summary"]
    assert "total_time" in analysis["summary"]
    assert "avg_execution_time" in analysis["summary"]
    
    metric_stats = analysis["metrics"]["test_metric"]
    assert "mean" in metric_stats
    assert "median" in metric_stats
    assert "std_dev" in metric_stats
    assert metric_stats["mean"] == 0.5  # All mock scores are 0.5

def test_invalid_format(results_manager, test_batch_result):
    """Test handling invalid format."""
    with pytest.raises(LeekyError):
        results_manager.save_batch_result(
            test_batch_result,
            format="invalid"
        )

def test_invalid_file(results_manager, tmp_path):
    """Test loading from invalid file."""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("invalid content")
    
    with pytest.raises(LeekyError):
        results_manager.load_batch_result(invalid_file)

def test_results_directory_creation(tmp_path):
    """Test results directory is created if it doesn't exist."""
    results_dir = tmp_path / "nonexistent" / "results"
    manager = ResultsManager(results_dir)
    assert results_dir.exists()

def test_batch_metadata_preservation(results_manager, test_batch_result):
    """Test batch metadata is preserved in JSON format."""
    file_path = results_manager.save_batch_result(
        test_batch_result,
        format=DataFormat.JSON
    )
    
    loaded = results_manager.load_batch_result(file_path)
    assert loaded.metadata == test_batch_result.metadata

def test_save_load_with_custom_metadata(results_manager, test_batch_result):
    """Test saving and loading with custom metadata."""
    # Add custom metadata
    test_batch_result.metadata["custom_field"] = "custom_value"
    test_batch_result.prompt_results[0].metadata["result_field"] = "result_value"
    
    # Save and load
    file_path = results_manager.save_batch_result(
        test_batch_result,
        format=DataFormat.JSON
    )
    loaded = results_manager.load_batch_result(file_path)
    
    # Verify metadata preservation
    assert loaded.metadata["custom_field"] == "custom_value"
    assert loaded.prompt_results[0].metadata["result_field"] == "result_value"

def test_report_generation_with_no_metrics(results_manager, test_batch_result, tmp_path):
    """Test report generation with no metrics."""
    output_path = tmp_path / "report.pdf"
    
    results_manager.generate_report(
        test_batch_result,
        {},  # Empty metrics
        output_path,
        format=DataFormat.PDF
    )
    
    assert output_path.exists()
