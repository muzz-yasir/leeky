"""Common types and exceptions for the leeky package."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

# Custom exceptions
class LeekyError(Exception):
    """Base exception for all leeky errors."""
    pass

class ConfigurationError(LeekyError):
    """Raised when there is an error in configuration."""
    pass

class DataLoadError(LeekyError):
    """Raised when there is an error loading data."""
    pass

class PromptError(LeekyError):
    """Raised when there is an error with prompt templates."""
    pass

class EvaluationError(LeekyError):
    """Raised when there is an error during evaluation."""
    pass

# Common types
@dataclass
class TextSource:
    """Represents a source of text data with metadata."""
    content: str
    source_id: str
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class PromptTemplate:
    """Represents a prompt template with metadata."""
    template: str
    name: str
    version: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class PromptResult:
    """Represents the result of applying a prompt template."""
    prompt_template: PromptTemplate
    input_text: TextSource
    output_text: str
    metadata: Dict[str, Any]
    timestamp: datetime
    execution_time: float

@dataclass
class EvaluationMetric:
    """Base class for evaluation metrics."""
    name: str
    description: str
    version: str

@dataclass
class MetricResult:
    """Represents the result of applying an evaluation metric."""
    metric: EvaluationMetric
    score: float
    details: Dict[str, Any]
    timestamp: datetime

@dataclass
class BatchResult:
    """Represents results from a batch of prompt tests."""
    prompt_results: List[PromptResult]
    metadata: Dict[str, Any]
    start_time: datetime
    end_time: datetime

class DataFormat(Enum):
    """Supported data formats for import/export."""
    JSON = "json"
    CSV = "csv"
    YAML = "yaml"
    PDF = "pdf"

@dataclass
class EngineConfig:
    """Configuration for a completion engine."""
    engine_type: str
    api_key: Optional[str]
    parameters: Dict[str, Any]

@dataclass
class TestConfig:
    """Configuration for test execution."""
    batch_size: int
    max_retries: int
    timeout: float
    sampling_strategy: str
    sampling_parameters: Dict[str, Any]
