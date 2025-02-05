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
    """Represents a source of text data with metadata and optional splitting for completion tasks."""
    content: str
    source_id: str
    metadata: Dict[str, Any]
    timestamp: datetime
    context_portion: Optional[str] = None
    completion_portion: Optional[str] = None
    split_metadata: Optional[Dict[str, Any]] = None

    def split_for_completion(self, split_ratio: float = 0.8, strategy: str = "ratio") -> None:
        """Split the content into context and completion portions.
        
        Args:
            split_ratio: Ratio of text to use as context (0.0 to 1.0)
            strategy: Splitting strategy ('ratio', 'sentence', 'paragraph')
            
        Raises:
            ValueError: If split_ratio is not between 0 and 1
        """
        if not 0 < split_ratio < 1:
            raise ValueError("split_ratio must be between 0 and 1")
            
        if strategy == "ratio":
            split_point = int(len(self.content) * split_ratio)
            self.context_portion = self.content[:split_point]
            self.completion_portion = self.content[split_point:]
            
        elif strategy == "sentence":
            sentences = self.content.split(". ")
            split_point = int(len(sentences) * split_ratio)
            self.context_portion = ". ".join(sentences[:split_point]) + "."
            self.completion_portion = ". ".join(sentences[split_point:])
            
        elif strategy == "paragraph":
            paragraphs = self.content.split("\n\n")
            split_point = int(len(paragraphs) * split_ratio)
            self.context_portion = "\n\n".join(paragraphs[:split_point])
            self.completion_portion = "\n\n".join(paragraphs[split_point:])
            
        else:
            raise ValueError(f"Unknown splitting strategy: {strategy}")
            
        self.split_metadata = {
            "strategy": strategy,
            "split_ratio": split_ratio,
            "context_length": len(self.context_portion),
            "completion_length": len(self.completion_portion)
        }

class TemplateType(Enum):
    """Types of prompt templates."""
    INSTRUCTION = "instruction"
    JAILBREAK = "jailbreak"

@dataclass
class PromptTemplate:
    """Represents a prompt template with metadata."""
    template: str
    name: str
    version: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    template_type: TemplateType = TemplateType.INSTRUCTION

@dataclass
class PromptResult:
    """Represents the result of applying a prompt template."""
    prompt_template: Dict[str, Any]
    prompt_string: str
    input_text: TextSource
    output_text: str
    metadata: Dict[str, Any]
    timestamp: datetime
    execution_time: float
    completion_comparison: Optional[Dict[str, Any]] = None

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
@dataclass
class TestConfig:
    """Configuration for test execution."""
    batch_size: int
    max_retries: int
    timeout: float
    sampling_strategy: str
    sampling_parameters: Dict[str, Any]
    text_splitting: Dict[str, Any] = None
