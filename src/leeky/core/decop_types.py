"""Types for DE-COP training data detection feature."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum

@dataclass
class DecopPassage:
    """Represents a text passage and its paraphrased versions."""
    text: str
    source_url: str
    token_count: int
    paraphrases: List[str] = field(default_factory=list)
    metadata: Dict[str, any] = field(default_factory=dict)

@dataclass
class QuizPermutation:
    """Represents a single quiz permutation with answer options."""
    question: str
    options: List[str]  # First option is always correct in this internal representation
    correct_index: int  # Index of correct answer in shuffled options
    source_passage: DecopPassage
    metadata: Dict[str, any] = field(default_factory=dict)

@dataclass
class QuizResponse:
    """Represents the model's response to a quiz permutation."""
    permutation: QuizPermutation
    selected_index: int
    confidence_score: float
    response_time: float
    raw_response: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PassageResults:
    """Results for all permutations of a single passage."""
    passage: DecopPassage
    responses: List[QuizResponse]
    correct_count: int
    accuracy: float
    avg_confidence: float
    position_bias: Dict[int, float]  # Maps position to selection frequency

@dataclass
class ComparisonStats:
    """Statistical comparison between suspect and clean passages."""
    suspect_accuracy: float
    clean_accuracy: float
    accuracy_difference: float
    p_value: float
    confidence_interval: tuple[float, float]
    effect_size: float
    sample_sizes: Dict[str, int]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DecopResult:
    """Complete results of a DE-COP analysis run."""
    suspect_passages: List[PassageResults]
    clean_passages: List[PassageResults]
    stats: ComparisonStats
    metadata: Dict[str, any]
    start_time: datetime
    end_time: datetime
    
    @property
    def duration(self) -> float:
        """Get analysis duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()

class ProcessingStage(Enum):
    """Stages of DE-COP processing pipeline."""
    EXTRACTING_PASSAGES = "extracting_passages"
    GENERATING_PARAPHRASES = "generating_paraphrases"
    CREATING_QUIZZES = "creating_quizzes"
    TESTING_MODEL = "testing_model"
    ANALYZING_RESULTS = "analyzing_results"
    COMPLETE = "complete"
    ERROR = "error"

@dataclass
class ProcessingStatus:
    """Status tracking for DE-COP processing."""
    stage: ProcessingStage
    progress: float  # 0.0 to 1.0
    message: str
    error: Optional[str] = None
    details: Dict[str, any] = field(default_factory=dict)
