"""Evaluation and ranking functionality for prompt testing."""

import logging
from typing import Dict, List, Optional, Sequence, Any, Set
from datetime import datetime
from abc import ABC, abstractmethod
import statistics
from rapidfuzz import fuzz
from rapidfuzz.process import extractOne

from .types import (
    EvaluationMetric,
    MetricResult,
    PromptResult,
    PromptTemplate,
    BatchResult,
    EvaluationError
)

logger = logging.getLogger(__name__)

class BaseMetric(ABC):
    """Base class for evaluation metrics."""
    
    @abstractmethod
    def evaluate(
        self,
        result: PromptResult
    ) -> MetricResult:
        """Evaluate a single prompt result.
        
        Args:
            result: PromptResult to evaluate.
            
        Returns:
            MetricResult containing the evaluation score and details.
            
        Raises:
            NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError("Metrics must implement evaluate()")

    @abstractmethod
    def evaluate_batch(
        self,
        results: Sequence[PromptResult]
    ) -> List[MetricResult]:
        """Evaluate a batch of prompt results.
        
        Args:
            results: Sequence of PromptResults to evaluate.
            
        Returns:
            List of MetricResults for the batch.
            
        Raises:
            NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError("Metrics must implement evaluate_batch()")

class CompletionOverlapMetric(BaseMetric):
    """Evaluates overlap between model completion and held-out text portion."""
    
    def __init__(
        self,
        name: str = "completion_overlap",
        description: str = "Measures overlap between model completion and held-out text",
        overlap_type: str = "token"
    ):
        """Initialize the completion overlap metric.
        
        Args:
            name: Name of the metric
            description: Description of the metric
            overlap_type: Type of overlap comparison ('token', 'char', 'sentence')
        """
        self.metric = EvaluationMetric(
            name=name,
            description=description,
            version="1.0.0"
        )
        self.overlap_type = overlap_type

    def _get_tokens(self, text: str) -> Set[str]:
        """Convert text to set of tokens for comparison.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Set of tokens
        """
        return set(text.lower().split())

    def _get_sentences(self, text: str) -> Set[str]:
        """Convert text to set of sentences for comparison.
        
        Args:
            text: Input text to split into sentences
            
        Returns:
            Set of sentences
        """
        return set(sent.strip() for sent in text.split(".") if sent.strip())

    def evaluate(self, result: PromptResult) -> MetricResult:
        """Evaluate overlap between model completion and held-out text.
        
        Args:
            result: PromptResult to evaluate
            
        Returns:
            MetricResult containing overlap score and details
        """
        try:
            if not result.input_text.completion_portion:
                return MetricResult(
                    metric=self.metric,
                    score=0.0,
                    details={"error": "No completion portion available"},
                    timestamp=datetime.now()
                )

            if self.overlap_type == "token":
                expected = self._get_tokens(result.input_text.completion_portion)
                actual = self._get_tokens(result.output_text)
            elif self.overlap_type == "char":
                expected = set(result.input_text.completion_portion.lower())
                actual = set(result.output_text.lower())
            elif self.overlap_type == "sentence":
                expected = self._get_sentences(result.input_text.completion_portion)
                actual = self._get_sentences(result.output_text)
            else:
                raise ValueError(f"Unknown overlap type: {self.overlap_type}")

            if not expected:
                return MetricResult(
                    metric=self.metric,
                    score=0.0,
                    details={"error": "Empty expected completion"},
                    timestamp=datetime.now()
                )

            overlap = len(expected.intersection(actual))
            precision = overlap / len(actual) if actual else 0.0
            recall = overlap / len(expected)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            details = {
                "overlap_type": self.overlap_type,
                "overlap_count": overlap,
                "expected_count": len(expected),
                "actual_count": len(actual),
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "prompt_template": result.prompt_template
            }

            # Store comparison details in the PromptResult
            if result.completion_comparison is None:
                result.completion_comparison = {}
            result.completion_comparison[self.metric.name] = details

            return MetricResult(
                metric=self.metric,
                score=f1,  # Use F1 score as the main metric
                details=details,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Completion overlap evaluation failed: {str(e)}")
            return MetricResult(
                metric=self.metric,
                score=0.0,
                details={"error": str(e)},
                timestamp=datetime.now()
            )

    def evaluate_batch(
        self,
        results: Sequence[PromptResult]
    ) -> List[MetricResult]:
        """Evaluate a batch of prompt results.
        
        Args:
            results: Sequence of PromptResults to evaluate
            
        Returns:
            List of MetricResults for the batch
        """
        return [self.evaluate(result) for result in results]


class LCSMetric(BaseMetric):
    """Evaluates prompt results using Longest Common Subsequence."""
    
    def __init__(self, name: str = "lcs", description: str = "Longest Common Subsequence metric"):
        """Initialize the LCS metric.
        
        Args:
            name: Name of the metric.
            description: Description of the metric.
        """
        self.metric = EvaluationMetric(
            name=name,
            description=description,
            version="1.0.0"
        )

    def _find_longest_common_subsequence(self, text1: str, text2: str) -> tuple[int, str]:
        """Find the longest common subsequence of words between two texts using rapidfuzz.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Tuple of (length of LCS, the LCS string)
        """
        # Split into sentences for more granular matching
        sentences1 = [s.strip() for s in text1.split('.') if s.strip()]
        sentences2 = [s.strip() for s in text2.split('.') if s.strip()]
        
        if not sentences1 or not sentences2:
            return 0, ""
            
        # Find best matching sequence using rapidfuzz
        best_match = extractOne(
            text1,
            [text2],
            scorer=fuzz.ratio,
            score_cutoff=50  # Minimum similarity threshold
        )
        
        if not best_match:
            return 0, ""
            
        matched_text = best_match[0]
        match_ratio = best_match[1] / 100.0  # Convert percentage to ratio
        
        # Count words in matched text
        words = matched_text.split()
        match_length = int(len(words) * match_ratio)  # Scale length by match ratio
        
        return match_length, matched_text

    def evaluate(self, result: PromptResult) -> MetricResult:
        """Evaluate using LCS between completion portion and output.
        
        Args:
            result: PromptResult to evaluate
            
        Returns:
            MetricResult containing LCS score and details
        """
        try:
            if not result.input_text.completion_portion:
                return MetricResult(
                    metric=self.metric,
                    score=0.0,
                    details={"error": "No completion portion available"},
                    timestamp=datetime.now()
                )

            expected = result.input_text.completion_portion
            actual = result.output_text
            
            if not expected:
                return MetricResult(
                    metric=self.metric,
                    score=0.0,
                    details={"error": "Empty expected completion"},
                    timestamp=datetime.now()
                )

            lcs_length, lcs = self._find_longest_common_subsequence(expected, actual)
            
            # Calculate score using both LCS length and fuzzy ratio
            expected_words = len(expected.split())
            if expected_words == 0:
                return MetricResult(
                    metric=self.metric,
                    score=0.0,
                    details={"error": "Empty expected completion"},
                    timestamp=datetime.now()
                )
                
            # Get fuzzy match ratio between expected and actual
            similarity_ratio = fuzz.ratio(expected, actual) / 100.0  # Convert percentage to ratio
            
            # Combine LCS and fuzzy ratio for final score
            lcs_score = lcs_length / expected_words if expected_words > 0 else 0.0
            score = (lcs_score + similarity_ratio) / 2  # Average of both metrics

            details = {
                "lcs_length": lcs_length,
                "expected_word_count": expected_words,
                "actual_word_count": len(actual.split()),
                "lcs": lcs,
                "prompt_template": result.prompt_template
            }

            # Store comparison details in the PromptResult
            if result.completion_comparison is None:
                result.completion_comparison = {}
            result.completion_comparison[self.metric.name] = details

            return MetricResult(
                metric=self.metric,
                score=score,
                details=details,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"LCS evaluation failed: {str(e)}")
            return MetricResult(
                metric=self.metric,
                score=0.0,
                details={"error": str(e)},
                timestamp=datetime.now()
            )

    def evaluate_batch(
        self,
        results: Sequence[PromptResult]
    ) -> List[MetricResult]:
        """Evaluate a batch of prompt results using LCS.
        
        Args:
            results: Sequence of PromptResults to evaluate
            
        Returns:
            List of MetricResults for the batch
        """
        return [self.evaluate(result) for result in results]


class FuzzyMatchMetric(BaseMetric):
    """Evaluates prompt results using multiple fuzzy string matching algorithms."""
    
    def __init__(
        self,
        name: str = "fuzzy_match",
        description: str = "Multi-algorithm fuzzy string matching metric"
    ):
        """Initialize the fuzzy match metric.
        
        Args:
            name: Name of the metric.
            description: Description of the metric.
        """
        self.metric = EvaluationMetric(
            name=name,
            description=description,
            version="1.0.0"
        )

    def evaluate(self, result: PromptResult) -> MetricResult:
        """Evaluate using multiple fuzzy matching algorithms.
        
        Args:
            result: PromptResult to evaluate
            
        Returns:
            MetricResult containing fuzzy match scores and details
        """
        try:
            if not result.input_text.completion_portion:
                return MetricResult(
                    metric=self.metric,
                    score=0.0,
                    details={"error": "No completion portion available"},
                    timestamp=datetime.now()
                )

            expected = result.input_text.completion_portion
            actual = result.output_text
            
            if not expected:
                return MetricResult(
                    metric=self.metric,
                    score=0.0,
                    details={"error": "Empty expected completion"},
                    timestamp=datetime.now()
                )

            # Calculate various fuzzy match scores
            ratio_score = fuzz.ratio(expected, actual) / 100.0
            partial_ratio_score = fuzz.partial_ratio(expected, actual) / 100.0
            token_sort_score = fuzz.token_sort_ratio(expected, actual) / 100.0
            token_set_score = fuzz.token_set_ratio(expected, actual) / 100.0
            weighted_score = fuzz.WRatio(expected, actual) / 100.0
            
            # Combine scores with weights
            weights = {
                'ratio': 0.2,
                'partial_ratio': 0.2,
                'token_sort': 0.2,
                'token_set': 0.2,
                'weighted': 0.2
            }
            
            final_score = (
                ratio_score * weights['ratio'] +
                partial_ratio_score * weights['partial_ratio'] +
                token_sort_score * weights['token_sort'] +
                token_set_score * weights['token_set'] +
                weighted_score * weights['weighted']
            )

            details = {
                "ratio_score": ratio_score,
                "partial_ratio_score": partial_ratio_score,
                "token_sort_score": token_sort_score,
                "token_set_score": token_set_score,
                "weighted_score": weighted_score,
                "final_score": final_score,
                "expected_length": len(expected),
                "actual_length": len(actual),
                "prompt_template": result.prompt_template
            }

            # Store comparison details in the PromptResult
            if result.completion_comparison is None:
                result.completion_comparison = {}
            result.completion_comparison[self.metric.name] = details

            return MetricResult(
                metric=self.metric,
                score=final_score,
                details=details,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Fuzzy match evaluation failed: {str(e)}")
            return MetricResult(
                metric=self.metric,
                score=0.0,
                details={"error": str(e)},
                timestamp=datetime.now()
            )

    def evaluate_batch(
        self,
        results: Sequence[PromptResult]
    ) -> List[MetricResult]:
        """Evaluate a batch of prompt results using fuzzy matching.
        
        Args:
            results: Sequence of PromptResults to evaluate
            
        Returns:
            List of MetricResults for the batch
        """
        return [self.evaluate(result) for result in results]


class TokenMatchMetric(BaseMetric):
    """Evaluates prompt results based on token matching."""
    
    def __init__(self, name: str = "token_match", description: str = "Token matching metric"):
        """Initialize the token match metric.
        
        Args:
            name: Name of the metric.
            description: Description of the metric.
        """
        self.metric = EvaluationMetric(
            name=name,
            description=description,
            version="1.0.0"
        )

    def evaluate(self, result: PromptResult) -> MetricResult:
        """Evaluate a single prompt result using token matching.
        
        Args:
            result: PromptResult to evaluate.
            
        Returns:
            MetricResult containing the token match score.
        """
        try:
            # Simple token overlap implementation
            input_tokens = set(result.input_text.content.lower().split())
            output_tokens = set(result.output_text.lower().split())
            
            if not input_tokens:
                return MetricResult(
                    metric=self.metric,
                    score=0.0,
                    details={"error": "Empty input text"},
                    timestamp=datetime.now()
                )
            
            overlap = len(input_tokens.intersection(output_tokens))
            score = overlap / len(input_tokens)
            
            return MetricResult(
                metric=self.metric,
                score=score,
                details={
                    "overlap_tokens": overlap,
                    "input_tokens": len(input_tokens),
                    "output_tokens": len(output_tokens),
                    "prompt_template": result.prompt_template
                },
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Token match evaluation failed: {str(e)}")
            return MetricResult(
                metric=self.metric,
                score=0.0,
                details={"error": str(e)},
                timestamp=datetime.now()
            )

    def evaluate_batch(
        self,
        results: Sequence[PromptResult]
    ) -> List[MetricResult]:
        """Evaluate a batch of prompt results using token matching.
        
        Args:
            results: Sequence of PromptResults to evaluate.
            
        Returns:
            List of MetricResults for the batch.
        """
        return [self.evaluate(result) for result in results]

class Evaluator:
    """Handles evaluation and ranking of prompt test results."""
    
    def __init__(self, metrics: Sequence[BaseMetric]):
        """Initialize the evaluator.
        
        Args:
            metrics: Sequence of evaluation metrics to use.
        """
        self.metrics = list(metrics)

    def evaluate_single(
        self,
        result: PromptResult
    ) -> Dict[str, MetricResult]:
        """Evaluate a single prompt result using all metrics.
        
        Args:
            result: PromptResult to evaluate.
            
        Returns:
            Dictionary mapping metric names to MetricResults.
            
        Raises:
            EvaluationError: If evaluation fails.
        """
        try:
            return {
                metric.metric.name: metric.evaluate(result)
                for metric in self.metrics
            }
        except Exception as e:
            raise EvaluationError(f"Failed to evaluate result: {str(e)}")

    def evaluate_batch(
        self,
        batch_result: BatchResult
    ) -> Dict[str, List[MetricResult]]:
        """Evaluate a batch of results using all metrics.
        
        Args:
            batch_result: BatchResult to evaluate.
            
        Returns:
            Dictionary mapping metric names to lists of MetricResults.
            
        Raises:
            EvaluationError: If batch evaluation fails.
        """
        try:
            return {
                metric.metric.name: metric.evaluate_batch(batch_result.prompt_results)
                for metric in self.metrics
            }
        except Exception as e:
            raise EvaluationError(f"Failed to evaluate batch: {str(e)}")

    def rank_prompts(
        self,
        evaluation_results: Dict[str, List[MetricResult]],
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Rank prompts based on evaluation results.
        
        Args:
            evaluation_results: Dictionary of evaluation results from evaluate_batch().
            weights: Optional dictionary mapping metric names to weights.
                    If None, all metrics are weighted equally.
                    
        Returns:
            List of dictionaries containing ranking information, sorted by score.
            Each dictionary contains:
                - template_name: Name of the prompt template
                - overall_score: Weighted average score across all metrics
                - metric_scores: Dictionary of individual metric scores
                - rank: Integer ranking (1-based)
            
        Raises:
            EvaluationError: If ranking fails.
        """
        try:
            if not evaluation_results:
                raise EvaluationError("No evaluation results provided")
                
            # Normalize weights
            if weights is None:
                weights = {name: 1.0 for name in evaluation_results.keys()}
            total_weight = sum(weights.values())
            normalized_weights = {
                name: weight / total_weight
                for name, weight in weights.items()
            }
            
            # Calculate per-template scores for each metric
            template_scores: Dict[str, Dict[str, float]] = {}
            
            for metric_name, metric_results in evaluation_results.items():
                # Group results by template
                template_results = {}
                for result in metric_results:
                    template_name = result.details["prompt_template"]["name"]
                    if template_name not in template_results:
                        template_results[template_name] = []
                    template_results[template_name].append(result.score)
                
                # Calculate average score for each template
                for template_name, scores in template_results.items():
                    if template_name not in template_scores:
                        template_scores[template_name] = {}
                    template_scores[template_name][metric_name] = statistics.mean(scores)
            
            # Calculate overall scores and create ranking
            rankings = []
            for template_name, metric_scores in template_scores.items():
                overall_score = sum(
                    score * normalized_weights[metric_name]
                    for metric_name, score in metric_scores.items()
                )
                
                rankings.append({
                    "template_name": template_name,
                    "overall_score": overall_score,
                    "metric_scores": metric_scores,
                })
            
            # Sort by overall score and add ranks
            rankings.sort(key=lambda x: x["overall_score"], reverse=True)
            for i, ranking in enumerate(rankings):
                ranking["rank"] = i + 1
            
            return rankings
            
        except Exception as e:
            raise EvaluationError(f"Failed to rank prompts: {str(e)}")

    def get_statistics(
        self,
        evaluation_results: Dict[str, List[MetricResult]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for evaluation results.
        
        Args:
            evaluation_results: Dictionary of evaluation results from evaluate_batch().
            
        Returns:
            Dictionary mapping metric names to statistics dictionaries containing:
                - mean: Mean score
                - median: Median score
                - std_dev: Standard deviation
                - min: Minimum score
                - max: Maximum score
            
        Raises:
            EvaluationError: If statistics calculation fails.
        """
        try:
            stats = {}
            for metric_name, results in evaluation_results.items():
                scores = [r.score for r in results]
                if not scores:
                    continue
                    
                stats[metric_name] = {
                    "mean": statistics.mean(scores),
                    "median": statistics.median(scores),
                    "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "min": min(scores),
                    "max": max(scores)
                }
            return stats
            
        except Exception as e:
            raise EvaluationError(f"Failed to calculate statistics: {str(e)}")
