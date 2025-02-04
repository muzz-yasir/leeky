"""Evaluation and ranking functionality for prompt testing."""

import logging
from typing import Dict, List, Optional, Sequence, Any
from datetime import datetime
from abc import ABC, abstractmethod
import statistics

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
                    template_name = result.details["prompt_template"].name
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
