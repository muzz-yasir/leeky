"""Results management and analysis functionality."""

import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import statistics

from .types import (
    BatchResult,
    PromptResult,
    MetricResult,
    DataFormat,
    LeekyError
)

logger = logging.getLogger(__name__)

class ResultsManager:
    """Manages storage and analysis of test results."""
    
    def __init__(self, results_dir: Union[str, Path]):
        """Initialize the results manager.
        
        Args:
            results_dir: Directory to store results in.
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_batch_result(
        self,
        result: BatchResult,
        format: DataFormat = DataFormat.JSON
    ) -> Path:
        """Save a batch result to file.
        
        Args:
            result: BatchResult to save.
            format: Format to save in (JSON or CSV).
            
        Returns:
            Path to the saved file.
            
        Raises:
            LeekyError: If saving fails.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            if format == DataFormat.JSON:
                return self._save_json(result, f"batch_result_{timestamp}.json")
            elif format == DataFormat.CSV:
                return self._save_csv(result, f"batch_result_{timestamp}.csv")
            else:
                raise LeekyError(f"Unsupported format: {format}")
                
        except Exception as e:
            raise LeekyError(f"Failed to save batch result: {str(e)}")

    def load_batch_result(self, file_path: Union[str, Path]) -> BatchResult:
        """Load a batch result from file.
        
        Args:
            file_path: Path to the file to load.
            
        Returns:
            Loaded BatchResult object.
            
        Raises:
            LeekyError: If loading fails.
        """
        path = Path(file_path)
        try:
            if path.suffix == ".json":
                return self._load_json(path)
            elif path.suffix == ".csv":
                return self._load_csv(path)
            else:
                raise LeekyError(f"Unsupported file format: {path.suffix}")
                
        except Exception as e:
            raise LeekyError(f"Failed to load batch result: {str(e)}")

    def generate_report(
        self,
        result: BatchResult,
        metrics: Dict[str, List[MetricResult]],
        output_path: Union[str, Path],
        format: DataFormat = DataFormat.PDF
    ) -> None:
        """Generate a report from results.
        
        Args:
            result: BatchResult to generate report for.
            metrics: Dictionary of metric results from evaluator.
            output_path: Path to save report to.
            format: Format to save report in.
            
        Raises:
            LeekyError: If report generation fails.
        """
        try:
            if format == DataFormat.PDF:
                self._generate_pdf_report(result, metrics, output_path)
            else:
                raise LeekyError(f"Unsupported report format: {format}")
                
        except Exception as e:
            raise LeekyError(f"Failed to generate report: {str(e)}")

    def _save_json(self, result: BatchResult, filename: str) -> Path:
        """Save batch result as JSON."""
        file_path = self.results_dir / filename
        
        try:
            data = {
                "metadata": result.metadata,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "results": [
                    {
                        "template": r.prompt_template,
                        "input_text": {
                            "content": r.input_text.content,
                            "source_id": r.input_text.source_id,
                            "metadata": r.input_text.metadata,
                            "timestamp": r.input_text.timestamp.isoformat(),
                            "context_portion": r.input_text.context_portion,
                            "completion_portion": r.input_text.completion_portion,
                            "split_metadata": r.input_text.split_metadata
                        },
                        "output_text": r.output_text,
                        "prompt_string": r.prompt_string,
                        "metadata": r.metadata,
                        "timestamp": r.timestamp.isoformat(),
                        "execution_time": r.execution_time,
                        "completion_comparison": r.completion_comparison if hasattr(r, 'completion_comparison') else None
                    }
                    for r in result.prompt_results
                ]
            }
            
            with file_path.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            return file_path
            
        except Exception as e:
            raise LeekyError(f"Failed to save JSON: {str(e)}")

    def _save_csv(self, result: BatchResult, filename: str) -> Path:
        """Save batch result as CSV."""
        file_path = self.results_dir / filename
        
        try:
            rows = []
            for r in result.prompt_results:
                # Flatten template metadata into columns
                template_metadata = {
                    f"template_{k}": v 
                    for k, v in r.prompt_template.metadata.items()
                }
                
                rows.append({
                    "template_name": r.prompt_template.name,
                    "template_version": r.prompt_template.version,
                    "input_text": r.input_text.content,
                    "output_text": r.output_text,
                    "prompt_string": r.prompt_string,
                    "execution_time": r.execution_time,
                    "timestamp": r.timestamp.isoformat(),
                    **template_metadata,
                    **r.metadata
                })
            
            df = pd.DataFrame(rows)
            df.to_csv(file_path, index=False)
            return file_path
            
        except Exception as e:
            raise LeekyError(f"Failed to save CSV: {str(e)}")

    def _load_json(self, file_path: Path) -> BatchResult:
        """Load batch result from JSON."""
        try:
            with file_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct BatchResult object
            # Note: This is a simplified reconstruction that may not preserve
            # all original object relationships
            return BatchResult(
                prompt_results=[
                    PromptResult(
                        prompt_template=r["template_name"],
                        input_text=r["input_text"],
                        output_text=r["output_text"],
                        metadata=r["metadata"],
                        timestamp=datetime.fromisoformat(r["timestamp"]),
                        execution_time=r["execution_time"]
                    )
                    for r in data["results"]
                ],
                metadata=data["metadata"],
                start_time=datetime.fromisoformat(data["start_time"]),
                end_time=datetime.fromisoformat(data["end_time"])
            )
            
        except Exception as e:
            raise LeekyError(f"Failed to load JSON: {str(e)}")

    def _load_csv(self, file_path: Path) -> BatchResult:
        """Load batch result from CSV."""
        try:
            df = pd.read_csv(file_path)
            
            # Reconstruct BatchResult object from DataFrame
            # Note: This is a simplified reconstruction that may not preserve
            # all original object relationships
            results = []
            for _, row in df.iterrows():
                results.append(
                    PromptResult(
                        prompt_template=row["template_name"],
                        input_text=row["input_text"],
                        output_text=row["output_text"],
                        metadata={
                            k: v for k, v in row.items()
                            if k not in ["template_name", "input_text", "output_text", 
                                       "execution_time", "timestamp"]
                        },
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        execution_time=row["execution_time"]
                    )
                )
            
            return BatchResult(
                prompt_results=results,
                metadata={},  # CSV format doesn't preserve batch metadata
                start_time=datetime.fromisoformat(df["timestamp"].min()),
                end_time=datetime.fromisoformat(df["timestamp"].max())
            )
            
        except Exception as e:
            raise LeekyError(f"Failed to load CSV: {str(e)}")

    def _generate_pdf_report(
        self,
        result: BatchResult,
        metrics: Dict[str, List[MetricResult]],
        output_path: Path
    ) -> None:
        """Generate PDF report with results and visualizations."""
        try:
            # Create figures for metrics
            plt.figure(figsize=(10, 6))
            for metric_name, metric_results in metrics.items():
                scores = [r.score for r in metric_results]
                plt.hist(scores, alpha=0.5, label=metric_name)
            
            plt.title("Distribution of Metric Scores")
            plt.xlabel("Score")
            plt.ylabel("Frequency")
            plt.legend()
            
            # Save plot to PDF
            plt.savefig(output_path)
            plt.close()
            
        except Exception as e:
            raise LeekyError(f"Failed to generate PDF report: {str(e)}")

    def analyze_results(
        self,
        result: BatchResult,
        metrics: Dict[str, List[MetricResult]]
    ) -> Dict[str, Any]:
        """Perform analysis on results.
        
        Args:
            result: BatchResult to analyze.
            metrics: Dictionary of metric results.
            
        Returns:
            Dictionary containing analysis results.
            
        Raises:
            LeekyError: If analysis fails.
        """
        try:
            analysis = {
                "summary": {
                    "total_tests": len(result.prompt_results),
                    "total_time": (result.end_time - result.start_time).total_seconds(),
                    "avg_execution_time": statistics.mean(
                        r.execution_time for r in result.prompt_results
                    )
                },
                "metrics": {}
            }
            
            # Analyze each metric
            for metric_name, metric_results in metrics.items():
                scores = [r.score for r in metric_results]
                analysis["metrics"][metric_name] = {
                    "mean": statistics.mean(scores),
                    "median": statistics.median(scores),
                    "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "min": min(scores),
                    "max": max(scores)
                }
            
            return analysis
            
        except Exception as e:
            raise LeekyError(f"Failed to analyze results: {str(e)}")
