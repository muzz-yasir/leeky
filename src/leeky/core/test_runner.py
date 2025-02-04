"""Test execution functionality with async support."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Sequence
from datetime import datetime
import random

from .types import (
    TextSource,
    PromptTemplate,
    PromptResult,
    BatchResult,
    LeekyError
)
from .prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class CompletionEngine:
    """Base class for completion engines."""
    
    async def complete(
        self,
        prompt: str,
        **kwargs: Any
    ) -> str:
        """Generate completion for a prompt.
        
        Args:
            prompt: The prompt to complete.
            **kwargs: Additional engine-specific parameters.
            
        Returns:
            Generated completion text.
            
        Raises:
            NotImplementedError: This is a base class.
        """
        raise NotImplementedError("Completion engines must implement complete()")

class TestRunner:
    """Handles execution of prompt tests with async support."""
    
    def __init__(
        self,
        prompt_manager: PromptManager,
        engine: CompletionEngine,
        config: Dict[str, Any]
    ):
        """Initialize the test runner.
        
        Args:
            prompt_manager: PromptManager instance for template management.
            engine: CompletionEngine instance for generating completions.
            config: Configuration dictionary.
        """
        self.prompt_manager = prompt_manager
        self.engine = engine
        self.batch_size = config.get("batch_size", 10)
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 30.0)
        self.sampling_strategy = config.get("sampling_strategy", "random")
        self.sampling_parameters = config.get("sampling_parameters", {})

    async def run_single(
        self,
        text: TextSource,
        template: PromptTemplate,
        **kwargs: Any
    ) -> PromptResult:
        """Run a single prompt test.
        
        Args:
            text: Input text source.
            template: Prompt template to use.
            **kwargs: Additional parameters for the completion engine.
            
        Returns:
            PromptResult containing the test results.
            
        Raises:
            LeekyError: If test execution fails.
        """
        start_time = time.time()
        
        try:
            # Format prompt with text and parameters
            prompt = template.template.format(
                text=text.content,
                **template.parameters,
                **kwargs
            )
            
            # Generate completion with retries
            for attempt in range(self.max_retries):
                try:
                    output = await asyncio.wait_for(
                        self.engine.complete(prompt, **kwargs),
                        timeout=self.timeout
                    )
                    break
                except asyncio.TimeoutError:
                    if attempt == self.max_retries - 1:
                        raise LeekyError("Test execution timed out")
                    logger.warning(f"Attempt {attempt + 1} timed out, retrying...")
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise LeekyError(f"Test execution failed: {str(e)}")
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}, retrying...")
            
            execution_time = time.time() - start_time
            
            return PromptResult(
                prompt_template=template,
                input_text=text,
                output_text=output,
                metadata={
                    "attempts": attempt + 1,
                    "execution_time": execution_time,
                    **kwargs
                },
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
        except Exception as e:
            raise LeekyError(f"Failed to run test: {str(e)}")

    async def run_batch(
        self,
        texts: Sequence[TextSource],
        templates: Sequence[PromptTemplate],
        **kwargs: Any
    ) -> BatchResult:
        """Run a batch of prompt tests.
        
        Args:
            texts: Input text sources.
            templates: Prompt templates to use.
            **kwargs: Additional parameters for the completion engine.
            
        Returns:
            BatchResult containing all test results.
            
        Raises:
            LeekyError: If batch execution fails.
        """
        start_time = datetime.now()
        
        try:
            # Sample texts if needed
            if self.sampling_strategy == "random":
                sample_size = self.sampling_parameters.get("size", len(texts))
                texts = random.sample(list(texts), min(sample_size, len(texts)))
            
            # Create all test combinations
            tests = [
                (text, template)
                for text in texts
                for template in templates
            ]
            
            # Run tests in batches
            results = []
            for i in range(0, len(tests), self.batch_size):
                batch = tests[i:i + self.batch_size]
                batch_results = await asyncio.gather(*[
                    self.run_single(text, template, **kwargs)
                    for text, template in batch
                ])
                results.extend(batch_results)
                
                # Log progress
                logger.info(f"Completed {len(results)}/{len(tests)} tests")
            
            return BatchResult(
                prompt_results=results,
                metadata={
                    "total_tests": len(tests),
                    "sampling_strategy": self.sampling_strategy,
                    "sampling_parameters": self.sampling_parameters,
                    **kwargs
                },
                start_time=start_time,
                end_time=datetime.now()
            )
            
        except Exception as e:
            raise LeekyError(f"Failed to run batch: {str(e)}")

    def _sample_texts(
        self,
        texts: Sequence[TextSource],
        strategy: str,
        parameters: Dict[str, Any]
    ) -> List[TextSource]:
        """Sample texts using the specified strategy.
        
        Args:
            texts: Input text sources to sample from.
            strategy: Sampling strategy to use.
            parameters: Strategy-specific parameters.
            
        Returns:
            List of sampled TextSource objects.
            
        Raises:
            LeekyError: If sampling strategy is invalid.
        """
        if strategy == "random":
            sample_size = parameters.get("size", len(texts))
            return random.sample(list(texts), min(sample_size, len(texts)))
        elif strategy == "first":
            sample_size = parameters.get("size", len(texts))
            return list(texts)[:sample_size]
        elif strategy == "all":
            return list(texts)
        else:
            raise LeekyError(f"Unknown sampling strategy: {strategy}")
