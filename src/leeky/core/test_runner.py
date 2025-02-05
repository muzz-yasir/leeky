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
    LeekyError,
    TestConfig
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
        config: TestConfig
    ):
        """Initialize the test runner.
        
        Args:
            prompt_manager: PromptManager instance for template management.
            engine: CompletionEngine instance for generating completions.
            config: Configuration dictionary.
        """
        """Initialize the test runner.
        
        Args:
            prompt_manager: PromptManager instance for template management.
            engine: CompletionEngine instance for generating completions.
            config: Configuration dictionary.
        """
        self.prompt_manager = prompt_manager
        self.engine = engine
        self.batch_size = config.batch_size
        self.max_retries = config.max_retries
        self.timeout = config.timeout
        self.sampling_strategy = config.sampling_strategy
        self.sampling_parameters = config.sampling_parameters
        
        # Text splitting configuration
        self.text_splitting = config.text_splitting if config.text_splitting else {
            "enabled": True,
            "strategy": "sentence",
            "split_ratio": 0.8
        }

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
            # Split text if enabled
            if self.text_splitting["enabled"]:
                text.split_for_completion(
                    split_ratio=self.text_splitting["split_ratio"],
                    strategy=self.text_splitting["strategy"]
                )
                input_content = text.context_portion
                
                # Estimate completion token count
                completion_token_count = len(text.completion_portion.split())
                kwargs["max_tokens"] = completion_token_count
            else:
                input_content = text.content

            # Format prompt with text and parameters
            params_dict = {param: kwargs.get(param) for param in template.parameters}
            prompt = template.template.format(
                text=input_content,
                **params_dict
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
            
            # Create a dictionary representation of the template with datetime converted to ISO format
            template_dict = {
                "name": template.name,
                "version": template.version,
                "template": template.template,
                "parameters": template.parameters,
                "metadata": template.metadata,
                "created_at": template.created_at.isoformat() if template.created_at else None,
                "template_type": template.template_type.value
            }
            
            return PromptResult(
                prompt_template=template_dict,
                prompt_string=str(prompt),
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
