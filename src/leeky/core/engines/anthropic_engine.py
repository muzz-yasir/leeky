"""Anthropic completion engine implementation."""

import anthropic
from typing import Any, Dict, AsyncIterator

from .base_engine import CompletionEngine


class AnthropicEngine(CompletionEngine):
    """Anthropic completion engine implementation."""
    
    def __init__(self, api_key: str, model: str = "claude-2"):
        """Initialize the Anthropic engine.
        
        Args:
            api_key: Anthropic API key.
            model: Model to use for completions (default: claude-2).
        """
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = {
            "claude-2": 100000,
            "claude-instant-1": 100000
        }.get(model, 100000)

    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion using Anthropic API.
        
        Args:
            prompt: Input prompt.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            str: Generated completion.
        """
        params = {**self.default_params, **kwargs}
        response = await self.client.messages.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **params
        )
        return response.content[0].text

    @property
    def name(self) -> str:
        """Get engine name."""
        return f"anthropic-{self._model}"

    @property
    def max_tokens(self) -> int:
        """Get maximum token limit."""
        return self._max_tokens

    @property
    def supports_streaming(self) -> bool:
        """Check if streaming is supported."""
        return True

    @property
    def default_params(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {
            "temperature": 0.7,
            "max_tokens": None,  # Let Anthropic handle token limit
            "top_p": 1.0
        }
