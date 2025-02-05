"""OpenAI completion engine implementation."""

import openai
from typing import Any, Dict, AsyncIterator

from .base_engine import CompletionEngine


class OpenAIEngine(CompletionEngine):
    """OpenAI completion engine implementation."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """Initialize the OpenAI engine.
        
        Args:
            api_key: OpenAI API key.
            model: Model to use for completions (default: gpt-3.5-turbo).
        """
        openai.api_key = api_key
        self.client = openai.AsyncClient()
        self._model = model
        self._max_tokens = {
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "gpt-4-32k": 32768
        }.get(model, 4096)

    def _format_prompt_with_length_guidance(self, prompt: str, max_tokens: int) -> str:
        """Add length guidance to the prompt.
        
        Args:
            prompt: Original prompt
            max_tokens: Target number of tokens
            
        Returns:
            Modified prompt with length guidance
        """
        guidance = f"\n\nPlease provide a response that is approximately {max_tokens} words in length."
        return prompt + guidance

    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion using OpenAI API.
        
        Args:
            prompt: Input prompt.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            str: Generated completion.
        """
        params = {**self.default_params, **kwargs}
        
        # Add length guidance if max_tokens is specified
        if params.get("max_tokens"):
            prompt = self._format_prompt_with_length_guidance(prompt, params["max_tokens"])
            
        response = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self._model,
            **params
        )
        return response.choices[0].message.content

    @property
    def name(self) -> str:
        """Get engine name."""
        return f"openai-{self._model}"

    @property
    def max_tokens(self) -> int:
        """Get maximum token limit."""
        return self._max_tokens

    @property
    def default_params(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {
            "temperature": 0.7,
            "max_tokens": None,  # Will be set based on completion portion length
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
