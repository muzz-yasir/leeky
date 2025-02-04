"""Abstract base class for completion engines."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, AsyncIterator


class CompletionEngine(ABC):
    """Abstract base class for completion engines."""
    
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion for the given prompt.
        
        Args:
            prompt: The input prompt to complete.
            **kwargs: Additional engine-specific parameters.
            
        Returns:
            str: The generated completion text.
            
        Raises:
            Exception: If completion generation fails.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the completion engine.
        
        Returns:
            str: The engine name.
        """
        pass

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Get the maximum number of tokens supported by the engine.
        
        Returns:
            int: Maximum token limit.
        """
        pass

    @property
    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """Get default parameters for completion requests.
        
        Returns:
            Dict[str, Any]: Default parameters.
        """
        pass
