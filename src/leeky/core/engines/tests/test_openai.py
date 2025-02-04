"""Tests for OpenAI completion engine."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from .test_base import BaseEngineTest
from ..openai_engine import OpenAIEngine


class TestOpenAIEngine(BaseEngineTest):
    """Test OpenAI engine implementation."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock OpenAI client."""
        with patch("openai.AsyncClient") as mock:
            client = AsyncMock()
            mock.return_value = client
            
            # Mock chat completion response
            completion = MagicMock()
            completion.choices = [MagicMock(message=MagicMock(content="Test response"))]
            client.chat.completions.create.return_value = completion
            
            yield client
    
    @pytest.fixture
    def engine(self, mock_client):
        """Return OpenAI engine instance with mocked client."""
        with patch("openai.api_key", "test-key"):
            engine = OpenAIEngine("test-key")
            return engine
    
    @pytest.mark.asyncio
    async def test_complete_uses_correct_model(self, engine, mock_client):
        """Test that complete uses the configured model."""
        await engine.complete("Test prompt")
        
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-3.5-turbo"
    
    @pytest.mark.asyncio
    async def test_complete_formats_messages(self, engine, mock_client):
        """Test that complete properly formats the messages."""
        prompt = "Test prompt"
        await engine.complete(prompt)
        
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": prompt}]
    
    def test_name_includes_model(self, engine):
        """Test that name includes the model identifier."""
        assert engine.name == "openai-gpt-3.5-turbo"
    
    def test_max_tokens_per_model(self):
        """Test max tokens for different models."""
        models = {
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "gpt-4-32k": 32768
        }
        
        for model, expected_tokens in models.items():
            with patch("openai.api_key", "test-key"):
                engine = OpenAIEngine("test-key", model=model)
                assert engine.max_tokens == expected_tokens
    
    def test_default_params(self, engine):
        """Test default parameters are properly set."""
        params = engine.default_params
        assert params["temperature"] == 0.7
        assert params["max_tokens"] is None
        assert params["top_p"] == 1.0
        assert params["frequency_penalty"] == 0.0
        assert params["presence_penalty"] == 0.0
