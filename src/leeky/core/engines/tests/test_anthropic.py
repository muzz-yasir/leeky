"""Tests for Anthropic completion engine."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from .test_base import BaseEngineTest
from ..anthropic_engine import AnthropicEngine


class TestAnthropicEngine(BaseEngineTest):
    """Test Anthropic engine implementation."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock Anthropic client."""
        with patch("anthropic.AsyncAnthropic") as mock:
            client = AsyncMock()
            mock.return_value = client
            
            # Mock messages response
            response = MagicMock()
            response.content = [MagicMock(text="Test response")]
            client.messages.create.return_value = response
            
            yield client
    
    @pytest.fixture
    def engine(self, mock_client):
        """Return Anthropic engine instance with mocked client."""
        engine = AnthropicEngine("test-key")
        return engine
    
    @pytest.mark.asyncio
    async def test_complete_uses_correct_model(self, engine, mock_client):
        """Test that complete uses the configured model."""
        await engine.complete("Test prompt")
        
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-2"
    
    @pytest.mark.asyncio
    async def test_complete_formats_messages(self, engine, mock_client):
        """Test that complete properly formats the messages."""
        prompt = "Test prompt"
        await engine.complete(prompt)
        
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": prompt}]
    
    def test_name_includes_model(self, engine):
        """Test that name includes the model identifier."""
        assert engine.name == "anthropic-claude-2"
    
    def test_max_tokens_per_model(self):
        """Test max tokens for different models."""
        models = {
            "claude-2": 100000,
            "claude-instant-1": 100000
        }
        
        for model, expected_tokens in models.items():
            engine = AnthropicEngine("test-key", model=model)
            assert engine.max_tokens == expected_tokens
    
    def test_default_params(self, engine):
        """Test default parameters are properly set."""
        params = engine.default_params
        assert params["temperature"] == 0.7
        assert params["max_tokens"] is None
        assert params["top_p"] == 1.0
    
