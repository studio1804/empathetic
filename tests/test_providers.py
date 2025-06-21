import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from empathetic.providers.anthropic import AnthropicProvider
from empathetic.providers.base import (
    APIError,
    ConfigurationError,
    ModelProvider,
    ModelResponse,
)
from empathetic.providers.factory import ProviderFactory, create_provider
from empathetic.providers.openai import OpenAIProvider


class TestProviderFactory:
    """Test provider factory functionality"""

    def test_factory_initialization(self):
        """Test that factory initializes with default providers"""
        factory = ProviderFactory()
        available = factory.get_available_providers()
        assert 'openai' in available
        assert 'anthropic' in available

    def test_model_detection_openai(self):
        """Test OpenAI model detection"""
        factory = ProviderFactory()

        openai_models = ['gpt-4', 'gpt-3.5-turbo', 'text-davinci-003']
        for model in openai_models:
            detected = factory._detect_provider(model)
            assert detected == 'openai'

    def test_model_detection_anthropic(self):
        """Test Anthropic model detection"""
        factory = ProviderFactory()

        anthropic_models = ['claude-3-opus', 'claude-3-sonnet', 'claude-2']
        for model in anthropic_models:
            detected = factory._detect_provider(model)
            assert detected == 'anthropic'

    def test_model_detection_huggingface(self):
        """Test HuggingFace model detection"""
        factory = ProviderFactory()

        hf_models = ['microsoft/DialoGPT-medium', 'facebook/blenderbot-400M']
        for model in hf_models:
            detected = factory._detect_provider(model)
            assert detected == 'huggingface'

    def test_create_provider_explicit(self):
        """Test creating provider with explicit type"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = create_provider('test-model', provider_type='openai')
            assert isinstance(provider, OpenAIProvider)

    def test_create_provider_auto_detect(self):
        """Test creating provider with auto-detection"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = create_provider('gpt-4')
            assert isinstance(provider, OpenAIProvider)

        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = create_provider('claude-3-opus')
            assert isinstance(provider, AnthropicProvider)

    def test_unknown_provider_type(self):
        """Test error handling for unknown provider type"""
        with pytest.raises(ValueError, match="Unknown provider type"):
            create_provider('test-model', provider_type='unknown')

    def test_register_custom_provider(self):
        """Test registering custom provider"""
        factory = ProviderFactory()

        class CustomProvider(ModelProvider):
            def __init__(self, model_name, **kwargs):
                super().__init__(model_name, **kwargs)

            async def generate(self, prompt, **kwargs):
                return ModelResponse("test", {})

            async def get_embeddings(self, text):
                return [0.1, 0.2, 0.3]

            def validate_config(self):
                return True

        factory.register_provider('custom', CustomProvider)
        assert 'custom' in factory.get_available_providers()

        provider = factory.create_provider('test', provider_type='custom')
        assert isinstance(provider, CustomProvider)

class TestOpenAIProvider:
    """Test OpenAI provider functionality"""

    @pytest.mark.asyncio
    async def test_successful_generation(self):
        """Test successful response generation"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIProvider('gpt-3.5-turbo')

            # Mock the HTTP client
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{
                    "message": {"content": "Test response"},
                    "finish_reason": "stop"
                }],
                "created": 1234567890,
                "id": "test-id",
                "model": "gpt-3.5-turbo",
                "usage": {"total_tokens": 10}
            }

            provider.client.post = AsyncMock(return_value=mock_response)

            response = await provider.generate("Test prompt")

            assert isinstance(response, ModelResponse)
            assert response.content == "Test response"
            assert response.model == "gpt-3.5-turbo"
            assert response.usage["total_tokens"] == 10

    @pytest.mark.asyncio
    async def test_api_error_handling(self):
        """Test API error handling"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIProvider('gpt-3.5-turbo')

            # Mock HTTP error
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.text = "Rate limit exceeded"

            provider.client.post = AsyncMock(return_value=mock_response)

            with pytest.raises(APIError, match="OpenAI API error: 429"):
                await provider.generate("Test prompt")

    def test_missing_api_key(self):
        """Test error when API key is missing"""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ConfigurationError, match="OpenAI API key not provided"):
                OpenAIProvider('gpt-3.5-turbo')

    def test_config_validation(self):
        """Test configuration validation"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIProvider('gpt-3.5-turbo')
            assert provider.validate_config() is True

        # Test with invalid config
        provider.api_key = None
        assert provider.validate_config() is False

class TestAnthropicProvider:
    """Test Anthropic provider functionality"""

    @pytest.mark.asyncio
    async def test_successful_generation(self):
        """Test successful response generation"""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = AnthropicProvider('claude-3-opus')

            # Mock the HTTP client
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "content": [{"text": "Test response"}],
                "stop_reason": "end_turn",
                "id": "test-id",
                "type": "message",
                "model": "claude-3-opus",
                "usage": {"input_tokens": 5, "output_tokens": 10}
            }

            provider.client.post = AsyncMock(return_value=mock_response)

            response = await provider.generate("Test prompt")

            assert isinstance(response, ModelResponse)
            assert response.content == "Test response"
            assert response.model == "claude-3-opus"

    def test_missing_api_key(self):
        """Test error when API key is missing"""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ConfigurationError, match="Anthropic API key not provided"):
                AnthropicProvider('claude-3-opus')

    @pytest.mark.asyncio
    async def test_embeddings_not_implemented(self):
        """Test that embeddings raises NotImplementedError"""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = AnthropicProvider('claude-3-opus')

            with pytest.raises(NotImplementedError, match="Anthropic does not provide embeddings"):
                await provider.get_embeddings("test text")

class TestEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.asyncio
    async def test_empty_prompt(self):
        """Test handling of empty prompts"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIProvider('gpt-3.5-turbo')

            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
                "created": 1234567890,
                "id": "test-id",
                "model": "gpt-3.5-turbo"
            }

            provider.client.post = AsyncMock(return_value=mock_response)

            response = await provider.generate("")
            assert response.content == ""

    @pytest.mark.asyncio
    async def test_very_long_prompt(self):
        """Test handling of very long prompts"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIProvider('gpt-3.5-turbo')

            long_prompt = "x" * 10000  # Very long prompt

            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Response"}, "finish_reason": "stop"}],
                "created": 1234567890,
                "id": "test-id",
                "model": "gpt-3.5-turbo"
            }

            provider.client.post = AsyncMock(return_value=mock_response)

            response = await provider.generate(long_prompt)
            assert response.content == "Response"

    @pytest.mark.asyncio
    async def test_network_timeout(self):
        """Test network timeout handling"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIProvider('gpt-3.5-turbo')

            # Mock timeout error
            import httpx
            provider.client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

            with pytest.raises(APIError, match="Request failed"):
                await provider.generate("Test prompt")

    def test_provider_context_manager(self):
        """Test provider context manager functionality"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIProvider('gpt-3.5-turbo')

            # Test async context manager
            async def test_context():
                async with provider:
                    assert provider.client is not None

            asyncio.run(test_context())

    def test_model_info_retrieval(self):
        """Test getting model information"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIProvider('gpt-3.5-turbo')

            info = provider.get_model_info()
            assert info['name'] == 'gpt-3.5-turbo'
            assert info['provider'] == 'OpenAIProvider'
            assert 'config' in info
