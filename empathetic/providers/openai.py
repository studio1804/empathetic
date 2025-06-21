import json
import time
from typing import Any, Optional

import httpx

from ..config import config
from ..utils.logging import get_logger
from .base import APIError, ConfigurationError, ModelProvider, ModelResponse


class OpenAIProvider(ModelProvider):
    """OpenAI model provider"""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or config.get_api_key("openai")
        self.base_url = kwargs.get("base_url", "https://api.openai.com/v1")
        self.logger = get_logger("empathetic.providers.openai")

        if not self.api_key:
            raise ConfigurationError("OpenAI API key not provided")

        self.logger.debug(f"Initialized OpenAI provider for model: {model_name}")

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=httpx.Timeout(30.0, connect=10.0, read=30.0, write=10.0)
        )

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using OpenAI API"""
        start_time = time.time()

        self.logger.debug("Generating response with OpenAI", extra={
            "model": self.model_name,
            "prompt_length": len(prompt),
            "temperature": kwargs.get("temperature", 0.7)
        })

        try:
            # Handle system prompt if provided
            messages = []
            if kwargs.get("system_prompt"):
                messages.append({"role": "system", "content": kwargs["system_prompt"]})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000),
                **{k: v for k, v in kwargs.items()
                   if k in ["top_p", "frequency_penalty", "presence_penalty"]}
            }

            response = await self.client.post(
                "/chat/completions",
                json=payload
            )

            if response.status_code != 200:
                raise APIError(f"OpenAI API error: {response.status_code} - {response.text}")

            data = response.json()

            response_content = data["choices"][0]["message"]["content"]

            self.logger.debug("OpenAI response received", extra={
                "response_length": len(response_content),
                "finish_reason": data["choices"][0]["finish_reason"],
                "usage": data.get("usage", {})
            })

            latency = time.time() - start_time

            return ModelResponse(
                content=response_content,
                metadata={
                    "finish_reason": data["choices"][0]["finish_reason"],
                    "created": data["created"],
                    "id": data["id"]
                },
                usage=data.get("usage"),
                model=data["model"],
                latency=latency
            )

        except httpx.RequestError as e:
            self.logger.error(f"OpenAI request failed: {str(e)}")
            raise APIError(f"Request failed: {str(e)}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON response from OpenAI: {str(e)}")
            raise APIError(f"Invalid JSON response: {str(e)}")
        except KeyError as e:
            self.logger.error(f"Unexpected OpenAI response format: missing {str(e)}")
            raise APIError(f"Unexpected response format: missing {str(e)}")

    async def get_embeddings(self, text: str) -> list[float]:
        """Get text embeddings using OpenAI embeddings API"""
        try:
            payload = {
                "model": "text-embedding-ada-002",
                "input": text
            }

            response = await self.client.post(
                "/embeddings",
                json=payload
            )

            if response.status_code != 200:
                raise APIError(f"OpenAI embeddings API error: {response.status_code}")

            data = response.json()
            return data["data"][0]["embedding"]

        except httpx.RequestError as e:
            raise APIError(f"Embeddings request failed: {str(e)}")
        except (json.JSONDecodeError, KeyError) as e:
            raise APIError(f"Invalid embeddings response: {str(e)}")

    def validate_config(self) -> bool:
        """Validate OpenAI configuration"""
        return bool(self.api_key and self.model_name)

    async def detect_capabilities(self):
        """Detect OpenAI-specific model capabilities"""
        from ..models.capabilities import CapabilityDetector

        # Use base detection but add OpenAI-specific info
        detector = CapabilityDetector()
        capabilities = await detector.detect_capabilities(self, quick_mode=True)

        # Add OpenAI-specific capability overrides
        openai_capabilities = self._get_openai_model_capabilities()

        # Update capabilities with known OpenAI model specs
        if openai_capabilities:
            capabilities.context_length = openai_capabilities.get('context_length', capabilities.context_length)
            capabilities.supports_system_prompt = openai_capabilities.get('supports_system_prompt', True)
            capabilities.supports_json_output = openai_capabilities.get('supports_json_output', True)
            capabilities.supports_function_calling = openai_capabilities.get('supports_function_calling', False)

        self.logger.info("OpenAI model capabilities detected", extra={
            "model": self.model_name,
            "context_length": capabilities.context_length,
            "empathy_baseline": capabilities.empathy_baseline,
            "bias_susceptibility": capabilities.bias_susceptibility
        })

        return capabilities

    def _get_openai_model_capabilities(self) -> Optional[dict[str, Any]]:
        """Get known capabilities for specific OpenAI models"""
        model_specs = {
            "gpt-4": {
                "context_length": 8192,
                "supports_system_prompt": True,
                "supports_json_output": True,
                "supports_function_calling": True,
                "expected_empathy_baseline": 0.75
            },
            "gpt-4-turbo": {
                "context_length": 128000,
                "supports_system_prompt": True,
                "supports_json_output": True,
                "supports_function_calling": True,
                "expected_empathy_baseline": 0.78
            },
            "gpt-4-turbo-preview": {
                "context_length": 128000,
                "supports_system_prompt": True,
                "supports_json_output": True,
                "supports_function_calling": True,
                "expected_empathy_baseline": 0.78
            },
            "gpt-3.5-turbo": {
                "context_length": 16385,
                "supports_system_prompt": True,
                "supports_json_output": True,
                "supports_function_calling": True,
                "expected_empathy_baseline": 0.65
            },
            "gpt-3.5-turbo-16k": {
                "context_length": 16385,
                "supports_system_prompt": True,
                "supports_json_output": True,
                "supports_function_calling": True,
                "expected_empathy_baseline": 0.65
            }
        }

        return model_specs.get(self.model_name)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
