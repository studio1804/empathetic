import asyncio
import json
import time
from typing import Optional

import httpx

from ..config import config
from .base import APIError, ConfigurationError, ModelProvider, ModelResponse


class AnthropicProvider(ModelProvider):
    """Anthropic model provider"""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or config.get_api_key("anthropic")
        self.base_url = kwargs.get("base_url", "https://api.anthropic.com/v1")

        if not self.api_key:
            raise ConfigurationError("Anthropic API key not provided")

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            timeout=30.0
        )

        # Rate limiting
        self.rate_limit_delay = 0.5  # 500ms between requests
        self.last_request_time = 0

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using Anthropic API with rate limiting and retry logic"""

        # Rate limiting - ensure proper spacing between requests
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)

        self.last_request_time = time.time()

        # Retry logic for rate limits
        max_retries = 3
        base_delay = 2.0

        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7),
                    **{k: v for k, v in kwargs.items()
                       if k in ["top_p", "top_k", "stop_sequences"]}
                }

                response = await self.client.post(
                    "/messages",
                    json=payload
                )

                if response.status_code == 429:  # Rate limit
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Rate limit hit, waiting {delay}s before retry {attempt + 1}/{max_retries}")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise APIError(f"Rate limit exceeded after {max_retries} retries")

                if response.status_code != 200:
                    raise APIError(f"Anthropic API error: {response.status_code} - {response.text}")

                data = response.json()

                return ModelResponse(
                    content=data["content"][0]["text"],
                    metadata={
                        "stop_reason": data["stop_reason"],
                        "id": data["id"],
                        "type": data["type"]
                    },
                    usage=data.get("usage"),
                    model=data["model"]
                )

            except httpx.RequestError as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    print(f"Request failed, retrying in {delay}s: {str(e)}")
                    await asyncio.sleep(delay)
                    continue
                raise APIError(f"Request failed after {max_retries} retries: {str(e)}")
            except json.JSONDecodeError as e:
                raise APIError(f"Invalid JSON response: {str(e)}")
            except KeyError as e:
                raise APIError(f"Unexpected response format: missing {str(e)}")

    async def get_embeddings(self, text: str) -> list[float]:
        """Get text embeddings - Note: Anthropic doesn't provide embeddings API"""
        raise NotImplementedError("Anthropic does not provide embeddings API")

    def validate_config(self) -> bool:
        """Validate Anthropic configuration"""
        return bool(self.api_key and self.model_name)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
