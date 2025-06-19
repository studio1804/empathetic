import os
from typing import List, Dict, Any, Optional
import httpx
import json
from .base import ModelProvider, ModelResponse, APIError, ConfigurationError

class AnthropicProvider(ModelProvider):
    """Anthropic model provider"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
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
        
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using Anthropic API"""
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
            raise APIError(f"Request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise APIError(f"Invalid JSON response: {str(e)}")
        except KeyError as e:
            raise APIError(f"Unexpected response format: missing {str(e)}")
            
    async def get_embeddings(self, text: str) -> List[float]:
        """Get text embeddings - Note: Anthropic doesn't provide embeddings API"""
        raise NotImplementedError("Anthropic does not provide embeddings API")
            
    def validate_config(self) -> bool:
        """Validate Anthropic configuration"""
        return bool(self.api_key and self.model_name)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()