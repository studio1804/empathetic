from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ModelResponse:
    """Standardized response from a model provider"""
    content: str
    metadata: dict[str, Any]
    usage: Optional[dict[str, int]] = None
    model: Optional[str] = None
    latency: float = 0.0  # Response time in seconds

    @property
    def text(self) -> str:
        """Alias for content for backward compatibility"""
        return self.content

class ModelProvider(ABC):
    """Base class for model providers"""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        **kwargs
    ) -> ModelResponse:
        """Generate response from model"""
        pass

    @abstractmethod
    async def get_embeddings(
        self,
        text: str
    ) -> list[float]:
        """Get text embeddings"""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate provider configuration"""
        pass

    async def detect_capabilities(self):
        """Detect model capabilities - override in subclasses for provider-specific detection"""
        from ..models.capabilities import CapabilityDetector

        detector = CapabilityDetector()
        return await detector.detect_capabilities(self, quick_mode=True)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the model"""
        return {
            "name": self.model_name,
            "provider": self.__class__.__name__,
            "config": self.config
        }

class ProviderError(Exception):
    """Base exception for provider errors"""
    pass

class ConfigurationError(ProviderError):
    """Raised when provider configuration is invalid"""
    pass

class APIError(ProviderError):
    """Raised when API calls fail"""
    pass
