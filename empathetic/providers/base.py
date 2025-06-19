from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class ModelResponse:
    """Standardized response from a model provider"""
    content: str
    metadata: Dict[str, Any]
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None

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
    ) -> List[float]:
        """Get text embeddings"""
        pass
        
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate provider configuration"""
        pass
        
    def get_model_info(self) -> Dict[str, Any]:
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