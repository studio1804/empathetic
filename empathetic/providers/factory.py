from typing import Dict, Type, Optional
from .base import ModelProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from ..utils.logging import get_logger

class ProviderFactory:
    """Factory for creating model providers based on model names or explicit provider types"""
    
    def __init__(self):
        self.logger = get_logger("empathetic.providers.factory")
        self._providers: Dict[str, Type[ModelProvider]] = {
            'openai': OpenAIProvider,
            'anthropic': AnthropicProvider,
        }
        
        # Model name patterns for automatic detection
        self._model_patterns = {
            'openai': ['gpt-', 'text-', 'davinci', 'curie', 'babbage', 'ada'],
            'anthropic': ['claude-'],
            'huggingface': ['/'],  # HuggingFace models typically have format 'org/model'
        }
        
    def register_provider(self, name: str, provider_class: Type[ModelProvider]):
        """Register a new provider type"""
        self._providers[name] = provider_class
        self.logger.debug(f"Registered provider: {name}")
        
    def create_provider(
        self, 
        model: str, 
        provider_type: Optional[str] = None,
        **kwargs
    ) -> ModelProvider:
        """Create a provider instance for the given model"""
        
        # If provider type is explicitly specified, use it
        if provider_type:
            if provider_type not in self._providers:
                raise ValueError(f"Unknown provider type: {provider_type}")
            
            provider_class = self._providers[provider_type]
            self.logger.debug(f"Creating {provider_type} provider for model: {model}")
            return provider_class(model, **kwargs)
        
        # Auto-detect provider based on model name
        detected_provider = self._detect_provider(model)
        
        if detected_provider not in self._providers:
            # Default to OpenAI for unknown models
            self.logger.warning(f"Could not detect provider for model '{model}', defaulting to OpenAI")
            detected_provider = 'openai'
            
        provider_class = self._providers[detected_provider]
        self.logger.debug(f"Auto-detected {detected_provider} provider for model: {model}")
        
        return provider_class(model, **kwargs)
        
    def _detect_provider(self, model: str) -> str:
        """Detect provider based on model name patterns"""
        model_lower = model.lower()
        
        for provider, patterns in self._model_patterns.items():
            for pattern in patterns:
                if model_lower.startswith(pattern) or pattern in model_lower:
                    return provider
                    
        return 'openai'  # Default fallback
        
    def get_available_providers(self) -> list[str]:
        """Get list of available provider types"""
        return list(self._providers.keys())
        
    def get_provider_info(self, provider_type: str) -> Dict[str, str]:
        """Get information about a specific provider"""
        if provider_type not in self._providers:
            raise ValueError(f"Unknown provider type: {provider_type}")
            
        provider_class = self._providers[provider_type]
        return {
            'name': provider_type,
            'class': provider_class.__name__,
            'module': provider_class.__module__,
            'description': provider_class.__doc__ or "No description available"
        }
        
    def validate_model_for_provider(self, model: str, provider_type: str) -> bool:
        """Validate if a model is compatible with a specific provider"""
        if provider_type not in self._model_patterns:
            return True  # Assume compatible if no patterns defined
            
        patterns = self._model_patterns[provider_type]
        model_lower = model.lower()
        
        return any(
            model_lower.startswith(pattern) or pattern in model_lower 
            for pattern in patterns
        )

# Global factory instance
provider_factory = ProviderFactory()

def create_provider(model: str, provider_type: Optional[str] = None, **kwargs) -> ModelProvider:
    """Convenience function to create a provider"""
    return provider_factory.create_provider(model, provider_type, **kwargs)

def register_provider(name: str, provider_class: Type[ModelProvider]):
    """Convenience function to register a new provider"""
    provider_factory.register_provider(name, provider_class)

def get_available_providers() -> list[str]:
    """Get list of available providers"""
    return provider_factory.get_available_providers()