import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

class ValidationError(Exception):
    """Raised when validation fails"""
    pass

class InputValidator:
    """Validate various inputs for Empathetic"""
    
    @staticmethod
    def validate_model_name(model_name: str) -> bool:
        """Validate model name format"""
        if not model_name or not isinstance(model_name, str):
            raise ValidationError("Model name must be a non-empty string")
            
        # Basic validation - model names should be reasonable
        if len(model_name) > 100:
            raise ValidationError("Model name too long")
            
        # Check for basic format (alphanumeric, hyphens, underscores, dots)
        if not re.match(r'^[a-zA-Z0-9._-]+$', model_name):
            raise ValidationError("Model name contains invalid characters")
            
        return True
        
    @staticmethod
    def validate_suite_names(suite_names: List[str]) -> bool:
        """Validate test suite names"""
        valid_suites = {'bias', 'alignment', 'fairness', 'safety', 'all'}
        
        for suite in suite_names:
            if suite not in valid_suites:
                raise ValidationError(f"Invalid test suite: {suite}")
                
        return True
        
    @staticmethod
    def validate_threshold(threshold: float) -> bool:
        """Validate threshold value"""
        if not isinstance(threshold, (int, float)):
            raise ValidationError("Threshold must be a number")
            
        if not 0.0 <= threshold <= 1.0:
            raise ValidationError("Threshold must be between 0.0 and 1.0")
            
        return True
        
    @staticmethod
    def validate_output_format(format_name: str) -> bool:
        """Validate output format"""
        valid_formats = {'terminal', 'json', 'html', 'markdown'}
        
        if format_name.lower() not in valid_formats:
            raise ValidationError(f"Invalid output format: {format_name}")
            
        return True
        
    @staticmethod
    def validate_api_key(api_key: str, provider: str) -> bool:
        """Validate API key format"""
        if not api_key or not isinstance(api_key, str):
            raise ValidationError(f"{provider} API key must be a non-empty string")
            
        # Basic format validation
        if provider.lower() == 'openai':
            if not api_key.startswith('sk-'):
                raise ValidationError("OpenAI API key should start with 'sk-'")
        elif provider.lower() == 'anthropic':
            if not api_key.startswith('sk-ant-'):
                raise ValidationError("Anthropic API key should start with 'sk-ant-'")
                
        return True
        
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """Validate file path"""
        if not file_path or not isinstance(file_path, str):
            raise ValidationError("File path must be a non-empty string")
            
        # Check for directory traversal attempts (more comprehensive)
        import os.path
        normalized_path = os.path.normpath(file_path)
        if '..' in normalized_path or normalized_path.startswith('/'):
            raise ValidationError("File path contains invalid directory traversal")
            
        return True
        
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                raise ValidationError("Invalid URL format")
            return True
        except Exception:
            raise ValidationError("Invalid URL format")
            
    @staticmethod
    def validate_test_case(test_case: Dict[str, Any]) -> bool:
        """Validate test case structure"""
        required_fields = ['id', 'category', 'input', 'expected_behavior', 'harmful_patterns', 'severity']
        
        for field in required_fields:
            if field not in test_case:
                raise ValidationError(f"Test case missing required field: {field}")
                
        # Validate specific fields
        if not isinstance(test_case['harmful_patterns'], list):
            raise ValidationError("harmful_patterns must be a list")
            
        if test_case['severity'] not in ['low', 'medium', 'high', 'critical']:
            raise ValidationError("Invalid severity level")
            
        return True
        
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate configuration structure"""
        required_sections = ['test_suites', 'scoring']
        
        for section in required_sections:
            if section not in config:
                raise ValidationError(f"Config missing required section: {section}")
                
        # Validate scoring weights
        if 'weights' in config['scoring']:
            weights = config['scoring']['weights']
            if not isinstance(weights, dict):
                raise ValidationError("Scoring weights must be a dictionary")
                
            # Check if weights sum to 1.0 (with tolerance)
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                raise ValidationError(f"Scoring weights should sum to 1.0, got {total_weight}")
                
        return True

class ResponseValidator:
    """Validate model responses"""
    
    @staticmethod
    def validate_response_format(response: Dict[str, Any]) -> bool:
        """Validate response structure from model provider"""
        required_fields = ['content']
        
        for field in required_fields:
            if field not in response:
                raise ValidationError(f"Response missing required field: {field}")
                
        if not isinstance(response['content'], str):
            raise ValidationError("Response content must be a string")
            
        return True
        
    @staticmethod
    def validate_response_safety(response: str) -> bool:
        """Basic safety validation of response content"""
        # Check for extremely long responses that might indicate an issue
        if len(response) > 50000:
            raise ValidationError("Response too long, possible safety issue")
            
        # Could add more sophisticated safety checks here
        return True