from typing import Dict, Any, List, Optional
import yaml
from pathlib import Path
from .validators import ValidationError

class ConfigValidator:
    """Validate configuration files and settings"""
    
    def __init__(self):
        self.required_sections = ['test_suites', 'scoring']
        self.valid_suite_names = ['bias', 'alignment', 'fairness', 'safety']
        self.valid_providers = ['openai', 'anthropic', 'huggingface']
        
    def validate_config_file(self, config_path: str) -> Dict[str, Any]:
        """Validate configuration file and return validated config"""
        if not Path(config_path).exists():
            raise ValidationError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid YAML syntax in config file: {e}")
        except Exception as e:
            raise ValidationError(f"Error reading config file: {e}")
            
        return self.validate_config_dict(config)
        
    def validate_config_dict(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration dictionary"""
        if not isinstance(config, dict):
            raise ValidationError("Configuration must be a dictionary")
            
        self._validate_required_sections(config)
        self._validate_test_suites(config.get('test_suites', {}))
        self._validate_scoring(config.get('scoring', {}))
        
        if 'providers' in config:
            self._validate_providers(config['providers'])
            
        if 'reporting' in config:
            self._validate_reporting(config['reporting'])
            
        if 'defaults' in config:
            self._validate_defaults(config['defaults'])
            
        return config
        
    def _validate_required_sections(self, config: Dict[str, Any]):
        """Validate that required sections are present"""
        for section in self.required_sections:
            if section not in config:
                raise ValidationError(f"Required configuration section missing: {section}")
                
    def _validate_test_suites(self, test_suites: Dict[str, Any]):
        """Validate test suites configuration"""
        if not isinstance(test_suites, dict):
            raise ValidationError("test_suites must be a dictionary")
            
        for suite_name, suite_config in test_suites.items():
            if suite_name not in self.valid_suite_names:
                raise ValidationError(f"Invalid test suite name: {suite_name}")
                
            if not isinstance(suite_config, dict):
                raise ValidationError(f"Test suite {suite_name} config must be a dictionary")
                
            # Validate suite-specific settings
            if 'enabled' in suite_config:
                if not isinstance(suite_config['enabled'], bool):
                    raise ValidationError(f"enabled setting for {suite_name} must be boolean")
                    
            if 'test_files' in suite_config:
                if not isinstance(suite_config['test_files'], list):
                    raise ValidationError(f"test_files for {suite_name} must be a list")
                    
                for file_path in suite_config['test_files']:
                    if not isinstance(file_path, str):
                        raise ValidationError(f"test_files entries must be strings")
                        
            if 'quick_mode_limit' in suite_config:
                limit = suite_config['quick_mode_limit']
                if not isinstance(limit, int) or limit < 1:
                    raise ValidationError(f"quick_mode_limit for {suite_name} must be positive integer")
                    
    def _validate_scoring(self, scoring: Dict[str, Any]):
        """Validate scoring configuration"""
        if not isinstance(scoring, dict):
            raise ValidationError("scoring must be a dictionary")
            
        # Validate weights
        if 'weights' in scoring:
            weights = scoring['weights']
            if not isinstance(weights, dict):
                raise ValidationError("scoring.weights must be a dictionary")
                
            # Check weight values
            for suite_name, weight in weights.items():
                if suite_name not in self.valid_suite_names:
                    raise ValidationError(f"Invalid suite name in weights: {suite_name}")
                    
                if not isinstance(weight, (int, float)) or weight < 0 or weight > 1:
                    raise ValidationError(f"Weight for {suite_name} must be between 0 and 1")
                    
            # Check that weights sum to approximately 1.0
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                raise ValidationError(f"Scoring weights must sum to 1.0, got {total_weight}")
                
        # Validate thresholds
        if 'thresholds' in scoring:
            thresholds = scoring['thresholds']
            if not isinstance(thresholds, dict):
                raise ValidationError("scoring.thresholds must be a dictionary")
                
            required_thresholds = ['pass', 'warning', 'fail']
            for threshold_name in required_thresholds:
                if threshold_name in thresholds:
                    value = thresholds[threshold_name]
                    if not isinstance(value, (int, float)) or value < 0 or value > 1:
                        raise ValidationError(f"Threshold {threshold_name} must be between 0 and 1")
                        
            # Validate threshold ordering
            if all(t in thresholds for t in required_thresholds):
                if not (thresholds['fail'] <= thresholds['warning'] <= thresholds['pass']):
                    raise ValidationError("Thresholds must be ordered: fail <= warning <= pass")
                    
        # Validate severity weights
        if 'severity_weights' in scoring:
            severity_weights = scoring['severity_weights']
            if not isinstance(severity_weights, dict):
                raise ValidationError("scoring.severity_weights must be a dictionary")
                
            valid_severities = ['low', 'medium', 'high', 'critical']
            for severity, weight in severity_weights.items():
                if severity not in valid_severities:
                    raise ValidationError(f"Invalid severity level: {severity}")
                    
                if not isinstance(weight, (int, float)) or weight < 0:
                    raise ValidationError(f"Severity weight for {severity} must be non-negative")
                    
    def _validate_providers(self, providers: Dict[str, Any]):
        """Validate providers configuration"""
        if not isinstance(providers, dict):
            raise ValidationError("providers must be a dictionary")
            
        for provider_name, provider_config in providers.items():
            if provider_name not in self.valid_providers:
                raise ValidationError(f"Invalid provider name: {provider_name}")
                
            if not isinstance(provider_config, dict):
                raise ValidationError(f"Provider {provider_name} config must be a dictionary")
                
            # Validate models list
            if 'models' in provider_config:
                models = provider_config['models']
                if not isinstance(models, list):
                    raise ValidationError(f"models for {provider_name} must be a list")
                    
                for model in models:
                    if not isinstance(model, str):
                        raise ValidationError("Model names must be strings")
                        
            # Validate default settings
            for setting in ['default_temperature', 'default_max_tokens']:
                if setting in provider_config:
                    value = provider_config[setting]
                    if setting == 'default_temperature':
                        if not isinstance(value, (int, float)) or value < 0 or value > 2:
                            raise ValidationError(f"{setting} must be between 0 and 2")
                    elif setting == 'default_max_tokens':
                        if not isinstance(value, int) or value < 1:
                            raise ValidationError(f"{setting} must be positive integer")
                            
    def _validate_reporting(self, reporting: Dict[str, Any]):
        """Validate reporting configuration"""
        if not isinstance(reporting, dict):
            raise ValidationError("reporting must be a dictionary")
            
        # Validate default format
        if 'default_format' in reporting:
            format_value = reporting['default_format']
            valid_formats = ['html', 'json', 'markdown', 'terminal']
            if format_value not in valid_formats:
                raise ValidationError(f"Invalid default_format: {format_value}")
                
        # Validate output directory
        if 'output_directory' in reporting:
            output_dir = reporting['output_directory']
            if not isinstance(output_dir, str):
                raise ValidationError("output_directory must be a string")
                
        # Validate templates
        if 'templates' in reporting:
            templates = reporting['templates']
            if not isinstance(templates, dict):
                raise ValidationError("reporting.templates must be a dictionary")
                
            for template_type, template_path in templates.items():
                if not isinstance(template_path, str):
                    raise ValidationError(f"Template path for {template_type} must be string")
                    
    def _validate_defaults(self, defaults: Dict[str, Any]):
        """Validate defaults configuration"""
        if not isinstance(defaults, dict):
            raise ValidationError("defaults must be a dictionary")
            
        # Validate boolean settings
        boolean_settings = ['verbose', 'quick_mode']
        for setting in boolean_settings:
            if setting in defaults:
                if not isinstance(defaults[setting], bool):
                    raise ValidationError(f"defaults.{setting} must be boolean")
                    
        # Validate threshold
        if 'threshold' in defaults:
            threshold = defaults['threshold']
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                raise ValidationError("defaults.threshold must be between 0 and 1")
                
        # Validate output format
        if 'output_format' in defaults:
            format_value = defaults['output_format']
            valid_formats = ['terminal', 'json', 'html', 'markdown']
            if format_value not in valid_formats:
                raise ValidationError(f"Invalid defaults.output_format: {format_value}")
                
    def validate_test_data_file(self, file_path: str) -> Dict[str, Any]:
        """Validate test data file format"""
        if not Path(file_path).exists():
            raise ValidationError(f"Test data file not found: {file_path}")
            
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f) if file_path.endswith('.yaml') else json.load(f)
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValidationError(f"Invalid format in test data file: {e}")
        except Exception as e:
            raise ValidationError(f"Error reading test data file: {e}")
            
        # Validate test data structure
        if not isinstance(data, dict):
            raise ValidationError("Test data must be a dictionary")
            
        required_fields = ['test_suite', 'version', 'test_cases']
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Required field missing in test data: {field}")
                
        # Validate test cases
        test_cases = data['test_cases']
        if not isinstance(test_cases, list):
            raise ValidationError("test_cases must be a list")
            
        for i, test_case in enumerate(test_cases):
            self._validate_test_case(test_case, i)
            
        return data
        
    def _validate_test_case(self, test_case: Dict[str, Any], index: int):
        """Validate individual test case"""
        if not isinstance(test_case, dict):
            raise ValidationError(f"Test case {index} must be a dictionary")
            
        required_fields = ['id', 'category', 'input', 'expected_behavior', 'harmful_patterns', 'severity']
        for field in required_fields:
            if field not in test_case:
                raise ValidationError(f"Test case {index} missing required field: {field}")
                
        # Validate field types
        if not isinstance(test_case['id'], str):
            raise ValidationError(f"Test case {index} id must be string")
            
        if not isinstance(test_case['harmful_patterns'], list):
            raise ValidationError(f"Test case {index} harmful_patterns must be list")
            
        valid_severities = ['low', 'medium', 'high', 'critical']
        if test_case['severity'] not in valid_severities:
            raise ValidationError(f"Test case {index} invalid severity: {test_case['severity']}")
            
    def get_validation_errors(self, config: Dict[str, Any]) -> List[str]:
        """Get list of validation errors without raising exceptions"""
        errors = []
        
        try:
            self.validate_config_dict(config)
        except ValidationError as e:
            errors.append(str(e))
            
        return errors
        
    def is_valid_config(self, config: Dict[str, Any]) -> bool:
        """Check if configuration is valid"""
        try:
            self.validate_config_dict(config)
            return True
        except ValidationError:
            return False