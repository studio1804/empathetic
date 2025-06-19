import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

class EmphatheticLogger:
    """Comprehensive logging system for Empathetic framework"""
    
    def __init__(self, name: str = "empathetic", log_level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
            
    def _setup_handlers(self):
        """Setup console and file handlers"""
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler for detailed logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"empathetic_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        self._log_with_extra(logging.DEBUG, message, extra)
        
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message"""
        self._log_with_extra(logging.INFO, message, extra)
        
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self._log_with_extra(logging.WARNING, message, extra)
        
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log error message"""
        self._log_with_extra(logging.ERROR, message, extra, exc_info)
        
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log critical message"""
        self._log_with_extra(logging.CRITICAL, message, extra, exc_info)
        
    def _log_with_extra(self, level: int, message: str, extra: Optional[Dict[str, Any]], exc_info: bool = False):
        """Log message with extra context"""
        if extra:
            # Add extra context to message
            extra_str = json.dumps(extra, default=str)
            message = f"{message} | Context: {extra_str}"
            
        self.logger.log(level, message, exc_info=exc_info)
        
    def log_test_start(self, model: str, suites: list, config: Optional[Dict] = None):
        """Log test execution start"""
        self.info(
            f"Starting tests for model: {model}",
            extra={
                "event": "test_start",
                "model": model,
                "suites": suites,
                "config": config or {}
            }
        )
        
    def log_test_complete(self, model: str, results: Dict[str, Any]):
        """Log test execution completion"""
        self.info(
            f"Tests completed for model: {model} | Score: {results.get('overall_score', 0):.3f}",
            extra={
                "event": "test_complete",
                "model": model,
                "overall_score": results.get('overall_score', 0),
                "suite_count": len(results.get('suite_results', {}))
            }
        )
        
    def log_suite_result(self, suite_name: str, result: Dict[str, Any]):
        """Log individual suite results"""
        self.info(
            f"Suite '{suite_name}' completed | Score: {result.get('score', 0):.3f} | "
            f"Tests: {result.get('tests_passed', 0)}/{result.get('tests_total', 0)}",
            extra={
                "event": "suite_complete",
                "suite": suite_name,
                "score": result.get('score', 0),
                "tests_passed": result.get('tests_passed', 0),
                "tests_total": result.get('tests_total', 0)
            }
        )
        
    def log_provider_call(self, provider: str, model: str, prompt_length: int, response_length: int):
        """Log provider API calls"""
        self.debug(
            f"Provider call: {provider} | Model: {model} | "
            f"Prompt: {prompt_length} chars | Response: {response_length} chars",
            extra={
                "event": "provider_call",
                "provider": provider,
                "model": model,
                "prompt_length": prompt_length,
                "response_length": response_length
            }
        )
        
    def log_error_detail(self, error_type: str, error_message: str, context: Optional[Dict] = None):
        """Log detailed error information"""
        self.error(
            f"Error: {error_type} | {error_message}",
            extra={
                "event": "error",
                "error_type": error_type,
                "error_message": error_message,
                "context": context or {}
            },
            exc_info=True
        )
        
    def log_performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log performance metrics"""
        self.debug(
            f"Performance: {metric_name} = {value} {unit}",
            extra={
                "event": "performance",
                "metric": metric_name,
                "value": value,
                "unit": unit
            }
        )

class ColoredFormatter(logging.Formatter):
    """Colored console formatter"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, '')}{log_message}{self.COLORS['RESET']}"

# Global logger instances
logger = EmphatheticLogger()
test_logger = EmphatheticLogger("empathetic.tests")
provider_logger = EmphatheticLogger("empathetic.providers")
metrics_logger = EmphatheticLogger("empathetic.metrics")

def get_logger(name: str = "empathetic") -> EmphatheticLogger:
    """Get logger instance for specific module"""
    return EmphatheticLogger(name)

def set_log_level(level: str):
    """Set global log level"""
    level_value = getattr(logging, level.upper())
    logging.getLogger("empathetic").setLevel(level_value)
    
def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
):
    """Configure global logging settings"""
    root_logger = logging.getLogger("empathetic")
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Setup new handlers based on configuration
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)