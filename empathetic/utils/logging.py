import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class EmphatheticLogger:
    """Comprehensive logging system for Empathetic framework"""

    def __init__(self, name: str = "empathetic", log_level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Prevent duplicate handlers - check for any existing handlers
        if not self.logger.handlers and not any(
            handler for handler in logging.getLogger().handlers
            if hasattr(handler, 'stream') and hasattr(handler.stream, 'name')
        ):
            self._setup_handlers()

        # Prevent propagation to avoid root logger duplication
        self.logger.propagate = False

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
        log_dir = Path("outputs/logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Organized log file naming
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = log_dir / f"empathetic_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def debug(self, message: str, extra: Optional[dict[str, Any]] = None):
        """Log debug message"""
        self._log_with_extra(logging.DEBUG, message, extra)

    def info(self, message: str, extra: Optional[dict[str, Any]] = None):
        """Log info message"""
        self._log_with_extra(logging.INFO, message, extra)

    def warning(self, message: str, extra: Optional[dict[str, Any]] = None):
        """Log warning message"""
        self._log_with_extra(logging.WARNING, message, extra)

    def error(self, message: str, extra: Optional[dict[str, Any]] = None, exc_info: bool = False):
        """Log error message"""
        self._log_with_extra(logging.ERROR, message, extra, exc_info)

    def critical(self, message: str, extra: Optional[dict[str, Any]] = None, exc_info: bool = False):
        """Log critical message"""
        self._log_with_extra(logging.CRITICAL, message, extra, exc_info)

    def _log_with_extra(self, level: int, message: str, extra: Optional[dict[str, Any]], exc_info: bool = False):
        """Log message with extra context"""
        if extra:
            # Add extra context to message
            extra_str = json.dumps(extra, default=str)
            message = f"{message} | Context: {extra_str}"

        self.logger.log(level, message, exc_info=exc_info)

    def log_test_start(self, model: str, suites: list, config: Optional[dict] = None):
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

    def log_test_complete(self, model: str, results: dict[str, Any]):
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

    def log_suite_result(self, suite_name: str, result: dict[str, Any]):
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

    def log_error_detail(self, error_type: str, error_message: str, context: Optional[dict] = None):
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
_logger_instances = {}

def get_logger(name: str = "empathetic") -> EmphatheticLogger:
    """Get logger instance for specific module"""
    if name not in _logger_instances:
        _logger_instances[name] = EmphatheticLogger(name)
    return _logger_instances[name]

# Initialize commonly used loggers
logger = get_logger()
test_logger = get_logger("empathetic.tests")
provider_logger = get_logger("empathetic.providers")
metrics_logger = get_logger("empathetic.metrics")

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
