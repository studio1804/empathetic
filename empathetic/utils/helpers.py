import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def get_api_key(provider: str, env_var: Optional[str] = None) -> Optional[str]:
    """Get API key for provider from environment"""
    if env_var:
        return os.getenv(env_var)

    # Standard environment variable names
    env_vars = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'huggingface': 'HUGGINGFACE_API_KEY'
    }

    return os.getenv(env_vars.get(provider.lower()))

def ensure_dir(path: str) -> str:
    """Ensure directory exists, create if needed"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def safe_filename(filename: str) -> str:
    """Create safe filename by removing/replacing invalid characters"""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    # Limit length
    if len(filename) > 200:
        filename = filename[:200]

    return filename

def hash_content(content: str) -> str:
    """Generate hash of content for caching/comparison"""
    return hashlib.sha256(content.encode()).hexdigest()

def format_timestamp(timestamp: Optional[str] = None) -> str:
    """Format timestamp for display"""
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return timestamp
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate_percentage(part: int, total: int) -> float:
    """Calculate percentage with safe division"""
    if total == 0:
        return 0.0
    return (part / total) * 100

def format_score(score: float, precision: int = 3) -> str:
    """Format score for display"""
    return f"{score:.{precision}f}"

def load_json_file(file_path: str) -> dict[str, Any]:
    """Safely load JSON file"""
    try:
        with open(file_path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")

def save_json_file(data: dict[str, Any], file_path: str) -> None:
    """Safely save data to JSON file"""
    # Ensure directory exists
    ensure_dir(os.path.dirname(file_path))

    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        raise OSError(f"Failed to save JSON file {file_path}: {e}")

def merge_configs(base_config: dict[str, Any], override_config: dict[str, Any]) -> dict[str, Any]:
    """Merge two configuration dictionaries"""
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text if it's too long"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def extract_patterns(text: str, patterns: list[str]) -> list[str]:
    """Extract matching patterns from text"""
    found = []
    text_lower = text.lower()

    for pattern in patterns:
        if pattern.lower() in text_lower:
            found.append(pattern)

    return found

def calculate_confidence(score: float, sample_size: int) -> str:
    """Calculate confidence level based on score and sample size"""
    if sample_size < 10:
        return "Low"
    elif sample_size < 50:
        return "Medium"
    elif score > 0.95 or score < 0.05:
        return "High"
    else:
        return "Very High"

def get_grade_color(score: float) -> str:
    """Get color code for score display"""
    if score >= 0.9:
        return "green"
    elif score >= 0.7:
        return "yellow"
    else:
        return "red"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

def group_by_category(items: list[dict[str, Any]], category_key: str = 'category') -> dict[str, list[dict[str, Any]]]:
    """Group items by category"""
    groups = {}

    for item in items:
        category = item.get(category_key, 'unknown')
        if category not in groups:
            groups[category] = []
        groups[category].append(item)

    return groups

def detect_provider_from_model(model_name: str) -> str:
    """Detect provider from model name"""
    model_lower = model_name.lower()

    if model_lower.startswith('gpt-'):
        return 'openai'
    elif model_lower.startswith('claude-'):
        return 'anthropic'
    elif model_lower.startswith('gemini-'):
        return 'google'
    elif '/' in model_name:  # HuggingFace format
        return 'huggingface'
    else:
        return 'unknown'

def validate_environment() -> dict[str, bool]:
    """Validate that environment is properly set up"""
    checks = {
        'python_version': True,  # Assume OK since we're running
        'openai_key': bool(os.getenv('OPENAI_API_KEY')),
        'anthropic_key': bool(os.getenv('ANTHROPIC_API_KEY')),
        'config_file': False,
        'data_files': False
    }

    # Check for config file
    config_paths = [
        './config/default.yaml',
        './empathetic.yaml'
    ]

    for path in config_paths:
        if Path(path).exists():
            checks['config_file'] = True
            break

    # Check for data files
    data_path = Path('./data/tests/bias_tests.json')
    if data_path.exists():
        checks['data_files'] = True

    return checks
