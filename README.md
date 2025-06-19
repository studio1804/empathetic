# Empathetic - AI Testing Framework

Test AI models for bias, alignment, and human values.

## Features

- **Bias Detection**: Test for gender, racial, age, and cultural biases
- **Alignment Testing**: Evaluate alignment with human values (coming soon)
- **Fairness Metrics**: Assess fairness across different groups (coming soon)
- **Safety Testing**: Check for harmful content generation (coming soon)
- **Multiple Providers**: Support for OpenAI, Anthropic, and HuggingFace models
- **Rich Reporting**: Generate detailed reports in HTML, JSON, or Markdown

## Installation

```bash
# Clone the repository
git clone https://github.com/studio1804/empathetic.git
cd empathetic

# Install with Poetry
pip install poetry
poetry install

# Or install with pip
pip install -e .
```

## Quick Start

```bash
# Interactive setup (recommended)
emp setup

# Or manually set your API key
export OPENAI_API_KEY="your-api-key-here"

# Check your environment
emp env-check

# Test a model
emp test gpt-4

# Test specific suites
emp test gpt-4 --suite bias

# Generate HTML report
emp test gpt-4 --output html

# Quick check
emp check gpt-4 --suite bias --quick
```

## Configuration

Create `empathetic.yaml` in your project directory:

```yaml
test_suites:
  bias:
    enabled: true
    test_files:
      - data/tests/bias_tests.json

scoring:
  thresholds:
    pass: 0.9
    warning: 0.7
```

## Test Suites

### Bias Testing (Available)
- Gender bias in occupations
- Racial assumptions
- Age stereotypes
- Cultural biases

### Alignment Testing
- Human values alignment
- Ethical decision making
- (Coming soon)

### Fairness Testing
- Group fairness metrics
- Individual fairness
- (Coming soon)

### Safety Testing
- Harmful content detection
- Toxicity assessment
- (Coming soon)

## API Usage

```python
from empathetic.core.tester import Tester
from empathetic.providers.openai import OpenAIProvider

# Create tester
tester = Tester()

# Run tests
results = await tester.run_tests("gpt-4", suites=["bias"])

print(f"Overall score: {results.overall_score}")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Development

```bash
# Install development dependencies
poetry install

# Run tests
pytest

# Run linting
black .
ruff check .

# Type checking
mypy .
```