# Empathetic - AI Testing Framework

Empathetic is an open-source testing framework that evaluates AI models for bias, fairness, and alignment with human
values. Built with affected communities, it provides comprehensive test suites, simple CLI tools, and actionable
insights to ensure AI systems demonstrate genuine understanding and respect for all people.

## Features

- **Bias Detection**: Test for gender, racial, age, and cultural biases
- **Alignment Testing**: Evaluate alignment with human values and ethics
- **Fairness Assessment**: Test fairness across different groups and demographics
- **Empathy Evaluation**: Assess understanding of human circumstances and dignity
- **Safety Testing**: Detect harmful content and safety violations
- **Multiple Providers**: Support for OpenAI, Anthropic, and HuggingFace models
- **Rich Reporting**: Generate detailed reports in HTML, JSON, or Markdown
- **Interactive Setup**: Easy configuration with `emp setup`
- **Comprehensive Logging**: Detailed logging for monitoring and debugging
- **Extensive Testing**: 100+ unit tests for reliability

## Installation

### Option 1: Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/studio1804/empathetic.git
cd empathetic

# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate the environment
poetry shell
```

### Option 2: Using pip + virtual environment

```bash
# Clone the repository
git clone https://github.com/studio1804/empathetic.git
cd empathetic

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Option 3: Using pyenv (Advanced)

```bash
# Install specific Python version and create environment
pyenv install 3.11.10
pyenv virtualenv 3.11.10 empathetic
pyenv local empathetic

# Install dependencies
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
emp test gpt-4 --suite bias,safety,empathy

# Test all suites
emp test gpt-4 --suite all

# Generate HTML report
emp test gpt-4 --output html

# Quick check with subset of tests
emp check gpt-4 --suite empathy --quick

# Set custom threshold
emp test claude-3-opus --threshold 0.95 --verbose
```

## Configuration

Create `empathetic.yaml` in your project directory:

```yaml
test_suites:
  bias:
    enabled: true
    test_files:
      - data/tests/bias_tests.json
  empathy:
    enabled: true
    test_files:
      - data/tests/empathy_tests.json

scoring:
  weights:
    bias: 0.25
    alignment: 0.25
    fairness: 0.2
    safety: 0.2
    empathy: 0.1
  thresholds:
    pass: 0.9
    warning: 0.7
```

## Test Suites

### Bias Testing ✅

- Gender bias in occupations and roles
- Racial and ethnic assumptions
- Age-related stereotypes
- Cultural and religious biases
- Comprehensive pattern detection

### Alignment Testing ✅

- Honesty and transparency
- Helpfulness vs. boundaries
- Respect for human autonomy
- Justice and fairness principles
- Human dignity and equality
- Privacy and consent

### Fairness Testing ✅

- Employment and hiring fairness
- Financial decision fairness
- Healthcare equity
- Educational assessment fairness
- Housing and services equity
- Criminal justice fairness
- Algorithmic fairness principles

### Empathy Testing ✅

- Understanding of human circumstances and challenges
- Recognition of systemic barriers and inequities
- Preservation of human dignity and agency
- Awareness of real-world impact of AI decisions
- Economic vulnerability and hardship
- Health challenges and disabilities
- Family circumstances and caregiving
- Immigration status and safety concerns
- Housing insecurity and poverty
- Educational barriers and learning differences

### Safety Testing ✅

- Violence and harm prevention
- Self-harm intervention
- Illegal activity detection
- Dangerous content blocking
- Medical misinformation prevention
- Fraud and scam protection
- Child safety and protection
- Hate speech detection

## CLI Commands

```bash
# Setup and configuration
emp setup                    # Interactive setup wizard
emp env-check               # Check environment configuration

# Testing commands  
emp test MODEL              # Run all tests on a model
emp test MODEL --suite SUITE1,SUITE2  # Run specific test suites
emp test MODEL --quick      # Run subset of tests for faster feedback
emp test MODEL --threshold 0.95       # Set custom passing threshold
emp test MODEL --output html          # Generate HTML report
emp test MODEL --verbose    # Verbose output with detailed information

# Quick checks
emp check MODEL --suite SUITE  # Quick check against specific suite
emp check MODEL --suite bias --quick  # Quick bias check

# Reports
emp report --format html    # Generate HTML report from results
emp report --input results.json --format markdown  # Convert results

# Validation
emp validate PATH           # Validate models at path (coming soon)
```

## API Usage

```python
import asyncio
from empathetic.core.tester import Tester


async def test_model():
    # Create tester
    tester = Tester()

    # Run comprehensive tests
    results = await tester.run_tests(
        model="gpt-4",
        suites=["bias", "alignment", "fairness", "empathy", "safety"]
    )

    print(f"Overall score: {results.overall_score:.3f}")

    # Check individual suite results
    for suite_name, result in results.suite_results.items():
        print(f"{suite_name}: {result.score:.3f} ({result.tests_passed}/{result.tests_total})")


# Run the test
asyncio.run(test_model())
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