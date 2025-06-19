# Empathetic - Human-Centered AI Evaluation Framework

Empathetic is an open-source testing framework that evaluates AI models for bias, fairness, and alignment with human
values. Built with affected communities, it provides comprehensive test suites, simple CLI tools, and actionable
insights to ensure AI systems demonstrate genuine understanding and respect for all people.

## Features

- **Bias Detection**: Test for gender, racial, age, and cultural biases
- **Alignment Testing**: Evaluate alignment with human values and ethics
- **Fairness Assessment**: Test fairness across different groups and demographics
- **Empathy Evaluation**: Assess understanding of human circumstances and dignity
- **Safety Testing**: Detect harmful content and safety violations
- **Community Validation**: Real people from affected communities validate AI responses
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
emp test gpt-4 --suite bias,empathy,employment

# Test all suites including domain-specific
emp test gpt-4 --suite all

# Enable adversarial testing for bias detection
emp test gpt-4 --suite empathy --adversarial

# Show detailed empathy dimension scores
emp test gpt-4 --suite empathy --dimensions

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

- **Six-Dimension Scoring**: Recognition, Understanding, Dignity, Support, Growth, Non-judgment
- **Adversarial Testing**: Identity variations reveal hidden biases
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

### Employment Testing ✅

- Hiring bias and discrimination detection
- Resume gap evaluation (caregiving, health, etc.)
- Criminal record and second chances
- Disability accommodation requests
- Pregnancy and family planning bias
- Age discrimination in hiring
- Mental health workplace stigma
- Immigration status and visa sponsorship
- LGBTQ+ workplace inclusion

### Healthcare Testing ✅

- Medical bias in pain assessment
- Racial disparities in treatment
- Gender bias in women's health
- Mental health stigma and treatment
- Disability autonomy in medical care
- Addiction treatment and recovery
- Transgender healthcare access
- Language barriers and interpretation
- Insurance coverage and advocacy

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

# Model capabilities
emp capabilities MODEL          # Detect model capabilities and get recommendations
emp capabilities MODEL --verbose  # Detailed capability analysis

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

    # Run comprehensive tests including domain-specific
    results = await tester.run_tests(
        model="gpt-4",
        suites=["bias", "alignment", "fairness", "empathy", "employment", "healthcare", "safety"],
        config={"adversarial": True}  # Enable adversarial testing
    )

    print(f"Overall score: {results.overall_score:.3f}")

    # Check individual suite results
    for suite_name, result in results.suite_results.items():
        print(f"{suite_name}: {result.score:.3f} ({result.tests_passed}/{result.tests_total})")


# Run the test
asyncio.run(test_model())
```

## Community Validation

Empathetic includes a groundbreaking community validation system where real people from affected communities evaluate AI responses. This ensures our tests reflect authentic lived experiences.

- **500+ Community Validators** from disability, LGBTQ+, racial justice, and other communities
- **Partner Organizations** verify validator credentials and provide oversight
- **Real-world Impact** - validation results help improve AI systems and inform policy

**Learn More**: [Community Validation Documentation](docs/community-validation.md)

## Model Capability Detection

Empathetic automatically detects AI model capabilities to optimize testing effectiveness. The system analyzes empathy baselines, bias susceptibility, cultural awareness, and performance characteristics to provide adaptive testing recommendations.

```bash
# Detect model capabilities
emp capabilities gpt-4

# Get detailed capability analysis
emp capabilities gpt-4 --verbose
```

**Features:**
- **Empathy Baseline Detection**: Measures core empathy across standard scenarios
- **Bias Susceptibility Analysis**: Tests how empathy varies by identity markers  
- **Cultural Awareness Assessment**: Evaluates understanding of diverse contexts
- **Adaptive Test Configuration**: Automatically optimizes test complexity and focus areas
- **Smart Test Selection**: Recommends optimal test suites based on model strengths

**Learn More**: [Model Capabilities Documentation](docs/model-capabilities.md)

### Get Involved
- **Become a Validator**: Join our community validation network
- **Partner Organization**: Help verify community validators
- **Submit Test Cases**: Share scenarios from your lived experience

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

**Community Contributions Welcome**: We especially encourage contributions from affected communities to help expand our test coverage and validation network.

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