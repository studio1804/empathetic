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
- **Multiple Providers**: Support for OpenAI, Anthropic, and HuggingFace models
- **Rich Reporting**: Generate detailed reports in HTML, JSON, or Markdown
- **REST API**: Full REST API for programmatic access
- **Interactive Setup**: Easy configuration with `emp setup` - creates secure .env file
- **API Key Management**: Secure local storage and management via `emp keys` commands
- **Web Interface**: API server with interactive documentation at `/docs`
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

### 1. Setup & Configuration

```bash
# Interactive setup (recommended) - creates .env file with API keys
emp setup

# Check your environment and API key configuration
emp env-check

# View configured API keys (masked for security)
emp keys show
```

### 2. Testing AI Models

```bash
# Test a model with default suites
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

### 3. API Server

```bash
# Launch the API server
emp serve

# Visit http://localhost:8000/docs for interactive API documentation
# Visit http://localhost:8000/api/health for health check
```

## Configuration

### Environment Variables (.env)

The framework uses a `.env` file to store sensitive configuration like API keys securely:

```bash
# Interactive setup (recommended) - creates .env file
emp setup

# Or manually copy and edit the template
cp .env.example .env
nano .env

# Manage API keys via CLI
emp keys show           # Show configured keys (masked for security)
emp keys set openai     # Set/update OpenAI API key
emp keys set anthropic  # Set/update Anthropic API key
emp keys remove openai  # Remove a key
```

**Environment Variables:**
- `OPENAI_API_KEY` - OpenAI API key (required for OpenAI models)
- `ANTHROPIC_API_KEY` - Anthropic API key (required for Claude models)
- `HUGGINGFACE_API_KEY` - HuggingFace API key (optional)
- `EMPATHETIC_DEFAULT_MODEL` - Default model for testing (default: gpt-3.5-turbo)
- `EMPATHETIC_CONFIG` - Path to YAML config file (default: ./empathetic.yaml)
- `SECRET_KEY` - Secret key for API authentication
- `DEBUG` - Enable debug mode (default: false)

**Security:**
- The `.env` file is automatically ignored by git
- API keys are never logged or displayed in full
- Keys are masked when displayed (e.g., `sk-test1...7890`)
- All keys are stored locally only

### Configuration File (empathetic.yaml)

Create `empathetic.yaml` in your project directory for test configuration:

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

### Setup & Configuration
```bash
emp setup                       # Interactive setup wizard - creates .env file
emp env-check                   # Check environment configuration and API keys
emp keys show                   # Display configured API keys (masked)
emp keys set PROVIDER           # Set API key for provider (openai, anthropic, huggingface)
emp keys remove PROVIDER        # Remove API key for provider
```

### API Server
```bash
emp serve                       # Launch API server on http://localhost:8000
emp serve --host 0.0.0.0        # Bind to all interfaces
emp serve --port 8080           # Use custom port
emp serve --reload              # Enable auto-reload for development
```

### Testing Commands  
```bash
emp test MODEL                  # Run all tests on a model
emp test MODEL --suite SUITE1,SUITE2  # Run specific test suites
emp test MODEL --quick          # Run subset of tests for faster feedback
emp test MODEL --threshold 0.95 # Set custom passing threshold
emp test MODEL --output html    # Generate HTML report
emp test MODEL --verbose        # Verbose output with detailed information
```

### Quick Checks
```bash
emp check MODEL --suite SUITE   # Quick check against specific suite
emp check MODEL --suite bias --quick  # Quick bias check
```

### Model Capabilities
```bash
emp capabilities MODEL          # Detect model capabilities and get recommendations
emp capabilities MODEL --verbose  # Detailed capability analysis
```

### Reports
```bash
emp report --format html        # Generate HTML report from results
emp report --input results.json --format markdown  # Convert results
```

## REST API

The Empathetic framework includes a REST API for web integration.

### Starting the API Server

```bash
# Start the API server
emp serve

# Custom configuration
emp serve --host 0.0.0.0 --port 8080 --reload
```

### API Endpoints

#### Testing Endpoints
- `POST /api/testing/run` - Run test suites on a model
- `GET /api/testing/models` - List available AI models
- `GET /api/testing/suites` - List available test suites
- `GET /api/testing/results/{model}` - Get recent test results for a model
- `POST /api/testing/capabilities/{model}` - Detect model capabilities

#### Report Endpoints
- `GET /api/reports/generate/{model}` - Generate report for a model
- `GET /api/reports/download/{model}` - Download report file
- `GET /api/reports/dashboard/{model}` - Get dashboard data

#### System Endpoints
- `GET /api/health/` - Health check
- `GET /api/health/ready` - Readiness check
- `GET /docs` - Interactive API documentation
- `GET /` - API information

### Example API Usage

```bash
# Test a model via API
curl -X POST "http://localhost:8000/api/testing/run" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "suites": ["empathy", "bias"],
    "quick_mode": false,
    "enable_validation": false
  }'

# Get health status
curl http://localhost:8000/api/health/

# List available models
curl http://localhost:8000/api/testing/models

```

## API Usage (Python)

### Direct Framework Usage

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

### REST API Usage

```python
import requests

# Start API server first: emp serve

# Test a model via REST API
response = requests.post("http://localhost:8000/api/testing/run", json={
    "model": "gpt-4",
    "suites": ["empathy", "bias"],
    "quick_mode": False,
    "enable_validation": False
})

if response.status_code == 200:
    result = response.json()
    print(f"Overall score: {result['overall_score']:.3f}")
    print(f"Passed: {result['passed']}")
else:
    print(f"Error: {response.status_code}")

# Get available models
models = requests.get("http://localhost:8000/api/testing/models").json()
print("Available models:", [m['id'] for m in models['models']])

```

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Development

### Setup Development Environment

```bash
# Install development dependencies
poetry install

# Create .env file for development
emp setup

# Run tests
pytest

# Run linting
black .
ruff check .

# Type checking
mypy .
```

### API Development

```bash
# Start API server with auto-reload
emp serve --reload

# Run API in development mode
emp serve --host 0.0.0.0 --port 8000 --reload

# Test API endpoints
curl http://localhost:8000/api/health/
curl http://localhost:8000/docs

# Test the API
curl http://localhost:8000/api/testing/models
```

### Environment Management

```bash
# Check current configuration
emp env-check

# View all configured API keys
emp keys show

# Test with different providers
emp keys set openai
emp test gpt-4

emp keys set anthropic  
emp test claude-3-sonnet
```

### Testing New Features

```bash
# Run specific test suites
pytest tests/test_api.py
pytest tests/test_cli.py

# Test CLI commands
emp test gpt-3.5-turbo --suite empathy --quick
emp capabilities gpt-4 --verbose

# Test API endpoints
python -c "
import requests
response = requests.get('http://localhost:8000/api/testing/models')
print(response.json())
"
```

## Troubleshooting

### Common Issues

#### API Key Configuration
```bash
# Check if API keys are configured
emp env-check

# API key not found
emp keys show
emp setup  # Re-run setup to configure keys

# Key format validation failed
emp keys set openai  # Re-enter key with correct format
```

#### Server Issues
```bash
# Port already in use
emp serve --port 8001

# Server not responding
curl http://localhost:8000/api/health/  # Check if server is running
emp serve --reload  # Restart with auto-reload

# Import errors
pip install -e .  # Reinstall in development mode
poetry install     # Install all dependencies
```

#### Test Failures
```bash
# No API key configured
emp setup  # Configure API keys first

# Test data files missing (warnings are normal for development)
# These warnings can be ignored during initial setup

# Rate limiting
# Wait a few minutes between test runs
# Use --quick flag for faster testing
emp test gpt-3.5-turbo --suite empathy --quick
```

#### Environment Issues
```bash
# .env file not found
cp .env.example .env
emp setup

# Permission errors
chmod 600 .env  # Secure .env file permissions

# Configuration not loading
rm .env && emp setup  # Recreate .env file
```

### Getting Help

- Check the logs: `emp serve` shows detailed error messages
- Validate API keys: `emp keys show` to verify configuration
- Test connectivity: `curl http://localhost:8000/api/health/`
- Review configuration: `emp env-check`
- Interactive API docs: Visit `http://localhost:8000/docs`

For additional support, please check the [GitHub Issues](https://github.com/studio1804/empathetic/issues) or create a new issue.