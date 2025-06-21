# Empathetic API

REST API for the Human-Centered AI Evaluation Framework.

## Quick Start

```bash
# Configure API keys
emp setup

# Start the server
emp serve

# Visit API documentation
open http://localhost:8000/docs
```

## Authentication

System endpoints (testing, health, reports) are publicly accessible.


## Core Endpoints

### Testing
```bash
# Run tests
POST /api/testing/run
{
  "model": "gpt-4",
  "suites": ["empathy", "bias"],
  "quick_mode": false,
  "enable_validation": true,
  "enhanced_evaluation": true  # Enable context-aware evaluation
}

# Run comparative evaluation (baseline vs enhanced)
POST /api/testing/compare
{
  "model": "gpt-4",
  "suites": ["bias", "safety"],
  "test_cases": ["specific_test_ids"]  # Optional: test specific cases
}

# Get models
GET /api/testing/models

# Get test suites
GET /api/testing/suites

# Get evaluation metrics
GET /api/testing/metrics/{model}
```


### Reports
```bash
# Generate report
GET /api/reports/generate/{model}?format=html&include_comparison=true

# Get dashboard data
GET /api/reports/dashboard/{model}

# Get false positive analysis
GET /api/reports/false-positives/{model}

# Get improvement metrics
GET /api/reports/improvements/{model}
```

### Health
```bash
# Health check
GET /api/health/

# Readiness check
GET /api/health/ready
```

## Development

```bash
# Start with auto-reload
emp serve --reload

# Custom port
emp serve --port 8080

# All interfaces
emp serve --host 0.0.0.0
```

## Error Handling

The API returns standard HTTP status codes:
- `200` - Success
- `400` - Bad Request (invalid input)
- `401` - Unauthorized (invalid token)
- `404` - Not Found
- `422` - Validation Error (Pydantic)
- `500` - Internal Server Error

Error responses include details:
```json
{
  "detail": "Error description",
  "status_code": 400
}
```

## Rate Limiting

Currently no rate limiting is implemented. For production use, implement rate limiting middleware.

## WebSocket Support

Planned for real-time test progress updates. Not yet implemented.

## Enhanced Evaluation API

The enhanced context-aware evaluation provides additional endpoints for advanced analysis:

### Context Analysis
```bash
# Analyze specific text with context
POST /api/evaluation/analyze-context
{
  "text": "The nurse walked into the room. She was tired.",
  "patterns": ["she", "nurse"],
  "category": "bias"
}

# Response includes:
{
  "baseline_result": {
    "failed": true,
    "patterns_found": ["she"]
  },
  "enhanced_result": {
    "failed": false,
    "explanation": "Pattern 'she' is properly used as pronoun reference",
    "intent": "neutral_description",
    "confidence": 0.85
  },
  "is_false_positive": true
}
```

### Batch Evaluation
```bash
# Process multiple responses efficiently
POST /api/evaluation/batch
{
  "responses": ["response1", "response2", ...],
  "test_cases": ["test_id1", "test_id2", ...],
  "use_cache": true
}
```

### Performance Metrics
```bash
# Get performance statistics
GET /api/evaluation/performance

# Response:
{
  "avg_processing_time_ms": 450,
  "cache_hit_rate": 0.75,
  "false_positive_reduction": 0.92,
  "loaded_models": ["spacy", "transformers"]
}