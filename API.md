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
  "enable_validation": true
}

# Get models
GET /api/testing/models

# Get test suites
GET /api/testing/suites
```


### Reports
```bash
# Generate report
GET /api/reports/generate/{model}?format=html

# Get dashboard data
GET /api/reports/dashboard/{model}
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