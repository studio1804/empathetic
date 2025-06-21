# Empathetic API

The Empathetic API provides a REST interface for the Human-Centered AI Evaluation Framework, enabling web-based testing and community validation.

## Quick Start

### 1. Start the API Server

```bash
# Install dependencies
poetry install

# Start the server
emp serve

# Or with options
emp serve --host 0.0.0.0 --port 8080 --reload
```

### 2. Access the API

- API Documentation: http://localhost:8000/docs
- Community Validator App: http://localhost:8000/
- Health Check: http://localhost:8000/api/health

## API Endpoints

### Testing
- `POST /api/testing/run` - Run test suites on a model
- `GET /api/testing/models` - List available models
- `GET /api/testing/suites` - List test suites
- `GET /api/testing/results/{model}` - Get test results
- `POST /api/testing/capabilities/{model}` - Detect model capabilities

### Community Validation
- `POST /api/validation/request` - Create validation request
- `GET /api/validation/pending` - Get pending validations
- `POST /api/validation/submit/{id}` - Submit validation response
- `GET /api/validation/consensus/{id}` - Get community consensus
- `POST /api/validation/register` - Register as validator

### Reports
- `GET /api/reports/generate/{model}` - Generate report
- `GET /api/reports/download/{model}` - Download report
- `GET /api/reports/dashboard/{model}` - Get dashboard data

## Architecture

```
empathetic/api/
├── main.py           # FastAPI app
├── config.py         # Configuration
├── models/           # Pydantic models
├── routers/          # API endpoints
├── services/         # Business logic
└── README.md         # This file
```

## Development

### Running Tests
```bash
pytest empathetic/api/tests/
```

### Adding New Endpoints
1. Create router in `routers/`
2. Add service logic in `services/`
3. Define models in `models/`
4. Include router in `main.py`

### Environment Variables
```bash
# Required
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...

# Optional
DATABASE_URL=postgresql://...
SECRET_KEY=your-secret-key
DEBUG=true
```

## Community Validator App

The validator app is built with Lit web components for simplicity and accessibility:

```bash
cd empathetic/validator
npm install
npm run dev
```

### Features
- Mobile-first responsive design
- Accessibility-first (WCAG AA)
- Offline capability
- Simple, fast interface

## Security

- JWT authentication for validators
- API key validation
- CORS configuration
- Rate limiting (coming soon)

## Deployment

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["emp", "serve", "--host", "0.0.0.0"]
```

### Production
```bash
# Use production server
gunicorn empathetic.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```