"""Testing endpoints for running AI model evaluations."""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import List, Optional

from ..models.testing import TestRequest, TestResult, ComparisonRequest, ComparisonResult
from ..services.testing import TestingService
from ..services.validation import ValidationService

router = APIRouter()
testing_service = TestingService()
validation_service = ValidationService()


@router.post("/run", response_model=TestResult)
async def run_tests(
    request: TestRequest,
    background_tasks: BackgroundTasks
):
    """Run test suites on an AI model."""
    try:
        result = await testing_service.run_tests(
            model=request.model,
            suites=request.suites,
            config=request.config,
            quick_mode=request.quick_mode
        )

        if request.enable_validation and result.passed:
            background_tasks.add_task(
                validation_service.request_validation_for_tests,
                result
            )
            result.community_validation_pending = True

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """List available AI models for testing."""
    return {
        "models": [
            {"id": "gpt-4", "name": "GPT-4", "provider": "openai"},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "provider": "openai"},
            {"id": "claude-3-opus", "name": "Claude 3 Opus", "provider": "anthropic"},
            {"id": "claude-3-sonnet", "name": "Claude 3 Sonnet", "provider": "anthropic"},
        ]
    }


@router.get("/suites")
async def list_test_suites():
    """List available test suites."""
    return {
        "suites": [
            {
                "id": "empathy",
                "name": "Empathy Testing",
                "description": "Tests for understanding human circumstances and dignity",
                "test_count": 50
            },
            {
                "id": "bias",
                "name": "Bias Detection",
                "description": "Tests for gender, racial, age, and cultural biases",
                "test_count": 40
            },
            {
                "id": "fairness",
                "name": "Fairness Assessment",
                "description": "Tests fairness across different groups",
                "test_count": 35
            },
            {
                "id": "safety",
                "name": "Safety Testing",
                "description": "Tests for harmful content and safety violations",
                "test_count": 30
            },
            {
                "id": "employment",
                "name": "Employment Domain",
                "description": "Tests for hiring bias and workplace discrimination",
                "test_count": 25
            },
            {
                "id": "healthcare",
                "name": "Healthcare Domain",
                "description": "Tests for medical bias and healthcare equity",
                "test_count": 25
            }
        ]
    }


@router.get("/results/{model}")
async def get_test_results(model: str, limit: int = 10):
    """Get recent test results for a model."""
    results = await testing_service.get_recent_results(model, limit)
    return {"model": model, "results": results}


@router.post("/capabilities/{model}")
async def detect_capabilities(model: str):
    """Detect model capabilities for adaptive testing."""
    try:
        capabilities = await testing_service.detect_capabilities(model)
        return capabilities
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=ComparisonResult)
async def run_comparative_evaluation(request: ComparisonRequest):
    """Run comparative evaluation (baseline vs enhanced) on a model."""
    try:
        result = await testing_service.run_comparative_evaluation(
            model=request.model,
            suites=request.suites,
            test_cases=request.test_cases,
            include_metrics=request.include_metrics
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{model}")
async def get_evaluation_metrics(
    model: str,
    days: int = 7,
    category: Optional[str] = None
):
    """Get evaluation metrics for a model."""
    try:
        metrics = await testing_service.get_evaluation_metrics(
            model=model,
            days=days,
            category=category
        )
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-context")
async def analyze_context(
    text: str,
    patterns: List[str],
    category: str = "bias"
):
    """Analyze specific text with context-aware evaluation."""
    try:
        result = await testing_service.analyze_context(
            text=text,
            patterns=patterns,
            category=category
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
