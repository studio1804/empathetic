"""Testing models for AI evaluation."""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class TestRequest(BaseModel):
    """Request to run tests on an AI model."""
    model: str = Field(description="Model identifier (e.g., gpt-4)")
    suites: list[str] = Field(
        default=["empathy", "bias"],
        description="Test suites to run"
    )
    config: Optional[dict[str, Any]] = Field(default={})
    quick_mode: bool = Field(default=False, description="Run subset of tests")
    enable_validation: bool = Field(
        default=False,
        description="Request community validation for responses"
    )
    enhanced_evaluation: bool = Field(
        default=False,
        description="Use context-aware evaluation with NLP analysis"
    )


class TestCase(BaseModel):
    """Individual test case result."""
    id: str
    name: str
    category: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    input: str
    expected_markers: list[str]
    actual_response: str
    detected_markers: list[str]
    issues: list[str] = []
    execution_time: float


class TestSuite(BaseModel):
    """Test suite results."""
    name: str
    score: float = Field(ge=0.0, le=1.0)
    tests_total: int
    tests_passed: int
    tests_failed: int
    execution_time: float
    test_cases: list[TestCase]

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        return self.tests_passed / self.tests_total if self.tests_total > 0 else 0.0


class TestResult(BaseModel):
    """Complete test result for a model."""
    model: str
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    threshold: float = Field(default=0.9)
    suite_results: dict[str, TestSuite]
    empathy_dimensions: Optional[dict[str, float]] = None
    bias_analysis: Optional[dict[str, Any]] = None
    community_validation_pending: bool = False
    started_at: datetime
    completed_at: datetime
    total_execution_time: float

    @property
    def summary(self) -> dict[str, Any]:
        """Get result summary."""
        return {
            "model": self.model,
            "passed": self.passed,
            "overall_score": self.overall_score,
            "suites_run": len(self.suite_results),
            "total_tests": sum(s.tests_total for s in self.suite_results.values()),
            "total_passed": sum(s.tests_passed for s in self.suite_results.values()),
        }


class ComparisonRequest(BaseModel):
    """Request for comparative evaluation."""
    model: str = Field(description="AI model to test")
    suites: list[str] = Field(default=["bias", "safety"], description="Test suites to run")
    test_cases: Optional[list[str]] = Field(default=None, description="Specific test case IDs")
    include_metrics: bool = Field(default=True, description="Include detailed metrics")


class BaselineEvaluation(BaseModel):
    """Baseline evaluation result."""
    failed: bool
    patterns_found: list[str]
    confidence: float
    explanation: str


class EnhancedEvaluation(BaseModel):
    """Enhanced evaluation result."""
    failed: bool
    confidence: float
    explanation: str
    intent: Optional[str] = None
    linguistic_features: Optional[dict[str, Any]] = None
    processing_time: float


class ComparisonResult(BaseModel):
    """Result of comparative evaluation."""
    model: str
    test_case_id: str
    baseline: BaselineEvaluation
    enhanced: EnhancedEvaluation
    is_false_positive: bool
    is_false_negative: bool
    improvement_explanation: str
    confidence_improvement: float
