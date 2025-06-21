"""Testing service for running AI evaluations."""
from datetime import datetime
from typing import Optional

from empathetic.core.tester import Tester

from ..models.testing import TestCase, TestResult, TestSuite


class TestingService:
    """Service for managing AI model testing."""

    def __init__(self):
        self.tester = Tester()
        self._results_cache = {}

    async def run_tests(
        self,
        model: str,
        suites: list[str],
        config: Optional[dict] = None,
        quick_mode: bool = False
    ) -> TestResult:
        """Run test suites on a model."""
        started_at = datetime.utcnow()

        # Run tests using the existing Empathetic framework
        results = await self.tester.run_tests(
            model=model,
            suites=suites,
            config=config or {}
        )

        completed_at = datetime.utcnow()

        # Convert to API model
        suite_results = {}
        for suite_name, suite_result in results.suite_results.items():
            test_cases = []
            for test in suite_result.tests:
                test_cases.append(TestCase(
                    id=test.id,
                    name=test.name,
                    category=test.category,
                    passed=test.passed,
                    score=test.score,
                    input=test.input,
                    expected_markers=test.expected_markers,
                    actual_response=test.response,
                    detected_markers=test.detected_markers,
                    issues=test.issues,
                    execution_time=0.1  # Placeholder
                ))

            suite_results[suite_name] = TestSuite(
                name=suite_name,
                score=suite_result.score,
                tests_total=suite_result.tests_total,
                tests_passed=suite_result.tests_passed,
                tests_failed=suite_result.tests_failed,
                execution_time=1.0,  # Placeholder
                test_cases=test_cases
            )

        test_result = TestResult(
            model=model,
            overall_score=results.overall_score,
            passed=results.passed,
            threshold=results.threshold,
            suite_results=suite_results,
            started_at=started_at,
            completed_at=completed_at,
            total_execution_time=(completed_at - started_at).total_seconds()
        )

        # Cache result
        if model not in self._results_cache:
            self._results_cache[model] = []
        self._results_cache[model].insert(0, test_result)

        return test_result

    async def get_recent_results(self, model: str, limit: int = 10) -> list[TestResult]:
        """Get recent test results for a model."""
        return self._results_cache.get(model, [])[:limit]

    async def detect_capabilities(self, model: str) -> dict:
        """Detect model capabilities."""
        provider = self.tester._get_provider(model)
        if hasattr(provider, 'detect_capabilities'):
            return await provider.detect_capabilities()
        else:
            return {
                "error": "Capability detection not supported for this model",
                "model": model
            }
