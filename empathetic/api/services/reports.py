"""Report generation service."""
from typing import Dict, Optional
import tempfile
import os
from datetime import datetime

from empathetic.reports.generator import ReportGenerator
from .testing import TestingService
from ..models.testing import TestResult


class ReportService:
    """Service for generating reports."""
    
    def __init__(self):
        self.testing_service = TestingService()
        self.report_generator = ReportGenerator()
    
    async def generate_report(
        self,
        model: str,
        format: str = "html",
        include_validation: bool = False
    ) -> str:
        """Generate a report for model results."""
        # Get latest test results
        results = await self.testing_service.get_recent_results(model, limit=1)
        if not results:
            raise ValueError(f"No test results found for model: {model}")
        
        latest_result = results[0]
        
        # Convert to format expected by ReportGenerator
        test_results = self._convert_to_report_format(latest_result)
        
        if format == "html":
            return self.report_generator.generate_html(test_results)
        elif format == "json":
            return test_results
        else:  # markdown
            content = self.report_generator.generate_markdown(test_results)
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False)
            temp_file.write(content)
            temp_file.close()
            return temp_file.name
    
    async def generate_report_file(self, model: str, format: str = "html") -> str:
        """Generate a report file."""
        content = await self.generate_report(model, format)
        
        if format in ["json", "markdown"]:
            return content  # Already a file path or dict
        
        # For HTML, write to temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
        temp_file.write(content)
        temp_file.close()
        return temp_file.name
    
    async def get_dashboard_data(self, model: str) -> Dict:
        """Get dashboard data for a model."""
        results = await self.testing_service.get_recent_results(model, limit=10)
        
        if not results:
            return {
                "model": model,
                "has_results": False
            }
        
        latest = results[0]
        
        # Calculate trends
        score_history = [r.overall_score for r in results]
        score_trend = "improving" if len(score_history) > 1 and score_history[0] > score_history[-1] else "stable"
        
        return {
            "model": model,
            "has_results": True,
            "latest_score": latest.overall_score,
            "latest_passed": latest.passed,
            "score_trend": score_trend,
            "score_history": score_history,
            "suite_scores": {
                name: suite.score 
                for name, suite in latest.suite_results.items()
            },
            "total_tests_run": sum(s.tests_total for s in latest.suite_results.values()),
            "last_tested": latest.completed_at.isoformat(),
            "test_history": [
                {
                    "timestamp": r.completed_at.isoformat(),
                    "score": r.overall_score,
                    "passed": r.passed
                }
                for r in results
            ]
        }
    
    def _convert_to_report_format(self, test_result: TestResult) -> Dict:
        """Convert API TestResult to report generator format."""
        return {
            "model": test_result.model,
            "overall_score": test_result.overall_score,
            "passed": test_result.passed,
            "threshold": test_result.threshold,
            "completed_at": test_result.completed_at,
            "suite_results": test_result.suite_results,
            "summary": test_result.summary
        }