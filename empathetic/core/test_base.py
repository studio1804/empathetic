from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TestCase:
    """Individual test case"""
    id: str
    category: str
    input: str
    expected_behavior: str
    harmful_patterns: List[str]
    severity: Severity
    metadata: Dict[str, Any] = None

@dataclass
class TestResult:
    """Result from running a test suite"""
    model: str
    suite: str
    tests_passed: int
    tests_total: int
    score: float
    details: Dict[str, Any]
    recommendations: List[str]

class TestSuite(ABC):
    """Base class for test suites"""
    
    def __init__(self, name: str):
        self.name = name
        self.test_cases: List[TestCase] = []
        
    @abstractmethod
    async def run(self, provider, config: Optional[Dict] = None) -> TestResult:
        """Run all tests in the suite"""
        pass
        
    @abstractmethod
    def evaluate_response(self, response: str, test_case: TestCase) -> bool:
        """Evaluate if a response passes the test case"""
        pass
        
    def compile_results(self, results: List[Dict[str, Any]]) -> TestResult:
        """Compile individual test results into suite result"""
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        
        # Weight by severity
        severity_weights = {
            Severity.LOW: 0.5,
            Severity.MEDIUM: 1.0,
            Severity.HIGH: 2.0,
            Severity.CRITICAL: 3.0
        }
        
        weighted_score = 0
        total_weight = 0
        
        for result in results:
            test_case = next(tc for tc in self.test_cases if tc.id == result["test_id"])
            weight = severity_weights[test_case.severity]
            total_weight += weight
            if result["passed"]:
                weighted_score += weight
                
        score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Generate recommendations
        failed_tests = [r for r in results if not r["passed"]]
        recommendations = self._generate_recommendations(failed_tests)
        
        return TestResult(
            model="", # Will be set by caller
            suite=self.name,
            tests_passed=passed,
            tests_total=total,
            score=score,
            details={"individual_results": results},
            recommendations=recommendations
        )
        
    def _generate_recommendations(self, failed_tests: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on failed tests"""
        recommendations = []
        
        if not failed_tests:
            return recommendations
            
        # Group by category
        categories = {}
        for test in failed_tests:
            test_case = next(tc for tc in self.test_cases if tc.id == test["test_id"])
            category = test_case.category
            if category not in categories:
                categories[category] = []
            categories[category].append(test_case)
            
        # Generate category-specific recommendations
        for category, tests in categories.items():
            count = len(tests)
            if count > 1:
                recommendations.append(f"Address {count} issues in {category}")
            else:
                recommendations.append(f"Address issue in {category}: {tests[0].expected_behavior}")
                
        return recommendations