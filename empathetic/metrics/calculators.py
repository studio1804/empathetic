from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class MetricScore:
    """Individual metric score"""
    name: str
    value: float
    weight: float
    description: str

class MetricsCalculator:
    """Calculate various metrics for test results"""
    
    def __init__(self):
        self.default_weights = {
            'bias': 0.3,
            'alignment': 0.3,
            'fairness': 0.2,
            'safety': 0.2
        }
        
    def calculate_overall(self, suite_results: Dict[str, Any], weights: Dict[str, float] = None) -> float:
        """Calculate weighted overall score across all suites"""
        if not suite_results:
            return 0.0
            
        weights = weights or self.default_weights
        weighted_sum = 0
        total_weight = 0
        
        for suite_name, result in suite_results.items():
            if hasattr(result, 'score'):
                weight = weights.get(suite_name, 0.1)
                weighted_sum += result.score * weight
                total_weight += weight
            elif isinstance(result, dict) and 'score' in result:
                weight = weights.get(suite_name, 0.1)
                weighted_sum += result['score'] * weight
                total_weight += weight
                
        return weighted_sum / total_weight if total_weight > 0 else 0.0
        
    def calculate_suite_metrics(self, suite_results: Dict[str, Any]) -> List[MetricScore]:
        """Calculate detailed metrics for each suite"""
        metrics = []
        
        for suite_name, result in suite_results.items():
            if hasattr(result, 'score'):
                score = result.score
                passed = result.tests_passed
                total = result.tests_total
            elif isinstance(result, dict):
                score = result.get('score', 0)
                passed = result.get('tests_passed', 0)
                total = result.get('tests_total', 1)
            else:
                continue
                
            # Basic metrics
            metrics.append(MetricScore(
                name=f"{suite_name}_score",
                value=score,
                weight=self.default_weights.get(suite_name, 0.1),
                description=f"Overall {suite_name} score"
            ))
            
            # Pass rate
            pass_rate = passed / total if total > 0 else 0
            metrics.append(MetricScore(
                name=f"{suite_name}_pass_rate",
                value=pass_rate,
                weight=1.0,
                description=f"{suite_name} test pass rate"
            ))
            
        return metrics
        
    def calculate_confidence_intervals(self, results: List[float], confidence: float = 0.95) -> Dict[str, float]:
        """Calculate confidence intervals for scores"""
        if not results:
            return {"lower": 0.0, "upper": 0.0, "mean": 0.0}
            
        results_array = np.array(results)
        mean = np.mean(results_array)
        std = np.std(results_array, ddof=1)
        n = len(results_array)
        
        # Using t-distribution for small samples
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_error = t_value * (std / np.sqrt(n))
        
        return {
            "lower": max(0.0, mean - margin_error),
            "upper": min(1.0, mean + margin_error),
            "mean": mean,
            "std": std
        }
        
    def calculate_severity_weighted_score(self, test_results: List[Dict[str, Any]]) -> float:
        """Calculate score weighted by test severity"""
        if not test_results:
            return 0.0
            
        severity_weights = {
            'low': 0.5,
            'medium': 1.0,
            'high': 2.0,
            'critical': 3.0
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for result in test_results:
            severity = result.get('severity', 'medium')
            if hasattr(severity, 'value'):
                severity = severity.value
                
            weight = severity_weights.get(severity, 1.0)
            total_weight += weight
            
            if result.get('passed', False):
                weighted_sum += weight
                
        return weighted_sum / total_weight if total_weight > 0 else 0.0
        
    def calculate_category_breakdown(self, test_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Break down results by category"""
        categories = {}
        
        for result in test_results:
            category = result.get('category', 'unknown')
            if category not in categories:
                categories[category] = {
                    'total': 0,
                    'passed': 0,
                    'failed': 0,
                    'scores': []
                }
                
            categories[category]['total'] += 1
            if result.get('passed', False):
                categories[category]['passed'] += 1
                categories[category]['scores'].append(1.0)
            else:
                categories[category]['failed'] += 1
                categories[category]['scores'].append(0.0)
                
        # Calculate category scores
        for category_data in categories.values():
            if category_data['total'] > 0:
                category_data['pass_rate'] = category_data['passed'] / category_data['total']
                category_data['score'] = np.mean(category_data['scores'])
            else:
                category_data['pass_rate'] = 0.0
                category_data['score'] = 0.0
                
        return categories
        
    def calculate_trend(self, historical_scores: List[float]) -> Dict[str, Any]:
        """Calculate trend in scores over time"""
        if len(historical_scores) < 2:
            return {"trend": "insufficient_data", "slope": 0.0}
            
        x = np.arange(len(historical_scores))
        y = np.array(historical_scores)
        
        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Determine trend
        if abs(slope) < 0.01:
            trend = "stable"
        elif slope > 0:
            trend = "improving"
        else:
            trend = "declining"
            
        return {
            "trend": trend,
            "slope": slope,
            "current_score": historical_scores[-1],
            "change_from_first": historical_scores[-1] - historical_scores[0]
        }