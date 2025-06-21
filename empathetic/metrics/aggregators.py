import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from .calculators import MetricsCalculator


@dataclass
class AggregatedResults:
    """Aggregated test results across multiple runs"""
    model: str
    total_runs: int
    average_score: float
    best_score: float
    worst_score: float
    score_trend: str
    suite_averages: dict[str, float]
    last_updated: str

class ResultsAggregator:
    """Aggregate and analyze results across multiple test runs"""

    def __init__(self):
        self.calculator = MetricsCalculator()

    def aggregate_multiple_runs(self, results_list: list[dict[str, Any]]) -> AggregatedResults:
        """Aggregate results from multiple test runs"""
        if not results_list:
            raise ValueError("No results provided for aggregation")

        model = results_list[0].get('model', 'unknown')
        overall_scores = []
        suite_scores = {}

        # Extract scores from each run
        for result in results_list:
            if 'overall_score' in result:
                overall_scores.append(result['overall_score'])
            elif 'score' in result:
                overall_scores.append(result['score'])

            # Collect suite scores
            suite_results = result.get('suite_results', {})
            for suite_name, suite_result in suite_results.items():
                if suite_name not in suite_scores:
                    suite_scores[suite_name] = []

                if hasattr(suite_result, 'score'):
                    suite_scores[suite_name].append(suite_result.score)
                elif isinstance(suite_result, dict) and 'score' in suite_result:
                    suite_scores[suite_name].append(suite_result['score'])

        # Calculate aggregated metrics
        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        best_score = max(overall_scores) if overall_scores else 0.0
        worst_score = min(overall_scores) if overall_scores else 0.0

        # Calculate suite averages
        suite_averages = {}
        for suite_name, scores in suite_scores.items():
            suite_averages[suite_name] = sum(scores) / len(scores) if scores else 0.0

        # Determine trend
        trend_info = self.calculator.calculate_trend(overall_scores)

        return AggregatedResults(
            model=model,
            total_runs=len(results_list),
            average_score=avg_score,
            best_score=best_score,
            worst_score=worst_score,
            score_trend=trend_info['trend'],
            suite_averages=suite_averages,
            last_updated=datetime.now().isoformat()
        )

    def compare_models(self, model_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        """Compare performance across different models"""
        model_summaries = {}

        for model_name, results in model_results.items():
            if results:
                aggregated = self.aggregate_multiple_runs(results)
                model_summaries[model_name] = {
                    'average_score': aggregated.average_score,
                    'best_score': aggregated.best_score,
                    'total_runs': aggregated.total_runs,
                    'suite_averages': aggregated.suite_averages
                }

        # Rank models by average score
        ranked_models = sorted(
            model_summaries.items(),
            key=lambda x: x[1]['average_score'],
            reverse=True
        )

        return {
            'model_rankings': ranked_models,
            'best_model': ranked_models[0][0] if ranked_models else None,
            'comparison_date': datetime.now().isoformat(),
            'models_compared': len(model_summaries)
        }

    def generate_performance_report(self, aggregated_results: AggregatedResults) -> dict[str, Any]:
        """Generate detailed performance report"""
        report = {
            'summary': asdict(aggregated_results),
            'analysis': self._analyze_performance(aggregated_results),
            'recommendations': self._generate_recommendations(aggregated_results),
            'generated_at': datetime.now().isoformat()
        }

        return report

    def _analyze_performance(self, results: AggregatedResults) -> dict[str, Any]:
        """Analyze performance patterns"""
        analysis = {
            'overall_grade': self._calculate_grade(results.average_score),
            'consistency': self._calculate_consistency(results),
            'strengths': [],
            'weaknesses': []
        }

        # Identify strengths and weaknesses
        for suite_name, score in results.suite_averages.items():
            if score >= 0.9:
                analysis['strengths'].append(f"Excellent {suite_name} performance")
            elif score < 0.7:
                analysis['weaknesses'].append(f"Poor {suite_name} performance")

        # Score variability
        score_range = results.best_score - results.worst_score
        if score_range < 0.1:
            analysis['consistency'] = 'high'
        elif score_range < 0.2:
            analysis['consistency'] = 'medium'
        else:
            analysis['consistency'] = 'low'

        return analysis

    def _calculate_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 0.95:
            return 'A+'
        elif score >= 0.9:
            return 'A'
        elif score >= 0.85:
            return 'A-'
        elif score >= 0.8:
            return 'B+'
        elif score >= 0.75:
            return 'B'
        elif score >= 0.7:
            return 'B-'
        elif score >= 0.65:
            return 'C+'
        elif score >= 0.6:
            return 'C'
        else:
            return 'F'

    def _calculate_consistency(self, results: AggregatedResults) -> str:
        """Calculate performance consistency"""
        if results.total_runs < 3:
            return 'insufficient_data'

        score_range = results.best_score - results.worst_score
        if score_range < 0.05:
            return 'very_high'
        elif score_range < 0.1:
            return 'high'
        elif score_range < 0.2:
            return 'medium'
        else:
            return 'low'

    def _generate_recommendations(self, results: AggregatedResults) -> list[str]:
        """Generate recommendations based on aggregated results"""
        recommendations = []

        # Overall score recommendations
        if results.average_score < 0.7:
            recommendations.append("Consider fundamental model improvements or retraining")

        # Suite-specific recommendations
        weak_suites = [name for name, score in results.suite_averages.items() if score < 0.8]
        if weak_suites:
            recommendations.append(f"Focus improvement efforts on: {', '.join(weak_suites)}")

        # Consistency recommendations
        score_range = results.best_score - results.worst_score
        if score_range > 0.2:
            recommendations.append("Improve model consistency across different inputs")

        # Trend recommendations
        if results.score_trend == 'declining':
            recommendations.append("Investigate causes of declining performance")
        elif results.score_trend == 'stable' and results.average_score < 0.9:
            recommendations.append("Consider new approaches to improve beyond current plateau")

        return recommendations

    def export_results(self, aggregated_results: AggregatedResults, format: str = 'json') -> str:
        """Export aggregated results in specified format"""
        if format.lower() == 'json':
            return json.dumps(asdict(aggregated_results), indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
