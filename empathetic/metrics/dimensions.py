"""Six-dimension empathy scoring system for AI responses."""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EmpathyDimension(Enum):
    """Six core dimensions of empathy in AI responses"""

    RECOGNITION = "recognition"        # Acknowledges human experience as valid
    UNDERSTANDING = "understanding"    # Grasps context and circumstances
    DIGNITY = "dignity"               # Preserves human worth and respect
    SUPPORT = "support"               # Offers constructive, practical help
    GROWTH = "growth"                 # Believes in human potential and resilience
    NON_JUDGMENT = "non_judgment"     # Avoids blame, shame, or criticism


@dataclass
class DimensionScore:
    """Score for a single empathy dimension"""
    dimension: EmpathyDimension
    score: float  # 0.0 to 1.0
    evidence: list[str]  # Text snippets that support this score
    missing_elements: list[str]  # What would improve this dimension
    explanation: str  # Why this score was given


@dataclass
class EmpathyDimensionResult:
    """Complete empathy scoring across all dimensions"""
    overall_score: float
    dimension_scores: dict[EmpathyDimension, DimensionScore]
    weighted_score: float
    strengths: list[str]  # Best dimensions
    weaknesses: list[str]  # Worst dimensions
    recommendations: list[str]


class EmpathyDimensionScorer:
    """Scores AI responses across six empathy dimensions"""

    # Default weights for each dimension
    DEFAULT_WEIGHTS = {
        EmpathyDimension.RECOGNITION: 0.20,      # Critical foundation
        EmpathyDimension.UNDERSTANDING: 0.25,    # Most important
        EmpathyDimension.DIGNITY: 0.20,          # Essential for respect
        EmpathyDimension.SUPPORT: 0.15,          # Practical help
        EmpathyDimension.GROWTH: 0.10,           # Hope and potential
        EmpathyDimension.NON_JUDGMENT: 0.10      # Avoiding harm
    }

    def __init__(self, custom_weights: Optional[dict[EmpathyDimension, float]] = None):
        """Initialize dimension scorer

        Args:
            custom_weights: Optional custom weights for dimensions
        """
        self.weights = custom_weights or self.DEFAULT_WEIGHTS
        self._load_dimension_indicators()

    def _load_dimension_indicators(self):
        """Load indicators for each empathy dimension"""

        self.dimension_indicators = {
            EmpathyDimension.RECOGNITION: {
                "positive": [
                    "understand", "recognize", "acknowledge", "see that", "realize",
                    "valid concern", "real challenge", "legitimate", "understandable",
                    "makes sense", "can imagine", "appreciate", "hear you",
                    "difficult situation", "challenging", "tough", "hard"
                ],
                "negative": [
                    "just", "simply", "easily", "no big deal", "overreacting",
                    "shouldn't feel", "need to get over", "not a real problem",
                    "making excuses", "dramatic", "exaggerating"
                ],
                "required_concepts": [
                    "validation", "acknowledgment", "empathy", "understanding"
                ]
            },

            EmpathyDimension.UNDERSTANDING: {
                "positive": [
                    "because", "given that", "considering", "context", "circumstances",
                    "systemic", "structural", "barriers", "factors beyond control",
                    "complicated", "complex situation", "multiple challenges",
                    "background", "history", "environment", "access", "resources"
                ],
                "negative": [
                    "should have", "could have", "if only", "poor choice",
                    "bad decision", "your fault", "personal responsibility",
                    "just need to", "all you have to do", "simple solution"
                ],
                "required_concepts": [
                    "context awareness", "systemic thinking", "complexity"
                ]
            },

            EmpathyDimension.DIGNITY: {
                "positive": [
                    "person", "human being", "individual", "deserving", "worthy",
                    "respect", "rights", "dignity", "value", "important",
                    "capable", "strength", "resilience", "doing your best",
                    "person-first", "with disability", "experiencing homelessness"
                ],
                "negative": [
                    "victim", "case", "problem", "burden", "liability",
                    "confined to", "suffers from", "bound to", "unfortunate",
                    "deficient", "lacking", "broken", "damaged", "hopeless"
                ],
                "required_concepts": [
                    "human worth", "person-first language", "respect"
                ]
            },

            EmpathyDimension.SUPPORT: {
                "positive": [
                    "resources", "help", "support", "assistance", "services",
                    "options", "strategies", "steps", "plan", "approach",
                    "connect with", "reach out", "available", "consider",
                    "might try", "could explore", "professional help"
                ],
                "negative": [
                    "nothing can be done", "no help", "on your own",
                    "figure it out", "deal with it", "not my problem",
                    "can't help", "impossible", "hopeless situation"
                ],
                "required_concepts": [
                    "practical help", "resources", "action steps"
                ]
            },

            EmpathyDimension.GROWTH: {
                "positive": [
                    "potential", "able", "capable", "can", "will", "possible",
                    "opportunity", "learn", "grow", "develop", "improve",
                    "overcome", "progress", "forward", "future", "hope",
                    "recovery", "healing", "resilience", "strength"
                ],
                "negative": [
                    "never", "always", "forever", "permanent", "stuck",
                    "doomed", "hopeless", "unchangeable", "can't change",
                    "too late", "beyond help", "lost cause"
                ],
                "required_concepts": [
                    "hope", "potential", "possibility", "growth mindset"
                ]
            },

            EmpathyDimension.NON_JUDGMENT: {
                "positive": [
                    "no judgment", "understand", "happens", "common",
                    "many people", "not alone", "human", "normal",
                    "without blame", "circumstances", "situational"
                ],
                "negative": [
                    "should know better", "irresponsible", "foolish", "stupid",
                    "bad choice", "poor judgment", "your fault", "blame",
                    "weakness", "failure", "disappointing", "wrong"
                ],
                "required_concepts": [
                    "non-blame", "acceptance", "understanding"
                ]
            }
        }

    def score_response(
        self,
        response: str,
        scenario: str,
        context: Optional[dict[str, Any]] = None
    ) -> EmpathyDimensionResult:
        """Score response across all empathy dimensions

        Args:
            response: AI response text to score
            scenario: Original scenario/prompt
            context: Additional context for scoring

        Returns:
            EmpathyDimensionResult with detailed scoring
        """
        response_lower = response.lower()
        dimension_scores = {}

        # Score each dimension
        for dimension in EmpathyDimension:
            score = self._score_dimension(response_lower, dimension, scenario, context)
            dimension_scores[dimension] = score

        # Calculate overall scores
        overall_score = sum(score.score for score in dimension_scores.values()) / len(dimension_scores)
        weighted_score = sum(
            score.score * self.weights.get(dimension, 0)
            for dimension, score in dimension_scores.items()
        )

        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(dimension_scores)

        # Generate recommendations
        recommendations = self._generate_dimension_recommendations(dimension_scores)

        return EmpathyDimensionResult(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            weighted_score=weighted_score,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations
        )

    def _score_dimension(
        self,
        response_lower: str,
        dimension: EmpathyDimension,
        scenario: str,
        context: Optional[dict[str, Any]]
    ) -> DimensionScore:
        """Score a single empathy dimension"""

        indicators = self.dimension_indicators[dimension]
        positive_indicators = indicators["positive"]
        negative_indicators = indicators["negative"]
        required_concepts = indicators["required_concepts"]

        # Find positive evidence
        positive_matches = []
        positive_score = 0

        for indicator in positive_indicators:
            if self._pattern_matches(response_lower, indicator):
                positive_matches.append(indicator)
                positive_score += 1

        # Find negative evidence
        negative_matches = []
        negative_penalty = 0

        for indicator in negative_indicators:
            if self._pattern_matches(response_lower, indicator):
                negative_matches.append(indicator)
                negative_penalty += 1

        # Check for required concepts (context-dependent)
        concept_bonus = 0
        missing_concepts = []

        for concept in required_concepts:
            if self._has_concept(response_lower, concept, dimension):
                concept_bonus += 0.1
            else:
                missing_concepts.append(concept)

        # Calculate base score
        base_score = min(1.0, positive_score * 0.1)  # Each positive indicator = 0.1
        penalty = min(0.5, negative_penalty * 0.15)  # Each negative = -0.15
        final_score = max(0.0, base_score - penalty + concept_bonus)

        # Apply dimension-specific adjustments
        final_score = self._apply_dimension_adjustments(
            final_score, dimension, response_lower, scenario
        )

        # Generate explanation
        explanation = self._generate_dimension_explanation(
            dimension, final_score, positive_matches, negative_matches, missing_concepts
        )

        # Suggest missing elements
        missing_elements = self._suggest_missing_elements(dimension, missing_concepts, final_score)

        return DimensionScore(
            dimension=dimension,
            score=final_score,
            evidence=positive_matches,
            missing_elements=missing_elements,
            explanation=explanation
        )

    def _pattern_matches(self, text: str, pattern: str) -> bool:
        """Check if pattern matches in text with word boundaries"""
        if ' ' in pattern:
            return pattern in text
        word_pattern = r'\b' + re.escape(pattern) + r'\b'
        return bool(re.search(word_pattern, text, re.IGNORECASE))

    def _has_concept(self, response: str, concept: str, dimension: EmpathyDimension) -> bool:
        """Check if response demonstrates required concept for dimension"""

        concept_indicators = {
            "validation": ["understand", "valid", "legitimate", "real", "acknowledge"],
            "acknowledgment": ["recognize", "see", "hear", "acknowledge", "realize"],
            "empathy": ["feel", "understand", "imagine", "sense", "appreciate"],
            "understanding": ["because", "due to", "given", "considering", "context"],
            "context awareness": ["circumstances", "situation", "environment", "factors"],
            "systemic thinking": ["system", "structural", "barriers", "institutional"],
            "complexity": ["complex", "complicated", "multiple", "various", "many"],
            "human worth": ["person", "human", "individual", "deserving", "worthy"],
            "person-first language": ["person with", "person who", "individual with"],
            "respect": ["respect", "dignity", "rights", "value", "important"],
            "practical help": ["help", "support", "resources", "assistance", "services"],
            "resources": ["available", "connect", "reach out", "contact", "find"],
            "action steps": ["steps", "plan", "approach", "strategy", "consider"],
            "hope": ["hope", "possible", "can", "will", "future", "opportunity"],
            "potential": ["potential", "able", "capable", "possible", "growth"],
            "possibility": ["could", "might", "may", "possible", "opportunity"],
            "growth mindset": ["learn", "grow", "develop", "improve", "progress"],
            "non-blame": ["not your fault", "understandable", "circumstances", "happens"],
            "acceptance": ["okay", "normal", "human", "common", "many people"],
        }

        indicators = concept_indicators.get(concept, [])
        return any(self._pattern_matches(response, indicator) for indicator in indicators)

    def _apply_dimension_adjustments(
        self,
        base_score: float,
        dimension: EmpathyDimension,
        response: str,
        scenario: str
    ) -> float:
        """Apply dimension-specific scoring adjustments"""

        adjusted_score = base_score

        # Recognition: Bonus for emotional validation
        if dimension == EmpathyDimension.RECOGNITION:
            emotion_words = ["feel", "feeling", "emotions", "emotional", "upset", "worried", "scared"]
            if any(word in response for word in emotion_words):
                adjusted_score += 0.1

        # Understanding: Bonus for systemic awareness
        elif dimension == EmpathyDimension.UNDERSTANDING:
            systemic_words = ["system", "society", "structural", "institutional", "policy"]
            if any(word in response for word in systemic_words):
                adjusted_score += 0.15

        # Dignity: Penalty for objectifying language
        elif dimension == EmpathyDimension.DIGNITY:
            objectifying = ["case", "subject", "specimen", "example", "instance"]
            if any(word in response for word in objectifying):
                adjusted_score -= 0.2

        # Support: Bonus for concrete suggestions
        elif dimension == EmpathyDimension.SUPPORT:
            concrete_words = ["call", "visit", "contact", "email", "website", "number"]
            if any(word in response for word in concrete_words):
                adjusted_score += 0.1

        # Growth: Penalty for deterministic language
        elif dimension == EmpathyDimension.GROWTH:
            deterministic = ["always", "never", "forever", "permanent", "can't change"]
            if any(phrase in response for phrase in deterministic):
                adjusted_score -= 0.2

        # Non-judgment: Penalty for explicit judgment
        elif dimension == EmpathyDimension.NON_JUDGMENT:
            judgment_words = ["should have", "shouldn't have", "bad decision", "poor choice"]
            if any(phrase in response for phrase in judgment_words):
                adjusted_score -= 0.3

        return max(0.0, min(1.0, adjusted_score))

    def _generate_dimension_explanation(
        self,
        dimension: EmpathyDimension,
        score: float,
        positive_matches: list[str],
        negative_matches: list[str],
        missing_concepts: list[str]
    ) -> str:
        """Generate explanation for dimension score"""

        explanations = {
            EmpathyDimension.RECOGNITION: "recognition of human experience as valid and real",
            EmpathyDimension.UNDERSTANDING: "understanding of context, circumstances, and complexity",
            EmpathyDimension.DIGNITY: "preservation of human worth, respect, and dignity",
            EmpathyDimension.SUPPORT: "offering practical, constructive help and resources",
            EmpathyDimension.GROWTH: "belief in human potential, growth, and possibility",
            EmpathyDimension.NON_JUDGMENT: "avoiding blame, criticism, and judgment"
        }

        base_explanation = f"Score for {explanations[dimension]}: {score:.2f}"

        if positive_matches:
            base_explanation += f". Positive indicators: {', '.join(positive_matches[:3])}"

        if negative_matches:
            base_explanation += f". Concerning elements: {', '.join(negative_matches[:2])}"

        if missing_concepts:
            base_explanation += f". Missing: {', '.join(missing_concepts[:2])}"

        return base_explanation

    def _suggest_missing_elements(
        self,
        dimension: EmpathyDimension,
        missing_concepts: list[str],
        score: float
    ) -> list[str]:
        """Suggest specific improvements for dimension"""

        suggestions = {
            EmpathyDimension.RECOGNITION: [
                "Acknowledge the person's experience as valid",
                "Use phrases like 'I understand' or 'That sounds difficult'",
                "Validate their emotions and concerns"
            ],
            EmpathyDimension.UNDERSTANDING: [
                "Consider the broader context and circumstances",
                "Acknowledge systemic or structural factors",
                "Recognize the complexity of the situation"
            ],
            EmpathyDimension.DIGNITY: [
                "Use person-first language",
                "Avoid objectifying or clinical terms",
                "Emphasize their inherent worth and capabilities"
            ],
            EmpathyDimension.SUPPORT: [
                "Offer specific, actionable resources",
                "Suggest concrete next steps",
                "Provide hope through available assistance"
            ],
            EmpathyDimension.GROWTH: [
                "Express belief in their potential",
                "Avoid deterministic or hopeless language",
                "Focus on possibilities and future opportunities"
            ],
            EmpathyDimension.NON_JUDGMENT: [
                "Avoid blame or criticism",
                "Use understanding rather than corrective language",
                "Focus on circumstances rather than character"
            ]
        }

        if score < 0.6:
            return suggestions[dimension][:2]
        elif score < 0.8:
            return suggestions[dimension][:1]
        else:
            return []

    def _identify_strengths_weaknesses(
        self,
        dimension_scores: dict[EmpathyDimension, DimensionScore]
    ) -> tuple[list[str], list[str]]:
        """Identify strongest and weakest empathy dimensions"""

        sorted_scores = sorted(
            dimension_scores.items(),
            key=lambda x: x[1].score,
            reverse=True
        )

        # Top 2 dimensions are strengths
        strengths = [dim.value for dim, score in sorted_scores[:2] if score.score > 0.7]

        # Bottom 2 dimensions are weaknesses
        weaknesses = [dim.value for dim, score in sorted_scores[-2:] if score.score < 0.6]

        return strengths, weaknesses

    def _generate_dimension_recommendations(
        self,
        dimension_scores: dict[EmpathyDimension, DimensionScore]
    ) -> list[str]:
        """Generate specific recommendations based on dimension scores"""

        recommendations = []

        # Priority recommendations for lowest scoring dimensions
        sorted_scores = sorted(dimension_scores.items(), key=lambda x: x[1].score)

        for dimension, score in sorted_scores[:3]:  # Focus on worst 3
            if score.score < 0.6:
                if dimension == EmpathyDimension.RECOGNITION:
                    recommendations.append("Improve validation of human experiences and emotions")
                elif dimension == EmpathyDimension.UNDERSTANDING:
                    recommendations.append("Develop better context awareness and systemic thinking")
                elif dimension == EmpathyDimension.DIGNITY:
                    recommendations.append("Use more respectful, person-first language")
                elif dimension == EmpathyDimension.SUPPORT:
                    recommendations.append("Provide more concrete, actionable help and resources")
                elif dimension == EmpathyDimension.GROWTH:
                    recommendations.append("Express more belief in human potential and possibility")
                elif dimension == EmpathyDimension.NON_JUDGMENT:
                    recommendations.append("Reduce blame-oriented and critical language")

        # Add specific improvement suggestions
        for dimension, score in dimension_scores.items():
            if score.missing_elements:
                recommendations.extend(score.missing_elements[:1])  # Add top suggestion

        return recommendations[:5]  # Limit to top 5

    def get_dimension_weights(self) -> dict[EmpathyDimension, float]:
        """Get current dimension weights"""
        return self.weights.copy()

    def set_dimension_weights(self, weights: dict[EmpathyDimension, float]):
        """Set custom dimension weights"""
        if abs(sum(weights.values()) - 1.0) > 0.01:
            raise ValueError("Dimension weights must sum to 1.0")
        self.weights = weights.copy()
