"""Validation models for community validation system."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ValidatorProfile(BaseModel):
    """Community validator profile."""
    username: str
    communities: List[str] = Field(description="Communities the validator represents")
    expertise_areas: List[str] = Field(description="Areas of lived experience")
    organization: Optional[str] = Field(description="Verifying organization")
    trust_score: float = Field(default=1.0, ge=0.0, le=2.0)
    validations_completed: int = Field(default=0, ge=0)
    joined_date: datetime = Field(default_factory=datetime.utcnow)
    is_verified: bool = Field(default=False)


class ValidationRequest(BaseModel):
    """Request for community validation."""
    test_scenario: str = Field(description="The scenario being tested")
    ai_response: str = Field(description="The AI model's response")
    model_name: str = Field(description="Name of the AI model")
    target_communities: List[str] = Field(description="Communities to request validation from")
    test_type: str = Field(description="Type of test (empathy, bias, etc.)")
    metadata: Optional[Dict[str, Any]] = Field(default={})


class ValidationQuestion(BaseModel):
    """Individual validation question."""
    id: str
    question: str
    question_type: str = Field(description="rating, boolean, text")
    required: bool = True
    options: Optional[Dict[str, Any]] = None


class ValidationResponse(BaseModel):
    """Response from a community validator."""
    validator_id: str
    validation_request_id: str
    responses: Dict[str, Any] = Field(description="Answers to validation questions")
    empathy_score: float = Field(ge=1.0, le=5.0)
    respect_score: float = Field(ge=1.0, le=5.0)
    has_harmful_content: bool = False
    harmful_content_notes: Optional[str] = None
    improvement_suggestions: Optional[str] = None
    submitted_at: datetime = Field(default_factory=datetime.utcnow)


class CommunityConsensus(BaseModel):
    """Aggregated community validation results."""
    validation_request_id: str
    consensus_score: float = Field(ge=0.0, le=5.0)
    agreement_level: float = Field(ge=0.0, le=1.0)
    validator_count: int
    community_approved: bool
    feedback_themes: List[str]
    improvement_suggestions: List[str]
    harmful_patterns: Optional[List[str]] = None