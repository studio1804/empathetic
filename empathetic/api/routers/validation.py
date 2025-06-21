"""Community validation endpoints."""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime

from ..models.validation import (
    ValidationRequest, 
    ValidationResponse,
    ValidatorProfile,
    CommunityConsensus
)
from ..services.validation import ValidationService
from ..services.auth import get_current_validator

router = APIRouter()
validation_service = ValidationService()


@router.post("/request")
async def create_validation_request(request: ValidationRequest):
    """Create a new validation request for community review."""
    try:
        validation_id = await validation_service.create_request(request)
        return {"validation_id": validation_id, "status": "pending"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pending")
async def get_pending_validations(
    communities: Optional[List[str]] = Query(None),
    expertise: Optional[List[str]] = Query(None),
    limit: int = 10,
    current_validator: ValidatorProfile = Depends(get_current_validator)
):
    """Get pending validation requests for a validator."""
    validations = await validation_service.get_pending_validations(
        validator=current_validator,
        communities=communities,
        expertise=expertise,
        limit=limit
    )
    return {"validations": validations}


@router.post("/submit/{validation_id}")
async def submit_validation(
    validation_id: str,
    response: ValidationResponse,
    current_validator: ValidatorProfile = Depends(get_current_validator)
):
    """Submit a validation response."""
    try:
        response.validator_id = current_validator.username
        response.validation_request_id = validation_id
        
        await validation_service.submit_response(response)
        
        return {
            "status": "submitted",
            "validation_id": validation_id,
            "validator": current_validator.username
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/consensus/{validation_id}")
async def get_consensus(validation_id: str):
    """Get community consensus for a validation request."""
    consensus = await validation_service.calculate_consensus(validation_id)
    if not consensus:
        raise HTTPException(status_code=404, detail="Validation not found")
    return consensus


@router.post("/register")
async def register_validator(profile: ValidatorProfile):
    """Register a new community validator."""
    try:
        validator_id = await validation_service.register_validator(profile)
        return {
            "validator_id": validator_id,
            "status": "pending_verification",
            "message": "Please complete verification with your organization"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/validators/stats")
async def get_validator_stats():
    """Get statistics about the validator community."""
    stats = await validation_service.get_validator_stats()
    return stats


@router.get("/questions/{test_type}")
async def get_validation_questions(test_type: str):
    """Get validation questions for a specific test type."""
    questions = validation_service.get_questions_for_test_type(test_type)
    return {"test_type": test_type, "questions": questions}