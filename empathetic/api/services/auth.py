"""Authentication service for validators."""
from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import get_settings
from ..models.validation import ValidatorProfile

security = HTTPBearer()
settings = get_settings()


class AuthService:
    """Simple authentication service."""

    @staticmethod
    def create_token(username: str) -> str:
        """Create a JWT token for a validator."""
        payload = {
            "sub": username,
            "exp": datetime.utcnow() + timedelta(days=7),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

    @staticmethod
    def verify_token(token: str) -> Optional[str]:
        """Verify a JWT token and return username."""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            return payload.get("sub")
        except jwt.ExpiredSignatureError:
            return None
        except jwt.JWTError:
            return None


async def get_current_validator(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> ValidatorProfile:
    """Get current validator from token."""
    token = credentials.credentials
    username = AuthService.verify_token(token)

    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # For demo purposes, return a mock validator
    # In production, this would fetch from database
    return ValidatorProfile(
        username=username,
        communities=["disabled", "LGBTQ+"],
        expertise_areas=["employment", "healthcare"],
        organization="Demo Organization",
        is_verified=True
    )
