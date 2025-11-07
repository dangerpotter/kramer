"""
Generic API response models.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class SuccessResponse(BaseModel):
    """Success response."""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
