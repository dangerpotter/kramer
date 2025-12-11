"""
API endpoint to fetch available Claude models from Anthropic.
"""

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import httpx

router = APIRouter()


class ModelInfo(BaseModel):
    """Model information from Anthropic API."""
    id: str
    display_name: str
    created_at: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response containing available models."""
    models: List[ModelInfo]


@router.get("", response_model=ModelsResponse)
async def list_models():
    """
    Fetch available Claude models from Anthropic API.

    Returns:
        List of available models with their IDs and display names
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY not configured"
        )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=10.0,
            )

            if response.status_code != 200:
                # Fallback to hardcoded models if API fails
                return ModelsResponse(models=_get_fallback_models())

            data = response.json()
            models = []

            for model in data.get("data", []):
                model_id = model.get("id", "")
                # Filter to only include claude-opus and claude-sonnet models
                if "claude-opus" in model_id or "claude-sonnet" in model_id:
                    models.append(ModelInfo(
                        id=model_id,
                        display_name=model.get("display_name", model_id),
                        created_at=model.get("created_at"),
                    ))

            # Sort by created_at descending (newest first)
            models.sort(key=lambda m: m.created_at or "", reverse=True)

            # If no models returned, use fallback
            if not models:
                return ModelsResponse(models=_get_fallback_models())

            return ModelsResponse(models=models)

    except Exception as e:
        print(f"Error fetching models from Anthropic: {e}")
        # Return fallback models on error
        return ModelsResponse(models=_get_fallback_models())


def _get_fallback_models() -> List[ModelInfo]:
    """Return fallback models if API is unavailable."""
    return [
        ModelInfo(
            id="claude-opus-4-5-20251101",
            display_name="Claude Opus 4.5",
        ),
        ModelInfo(
            id="claude-sonnet-4-5-20241022",
            display_name="Claude Sonnet 4.5",
        ),
    ]
