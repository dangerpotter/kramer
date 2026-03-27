"""
LLM Client wrapper that supports both direct Anthropic API and OpenAI-compatible endpoints.

Provides a unified interface for all LLM calls in the system, with retry logic
and support for routing through an OpenAI-compatible gateway.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import anthropic
import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    """Check if an exception is retryable (HTTP 429/500/502/503)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503)
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.InternalServerError):
        return True
    return False


@dataclass
class ContentBlock:
    """A content block in the response."""
    type: str
    text: str = ""
    thinking: str = ""


@dataclass
class Usage:
    """Token usage information."""
    input_tokens: int
    output_tokens: int


@dataclass
class LLMResponse:
    """Response object matching Anthropic's response format."""
    content: List[ContentBlock]
    usage: Usage


class LLMClient:
    """
    Unified LLM client that supports direct Anthropic API or OpenAI-compatible endpoints.

    Checks env var LLM_BACKEND:
    - "openai_compatible": Routes through an OpenAI-compatible endpoint
    - Otherwise: Uses anthropic.Anthropic directly
    """

    def __init__(self):
        self.backend = os.getenv("LLM_BACKEND", "anthropic").lower()
        self.base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:18789/v1/chat/completions")

        if self.backend != "openai_compatible":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY must be set")
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        else:
            self.anthropic_client = None

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def create_message(
        self,
        model: str,
        max_tokens: int,
        temperature: float,
        messages: List[Dict[str, Any]],
        thinking: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Create a message using the configured LLM backend.

        Args:
            model: Model name
            max_tokens: Maximum tokens for response
            temperature: Temperature for sampling
            messages: List of message dicts with role and content
            thinking: Optional extended thinking configuration

        Returns:
            Response object matching Anthropic's response format
        """
        if self.backend == "openai_compatible":
            return self._call_openai_compatible(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                thinking=thinking,
            )
        else:
            return self._call_anthropic(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                thinking=thinking,
            )

    def _call_anthropic(
        self,
        model: str,
        max_tokens: int,
        temperature: float,
        messages: List[Dict[str, Any]],
        thinking: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Call Anthropic API directly."""
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if thinking:
            kwargs["thinking"] = thinking

        return self.anthropic_client.messages.create(**kwargs)

    def _call_openai_compatible(
        self,
        model: str,
        max_tokens: int,
        temperature: float,
        messages: List[Dict[str, Any]],
        thinking: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Call an OpenAI-compatible endpoint and convert response to Anthropic format."""
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        response = httpx.post(
            self.base_url,
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()

        data = response.json()

        # Convert OpenAI format to Anthropic format
        content_blocks = []
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        text = message.get("content", "")

        if text:
            content_blocks.append(ContentBlock(type="text", text=text))

        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
        )

        return LLMResponse(content=content_blocks, usage=usage)


# Singleton instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Return a singleton LLMClient instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
