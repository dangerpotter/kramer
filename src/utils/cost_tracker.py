"""
Cost tracking utility for Claude API usage.

Tracks token usage and calculates costs based on Claude API pricing.
Pricing as of January 2025 - update as needed.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


# Claude API pricing (as of January 2025, in dollars per token)
# Source: https://www.anthropic.com/pricing

# Claude Sonnet 4.5 (claude-sonnet-4-20250514)
CLAUDE_SONNET_4_5_INPUT = 3.00 / 1_000_000   # $3 per million input tokens
CLAUDE_SONNET_4_5_OUTPUT = 15.00 / 1_000_000  # $15 per million output tokens

# Claude Haiku 3.5
CLAUDE_HAIKU_3_5_INPUT = 0.80 / 1_000_000    # $0.80 per million input tokens
CLAUDE_HAIKU_3_5_OUTPUT = 4.00 / 1_000_000   # $4 per million output tokens

# Claude Sonnet 3.5 (older model)
CLAUDE_SONNET_3_5_INPUT = 3.00 / 1_000_000
CLAUDE_SONNET_3_5_OUTPUT = 15.00 / 1_000_000


# Model name mappings to pricing
MODEL_PRICING = {
    # Sonnet 4.5
    "claude-sonnet-4-20250514": (CLAUDE_SONNET_4_5_INPUT, CLAUDE_SONNET_4_5_OUTPUT),
    "claude-sonnet-4": (CLAUDE_SONNET_4_5_INPUT, CLAUDE_SONNET_4_5_OUTPUT),

    # Sonnet 3.5
    "claude-3-5-sonnet-20241022": (CLAUDE_SONNET_3_5_INPUT, CLAUDE_SONNET_3_5_OUTPUT),
    "claude-3-5-sonnet": (CLAUDE_SONNET_3_5_INPUT, CLAUDE_SONNET_3_5_OUTPUT),
    "claude-sonnet-3.5": (CLAUDE_SONNET_3_5_INPUT, CLAUDE_SONNET_3_5_OUTPUT),

    # Haiku 3.5
    "claude-3-5-haiku-20241022": (CLAUDE_HAIKU_3_5_INPUT, CLAUDE_HAIKU_3_5_OUTPUT),
    "claude-3-5-haiku": (CLAUDE_HAIKU_3_5_INPUT, CLAUDE_HAIKU_3_5_OUTPUT),
    "claude-haiku-3.5": (CLAUDE_HAIKU_3_5_INPUT, CLAUDE_HAIKU_3_5_OUTPUT),
}


@dataclass
class CostBreakdown:
    """Detailed breakdown of API call costs."""

    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    model: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "model": self.model,
        }


class CostTracker:
    """
    Utility for tracking Claude API costs.

    Provides methods to calculate costs from API responses and track
    cumulative usage across multiple calls.
    """

    @staticmethod
    def calculate_cost(
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for a Claude API call.

        Args:
            model: Model name (e.g., "claude-sonnet-4-20250514")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in dollars

        Raises:
            ValueError: If model is not recognized
        """
        # Get pricing for model
        pricing = CostTracker._get_model_pricing(model)
        if not pricing:
            raise ValueError(f"Unknown model: {model}")

        input_rate, output_rate = pricing

        # Calculate costs
        input_cost = input_tokens * input_rate
        output_cost = output_tokens * output_rate

        return input_cost + output_cost

    @staticmethod
    def calculate_cost_detailed(
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> CostBreakdown:
        """
        Calculate detailed cost breakdown for a Claude API call.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            CostBreakdown object with detailed information
        """
        # Get pricing for model
        pricing = CostTracker._get_model_pricing(model)
        if not pricing:
            raise ValueError(f"Unknown model: {model}")

        input_rate, output_rate = pricing

        # Calculate costs
        input_cost = input_tokens * input_rate
        output_cost = output_tokens * output_rate
        total_cost = input_cost + output_cost

        return CostBreakdown(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            model=model,
        )

    @staticmethod
    def track_call(model: str, response: Any) -> float:
        """
        Track cost from an Anthropic API response.

        Extracts token usage from the response and calculates cost.

        Args:
            model: Model name used for the call
            response: Anthropic API response object (has .usage attribute)

        Returns:
            Cost in dollars

        Example:
            response = client.messages.create(...)
            cost = CostTracker.track_call("claude-sonnet-4-20250514", response)
        """
        # Extract usage from response
        usage = response.usage

        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        return CostTracker.calculate_cost(model, input_tokens, output_tokens)

    @staticmethod
    def track_call_detailed(model: str, response: Any) -> CostBreakdown:
        """
        Track detailed cost breakdown from an Anthropic API response.

        Args:
            model: Model name used for the call
            response: Anthropic API response object

        Returns:
            CostBreakdown object with detailed information
        """
        # Extract usage from response
        usage = response.usage

        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        return CostTracker.calculate_cost_detailed(model, input_tokens, output_tokens)

    @staticmethod
    def _get_model_pricing(model: str) -> Optional[tuple[float, float]]:
        """
        Get pricing for a model.

        Args:
            model: Model name

        Returns:
            Tuple of (input_rate, output_rate) or None if model not found
        """
        # Direct lookup
        if model in MODEL_PRICING:
            return MODEL_PRICING[model]

        # Fuzzy matching - check if model name contains a known model
        model_lower = model.lower()

        # Check for Sonnet 4
        if "sonnet-4" in model_lower or "sonnet4" in model_lower:
            return (CLAUDE_SONNET_4_5_INPUT, CLAUDE_SONNET_4_5_OUTPUT)

        # Check for Sonnet 3.5
        if "sonnet-3-5" in model_lower or "sonnet-3.5" in model_lower:
            return (CLAUDE_SONNET_3_5_INPUT, CLAUDE_SONNET_3_5_OUTPUT)

        # Check for Haiku 3.5
        if "haiku-3-5" in model_lower or "haiku-3.5" in model_lower:
            return (CLAUDE_HAIKU_3_5_INPUT, CLAUDE_HAIKU_3_5_OUTPUT)

        # Unknown model - return None
        return None

    @staticmethod
    def format_cost(cost: float) -> str:
        """
        Format cost as a human-readable string.

        Args:
            cost: Cost in dollars

        Returns:
            Formatted string (e.g., "$0.0045" or "$1.23")
        """
        if cost < 0.01:
            return f"${cost:.4f}"
        elif cost < 1.0:
            return f"${cost:.3f}"
        else:
            return f"${cost:.2f}"
