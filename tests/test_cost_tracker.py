"""
Tests for the CostTracker utility module.
"""

import pytest
from dataclasses import dataclass
from src.utils.cost_tracker import (
    CostTracker,
    CostBreakdown,
    MODEL_PRICING,
    CLAUDE_SONNET_4_5_INPUT,
    CLAUDE_SONNET_4_5_OUTPUT,
    CLAUDE_HAIKU_3_5_INPUT,
    CLAUDE_HAIKU_3_5_OUTPUT,
)


class TestCostTrackerBasics:
    """Test basic CostTracker functionality."""

    def test_model_pricing_constants(self):
        """Test that pricing constants are defined correctly."""
        # Check Sonnet 4.5 pricing
        assert CLAUDE_SONNET_4_5_INPUT == 3.00 / 1_000_000
        assert CLAUDE_SONNET_4_5_OUTPUT == 15.00 / 1_000_000

        # Check Haiku 3.5 pricing
        assert CLAUDE_HAIKU_3_5_INPUT == 0.80 / 1_000_000
        assert CLAUDE_HAIKU_3_5_OUTPUT == 4.00 / 1_000_000

    def test_model_pricing_dict(self):
        """Test MODEL_PRICING dictionary contains expected models."""
        assert "claude-sonnet-4-20250514" in MODEL_PRICING
        assert "claude-sonnet-4" in MODEL_PRICING
        assert "claude-3-5-sonnet-20241022" in MODEL_PRICING
        assert "claude-3-5-haiku-20241022" in MODEL_PRICING


class TestCostCalculation:
    """Test cost calculation methods."""

    def test_calculate_cost_sonnet_4(self):
        """Test cost calculation for Sonnet 4."""
        cost = CostTracker.calculate_cost(
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500
        )

        expected_input = 1000 * CLAUDE_SONNET_4_5_INPUT
        expected_output = 500 * CLAUDE_SONNET_4_5_OUTPUT
        expected_total = expected_input + expected_output

        assert cost == pytest.approx(expected_total)

    def test_calculate_cost_haiku(self):
        """Test cost calculation for Haiku 3.5."""
        cost = CostTracker.calculate_cost(
            model="claude-3-5-haiku-20241022",
            input_tokens=1000,
            output_tokens=500
        )

        expected_input = 1000 * CLAUDE_HAIKU_3_5_INPUT
        expected_output = 500 * CLAUDE_HAIKU_3_5_OUTPUT
        expected_total = expected_input + expected_output

        assert cost == pytest.approx(expected_total)

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        cost = CostTracker.calculate_cost(
            model="claude-sonnet-4-20250514",
            input_tokens=0,
            output_tokens=0
        )

        assert cost == 0.0

    def test_calculate_cost_large_tokens(self):
        """Test cost calculation with large token counts."""
        cost = CostTracker.calculate_cost(
            model="claude-sonnet-4-20250514",
            input_tokens=1_000_000,
            output_tokens=500_000
        )

        expected_input = 1_000_000 * CLAUDE_SONNET_4_5_INPUT  # $3
        expected_output = 500_000 * CLAUDE_SONNET_4_5_OUTPUT  # $7.5
        expected_total = expected_input + expected_output  # $10.5

        assert cost == pytest.approx(expected_total)
        assert cost == pytest.approx(10.5)

    def test_calculate_cost_unknown_model(self):
        """Test cost calculation with unknown model raises error."""
        with pytest.raises(ValueError, match="Unknown model"):
            CostTracker.calculate_cost(
                model="unknown-model-xyz",
                input_tokens=1000,
                output_tokens=500
            )


class TestCostBreakdown:
    """Test detailed cost breakdown."""

    def test_calculate_cost_detailed(self):
        """Test detailed cost breakdown."""
        breakdown = CostTracker.calculate_cost_detailed(
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500
        )

        assert isinstance(breakdown, CostBreakdown)
        assert breakdown.input_tokens == 1000
        assert breakdown.output_tokens == 500
        assert breakdown.model == "claude-sonnet-4-20250514"

        expected_input_cost = 1000 * CLAUDE_SONNET_4_5_INPUT
        expected_output_cost = 500 * CLAUDE_SONNET_4_5_OUTPUT

        assert breakdown.input_cost == pytest.approx(expected_input_cost)
        assert breakdown.output_cost == pytest.approx(expected_output_cost)
        assert breakdown.total_cost == pytest.approx(expected_input_cost + expected_output_cost)

    def test_cost_breakdown_to_dict(self):
        """Test converting cost breakdown to dictionary."""
        breakdown = CostTracker.calculate_cost_detailed(
            model="claude-sonnet-4",
            input_tokens=100,
            output_tokens=50
        )

        breakdown_dict = breakdown.to_dict()

        assert "input_tokens" in breakdown_dict
        assert "output_tokens" in breakdown_dict
        assert "input_cost" in breakdown_dict
        assert "output_cost" in breakdown_dict
        assert "total_cost" in breakdown_dict
        assert "model" in breakdown_dict

        assert breakdown_dict["input_tokens"] == 100
        assert breakdown_dict["output_tokens"] == 50
        assert breakdown_dict["model"] == "claude-sonnet-4"

    def test_cost_breakdown_unknown_model(self):
        """Test detailed breakdown with unknown model raises error."""
        with pytest.raises(ValueError, match="Unknown model"):
            CostTracker.calculate_cost_detailed(
                model="unknown-model",
                input_tokens=1000,
                output_tokens=500
            )


class TestTrackCall:
    """Test API response tracking."""

    def test_track_call(self):
        """Test tracking cost from API response."""
        # Create a mock response object
        @dataclass
        class MockUsage:
            input_tokens: int
            output_tokens: int

        @dataclass
        class MockResponse:
            usage: MockUsage

        response = MockResponse(usage=MockUsage(input_tokens=1000, output_tokens=500))

        cost = CostTracker.track_call("claude-sonnet-4-20250514", response)

        expected = CostTracker.calculate_cost(
            "claude-sonnet-4-20250514",
            1000, 500
        )

        assert cost == pytest.approx(expected)

    def test_track_call_detailed(self):
        """Test detailed tracking from API response."""
        @dataclass
        class MockUsage:
            input_tokens: int
            output_tokens: int

        @dataclass
        class MockResponse:
            usage: MockUsage

        response = MockResponse(usage=MockUsage(input_tokens=1000, output_tokens=500))

        breakdown = CostTracker.track_call_detailed("claude-sonnet-4", response)

        assert isinstance(breakdown, CostBreakdown)
        assert breakdown.input_tokens == 1000
        assert breakdown.output_tokens == 500


class TestModelPricingLookup:
    """Test model pricing lookup functionality."""

    def test_get_model_pricing_direct(self):
        """Test direct model name lookup."""
        pricing = CostTracker._get_model_pricing("claude-sonnet-4-20250514")

        assert pricing is not None
        assert pricing == (CLAUDE_SONNET_4_5_INPUT, CLAUDE_SONNET_4_5_OUTPUT)

    def test_get_model_pricing_fuzzy_sonnet4(self):
        """Test fuzzy matching for Sonnet 4."""
        pricing = CostTracker._get_model_pricing("some-custom-sonnet-4-model")

        assert pricing is not None
        assert pricing == (CLAUDE_SONNET_4_5_INPUT, CLAUDE_SONNET_4_5_OUTPUT)

    def test_get_model_pricing_fuzzy_sonnet35(self):
        """Test fuzzy matching for Sonnet 3.5."""
        pricing = CostTracker._get_model_pricing("custom-sonnet-3-5-variant")

        assert pricing is not None

    def test_get_model_pricing_fuzzy_haiku35(self):
        """Test fuzzy matching for Haiku 3.5."""
        pricing = CostTracker._get_model_pricing("custom-haiku-3-5-variant")

        assert pricing is not None
        assert pricing == (CLAUDE_HAIKU_3_5_INPUT, CLAUDE_HAIKU_3_5_OUTPUT)

    def test_get_model_pricing_unknown(self):
        """Test unknown model returns None."""
        pricing = CostTracker._get_model_pricing("completely-unknown-model")

        assert pricing is None


class TestFormatCost:
    """Test cost formatting."""

    def test_format_cost_small(self):
        """Test formatting small costs."""
        formatted = CostTracker.format_cost(0.0001)
        assert formatted == "$0.0001"

    def test_format_cost_medium(self):
        """Test formatting medium costs."""
        formatted = CostTracker.format_cost(0.123)
        assert formatted == "$0.123"

    def test_format_cost_large(self):
        """Test formatting large costs."""
        formatted = CostTracker.format_cost(12.345)
        assert formatted == "$12.35"

    def test_format_cost_zero(self):
        """Test formatting zero cost."""
        formatted = CostTracker.format_cost(0)
        assert "$0" in formatted

    def test_format_cost_boundary_001(self):
        """Test formatting at $0.01 boundary."""
        formatted = CostTracker.format_cost(0.01)
        assert formatted == "$0.010"

    def test_format_cost_boundary_1(self):
        """Test formatting at $1 boundary."""
        formatted = CostTracker.format_cost(1.0)
        assert formatted == "$1.00"


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_typical_chat_cost(self):
        """Test typical chat interaction cost."""
        # Typical short chat: ~500 input, ~200 output
        cost = CostTracker.calculate_cost(
            model="claude-sonnet-4",
            input_tokens=500,
            output_tokens=200
        )

        # Should be a small fraction of a cent
        assert cost < 0.01
        assert cost > 0

    def test_code_analysis_cost(self):
        """Test code analysis cost (larger context)."""
        # Code analysis: ~10000 input, ~2000 output
        cost = CostTracker.calculate_cost(
            model="claude-sonnet-4",
            input_tokens=10000,
            output_tokens=2000
        )

        # Should be a few cents
        assert cost < 0.10
        assert cost > 0.01

    def test_extended_thinking_cost(self):
        """Test extended thinking cost (very large output)."""
        # Extended thinking: ~5000 input, ~10000 output
        cost = CostTracker.calculate_cost(
            model="claude-sonnet-4",
            input_tokens=5000,
            output_tokens=10000
        )

        # Should be ~$0.16
        expected = (5000 * 0.000003) + (10000 * 0.000015)  # 0.015 + 0.15 = 0.165
        assert cost == pytest.approx(expected)

    def test_haiku_vs_sonnet_cost_comparison(self):
        """Test that Haiku is cheaper than Sonnet."""
        tokens = {"input": 1000, "output": 500}

        haiku_cost = CostTracker.calculate_cost(
            model="claude-3-5-haiku",
            input_tokens=tokens["input"],
            output_tokens=tokens["output"]
        )

        sonnet_cost = CostTracker.calculate_cost(
            model="claude-sonnet-4",
            input_tokens=tokens["input"],
            output_tokens=tokens["output"]
        )

        # Haiku should be significantly cheaper
        assert haiku_cost < sonnet_cost
        # Haiku is roughly 4x cheaper
        assert sonnet_cost / haiku_cost > 3

    def test_budget_tracking_scenario(self):
        """Test tracking multiple calls for budget management."""
        total_cost = 0.0

        # Simulate 10 API calls
        calls = [
            (1000, 500),
            (2000, 1000),
            (500, 200),
            (1500, 800),
            (3000, 1500),
            (800, 300),
            (1200, 600),
            (900, 450),
            (2500, 1200),
            (1800, 900),
        ]

        for input_tokens, output_tokens in calls:
            cost = CostTracker.calculate_cost(
                model="claude-sonnet-4",
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            total_cost += cost

        # Total should be calculable
        assert total_cost > 0
        assert isinstance(total_cost, float)

        # Format should work
        formatted = CostTracker.format_cost(total_cost)
        assert "$" in formatted
