"""
Simple test script to verify budget tracking implementation.

This script tests:
1. CostTracker utility with mock API responses
2. Cost calculation accuracy
3. Model pricing lookup
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.cost_tracker import CostTracker, CostBreakdown


class MockUsage:
    """Mock Anthropic API usage object."""
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockResponse:
    """Mock Anthropic API response object."""
    def __init__(self, input_tokens: int, output_tokens: int):
        self.usage = MockUsage(input_tokens, output_tokens)


def test_cost_calculation():
    """Test basic cost calculation."""
    print("Testing CostTracker.calculate_cost()...")

    # Test Sonnet 4.5 pricing
    # Input: $3 per million tokens, Output: $15 per million tokens
    model = "claude-sonnet-4-20250514"
    input_tokens = 1000
    output_tokens = 500

    expected_cost = (1000 / 1_000_000 * 3.0) + (500 / 1_000_000 * 15.0)
    actual_cost = CostTracker.calculate_cost(model, input_tokens, output_tokens)

    print(f"  Model: {model}")
    print(f"  Input tokens: {input_tokens}, Output tokens: {output_tokens}")
    print(f"  Expected cost: ${expected_cost:.6f}")
    print(f"  Actual cost: ${actual_cost:.6f}")
    print(f"  ✓ Test passed!" if abs(expected_cost - actual_cost) < 0.000001 else "  ✗ Test failed!")
    print()

    return abs(expected_cost - actual_cost) < 0.000001


def test_model_pricing_lookup():
    """Test model name to pricing lookup."""
    print("Testing model pricing lookup...")

    test_models = [
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ]

    for model in test_models:
        try:
            cost = CostTracker.calculate_cost(model, 1000, 500)
            print(f"  ✓ {model}: ${cost:.6f}")
        except ValueError as e:
            print(f"  ✗ {model}: {e}")
            return False

    print()
    return True


def test_track_call():
    """Test tracking cost from API response."""
    print("Testing CostTracker.track_call()...")

    model = "claude-sonnet-4-20250514"
    mock_response = MockResponse(input_tokens=2000, output_tokens=1000)

    expected_cost = (2000 / 1_000_000 * 3.0) + (1000 / 1_000_000 * 15.0)
    actual_cost = CostTracker.track_call(model, mock_response)

    print(f"  Model: {model}")
    print(f"  Mock response: 2000 input, 1000 output tokens")
    print(f"  Expected cost: ${expected_cost:.6f}")
    print(f"  Actual cost: ${actual_cost:.6f}")
    print(f"  ✓ Test passed!" if abs(expected_cost - actual_cost) < 0.000001 else "  ✗ Test failed!")
    print()

    return abs(expected_cost - actual_cost) < 0.000001


def test_cost_breakdown():
    """Test detailed cost breakdown."""
    print("Testing CostTracker.calculate_cost_detailed()...")

    model = "claude-sonnet-4-20250514"
    input_tokens = 5000
    output_tokens = 2000

    breakdown = CostTracker.calculate_cost_detailed(model, input_tokens, output_tokens)

    print(f"  Model: {model}")
    print(f"  Input tokens: {breakdown.input_tokens}")
    print(f"  Output tokens: {breakdown.output_tokens}")
    print(f"  Input cost: ${breakdown.input_cost:.6f}")
    print(f"  Output cost: ${breakdown.output_cost:.6f}")
    print(f"  Total cost: ${breakdown.total_cost:.6f}")
    print(f"  ✓ Breakdown generated successfully!")
    print()

    return True


def test_cost_formatting():
    """Test cost formatting."""
    print("Testing CostTracker.format_cost()...")

    test_cases = [
        (0.00045, "$0.0004"),  # Very small cost (4 decimal places)
        (0.0123, "$0.012"),    # Small cost (3 decimal places)
        (0.567, "$0.567"),     # Medium cost (3 decimal places)
        (1.234, "$1.23"),      # Large cost (2 decimal places)
        (12.567, "$12.57"),    # Very large cost (2 decimal places)
    ]

    all_passed = True
    for cost, expected in test_cases:
        actual = CostTracker.format_cost(cost)
        passed = actual == expected
        status = "✓" if passed else "✗"
        print(f"  {status} ${cost:.6f} -> {actual} (expected: {expected})")
        all_passed = all_passed and passed

    print()
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("BUDGET TRACKING TESTS")
    print("="*60 + "\n")

    tests = [
        ("Cost Calculation", test_cost_calculation),
        ("Model Pricing Lookup", test_model_pricing_lookup),
        ("Track Call", test_track_call),
        ("Cost Breakdown", test_cost_breakdown),
        ("Cost Formatting", test_cost_formatting),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"  ✗ {test_name} failed with exception: {e}\n")
            results.append((test_name, False))

    # Print summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {test_name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    print("="*60 + "\n")

    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
