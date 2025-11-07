"""
Simple integration test for HypothesisTesterAgent.
"""

import asyncio
from pathlib import Path

from src.kramer.hypothesis_tester_agent import HypothesisTesterAgent, TestResult
from src.world_model.graph import WorldModel, NodeType, EdgeType


async def test_hypothesis_tester():
    """Test the HypothesisTesterAgent with a simple hypothesis."""
    print("=== Testing HypothesisTesterAgent ===\n")

    # Create world model
    world_model = WorldModel()

    # Add some sample data
    print("1. Setting up world model with test data...")

    # Add a finding
    finding_id = world_model.add_finding(
        text="High correlation observed between variable A and variable B (r=0.85, p<0.001)",
        confidence=0.9,
        metadata={"source": "test_data_analysis"},
    )

    # Add a paper
    paper_id = world_model.add_paper(
        text="Previous research shows strong relationship between A and B in similar contexts",
        title="Study on A-B Relationship",
        authors=["Smith et al."],
        year=2023,
        metadata={"relevance": "high"},
    )

    # Add a hypothesis
    hypothesis_id = world_model.add_hypothesis(
        text="Variable A has a causal effect on variable B",
        confidence=0.7,
        metadata={
            "rationale": "Based on observed correlation and theoretical framework",
            "testability": "Can be tested using regression analysis and control variables",
            "novelty_score": 0.8,
        },
    )

    # Link hypothesis to finding
    world_model.add_edge(
        source=hypothesis_id,
        target=finding_id,
        edge_type=EdgeType.DERIVES_FROM,
    )

    print(f"   - Created finding: {finding_id}")
    print(f"   - Created paper: {paper_id}")
    print(f"   - Created hypothesis: {hypothesis_id}")
    print()

    # Test the hypothesis tester
    print("2. Creating HypothesisTesterAgent...")

    # Note: This will fail if ANTHROPIC_API_KEY is not set, which is expected
    try:
        tester = HypothesisTesterAgent(
            world_model=world_model,
            model="claude-sonnet-4-20250514",
            use_extended_thinking=False,  # Disable for faster testing
        )
        print("   ✓ Agent created successfully")
    except ValueError as e:
        print(f"   ✗ Agent creation failed: {e}")
        print("   Note: This is expected if ANTHROPIC_API_KEY is not set")
        return

    print()

    # Test hypothesis
    print("3. Testing hypothesis (literature-based only, no dataset)...")
    result = tester.test_hypothesis(
        hypothesis_id=hypothesis_id,
        dataset_path=None,  # No dataset, will only do literature test
        test_approaches=["literature"],
    )

    print(f"   - Test Type: {result.test_type}")
    print(f"   - Outcome: {result.outcome}")
    print(f"   - Confidence: {result.confidence:.2f}")
    print(f"   - Evidence Count: {len(result.evidence)}")
    print(f"   - Cost: ${result.cost:.4f}")
    print(f"   - Reasoning: {result.reasoning}")
    print()

    # Check world model was updated
    print("4. Verifying world model updates...")
    node_data = world_model.graph.nodes[hypothesis_id]
    metadata = node_data.get("metadata", {})

    if metadata.get("tested"):
        print("   ✓ Hypothesis marked as tested")
        print(f"   - Test outcome: {metadata.get('test_outcome')}")
        print(f"   - Test confidence: {metadata.get('test_confidence'):.2f}")
    else:
        print("   ✗ Hypothesis not marked as tested")

    print()
    print("=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_hypothesis_tester())
