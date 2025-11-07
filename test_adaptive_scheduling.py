#!/usr/bin/env python3
"""
Test script for adaptive task scheduling feature.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator.cycle_manager import Orchestrator
from src.world_model.graph import WorldModel, NodeType


def test_adaptive_scheduling():
    """Test the adaptive task scheduling implementation."""

    print("Testing Adaptive Task Scheduling Feature")
    print("=" * 60)

    # Create a world model
    world_model = WorldModel()

    # Add some test data
    print("\n1. Setting up test world model...")

    # Add a dataset
    dataset_id = world_model.add_node(
        node_type=NodeType.DATASET,
        text="Test dataset: Customer behavior data",
        confidence=1.0,
        metadata={"source": "test"},
    )
    print(f"   Added dataset: {dataset_id}")

    # Add a hypothesis with weak evidence
    hyp_id = world_model.add_node(
        node_type=NodeType.HYPOTHESIS,
        text="Customer satisfaction correlates with purchase frequency",
        confidence=0.5,
        metadata={"novelty": 0.6},
    )
    print(f"   Added hypothesis: {hyp_id}")

    # Add a novel finding
    finding_id = world_model.add_node(
        node_type=NodeType.FINDING,
        text="Strong positive correlation (r=0.87) between satisfaction and frequency",
        confidence=0.9,
        metadata={"novelty": 0.85},
    )
    print(f"   Added novel finding: {finding_id}")

    # Add a paper without follow-up
    paper_id = world_model.add_node(
        node_type=NodeType.PAPER,
        text="Research paper on customer behavior patterns in e-commerce",
        confidence=1.0,
        metadata={"authors": "Smith et al."},
    )
    print(f"   Added paper: {paper_id}")

    # Create orchestrator
    print("\n2. Creating orchestrator...")
    orchestrator = Orchestrator(
        world_model=world_model,
        max_concurrent_tasks=3,
        max_total_budget=100.0,
    )

    # Create a test cycle
    print("\n3. Creating test cycle...")
    cycle = orchestrator.create_cycle(
        objective="Analyze customer behavior patterns and satisfaction drivers",
        max_tasks=10,
    )
    print(f"   Created cycle: {cycle.cycle_id}")

    # Test _identify_knowledge_gaps
    print("\n4. Testing _identify_knowledge_gaps()...")
    gaps = orchestrator._identify_knowledge_gaps()
    print(f"   Unexplored data: {len(gaps['unexplored_data'])}")
    for gap in gaps['unexplored_data']:
        print(f"      - {gap.get('text', 'N/A')[:60]}")

    print(f"   Weak evidence: {len(gaps['weak_evidence'])}")
    for gap in gaps['weak_evidence']:
        print(f"      - {gap.get('text', 'N/A')[:60]} (confidence: {gap.get('confidence', 0):.2f})")

    print(f"   Novel findings: {gaps['novel_findings_count']}")

    # Test _score_task_priority
    print("\n5. Testing _score_task_priority()...")
    if gaps['unexplored_data']:
        gap = gaps['unexplored_data'][0]
        score = orchestrator._score_task_priority(gap, type="data_analysis")
        print(f"   Sample gap score: {score:.3f}")
        print(f"      Novelty: {gap.get('novelty', 0):.2f}")
        print(f"      Validation strength: {gap.get('validation_strength', 0):.2f}")
        print(f"      Objective alignment: {gap.get('objective_alignment', 0):.2f}")

    # Test _propose_next_tasks
    print("\n6. Testing _propose_next_tasks()...")
    tasks = orchestrator._propose_next_tasks(
        cycle_id=cycle.cycle_id,
        max_parallel_tasks=5
    )
    print(f"   Proposed {len(tasks)} tasks:")
    for i, task in enumerate(tasks, 1):
        print(f"      {i}. [{task.task_type.value}] {task.objective[:60]}")

    # Test task creation helpers
    print("\n7. Testing task creation helpers...")
    if gaps['unexplored_data']:
        analysis_task = orchestrator._create_analysis_task(
            gaps['unexplored_data'][0],
            cycle.cycle_id
        )
        print(f"   Analysis task: {analysis_task.objective[:60]}")

    if gaps['weak_evidence']:
        lit_task = orchestrator._create_literature_task(
            gaps['weak_evidence'][0],
            cycle.cycle_id
        )
        print(f"   Literature task: {lit_task.objective[:60]}")

    hyp_task = orchestrator._create_hypothesis_task(
        cycle.cycle_id,
        gaps.get('novel_findings', [])
    )
    print(f"   Hypothesis task: {hyp_task.objective[:60]}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("\nSummary:")
    print(f"  - Knowledge gaps identified: {len(gaps['unexplored_data']) + len(gaps['weak_evidence'])}")
    print(f"  - Tasks proposed: {len(tasks)}")
    print(f"  - Task types: {set(t.task_type.value for t in tasks)}")


if __name__ == "__main__":
    try:
        test_adaptive_scheduling()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
