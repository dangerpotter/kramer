"""
World Model usage example for Kramer.

This script demonstrates how to:
1. Create a world model
2. Add findings and hypotheses
3. Create and run discovery cycles
4. Save and load the world model
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator.cycle_manager import Orchestrator
from src.world_model.graph import EdgeType, NodeType, WorldModel


async def main():
    print("=" * 60)
    print("Kramer - World Model Usage Example")
    print("=" * 60)
    print()

    # ========================================
    # 1. Create a World Model
    # ========================================
    print("1. Creating World Model...")
    world_model = WorldModel()
    print(f"   Created: {world_model}")
    print()

    # ========================================
    # 2. Add Some Initial Findings
    # ========================================
    print("2. Adding initial findings...")

    finding1 = world_model.add_finding(
        text="Global average temperature has increased by 1.1°C since pre-industrial times",
        code_link="analysis/temperature_trends.py:142",
        confidence=0.98,
        metadata={"source": "climate_data_2023.csv"},
    )
    print(f"   Added finding: {finding1[:8]}...")

    finding2 = world_model.add_finding(
        text="CO2 concentration has risen from 280ppm to 420ppm",
        code_link="analysis/co2_analysis.py:89",
        confidence=0.97,
        metadata={"source": "atmospheric_data.csv"},
    )
    print(f"   Added finding: {finding2[:8]}...")

    finding3 = world_model.add_finding(
        text="Arctic sea ice extent has declined by 13% per decade",
        code_link="analysis/ice_trends.py:56",
        confidence=0.95,
        metadata={"source": "satellite_data.nc"},
    )
    print(f"   Added finding: {finding3[:8]}...")
    print()

    # ========================================
    # 3. Add Hypotheses Based on Findings
    # ========================================
    print("3. Generating hypotheses...")

    hypothesis1 = world_model.add_hypothesis(
        text="Rising CO2 is the primary driver of temperature increase",
        parent_finding=finding1,
        confidence=0.85,
        metadata={"mechanism": "greenhouse_effect"},
    )
    print(f"   Generated hypothesis: {hypothesis1[:8]}...")

    # Link hypothesis to supporting finding
    world_model.add_edge(finding2, hypothesis1, EdgeType.SUPPORTS)
    print("   Linked CO2 finding as support")

    hypothesis2 = world_model.add_hypothesis(
        text="Temperature increase causes accelerated ice melt",
        parent_finding=finding3,
        confidence=0.80,
    )
    print(f"   Generated hypothesis: {hypothesis2[:8]}...")
    world_model.add_edge(finding1, hypothesis2, EdgeType.SUPPORTS)
    print("   Linked temperature finding as support")
    print()

    # ========================================
    # 4. Add Research Questions
    # ========================================
    print("4. Adding research questions...")

    question1 = world_model.add_question(
        text="What are the feedback mechanisms amplifying warming?",
        metadata={"priority": "high"},
    )
    print(f"   Added question: {question1[:8]}...")

    question2 = world_model.add_question(
        text="How do ocean currents respond to temperature changes?",
        metadata={"priority": "medium"},
    )
    print(f"   Added question: {question2[:8]}...")
    print()

    # ========================================
    # 5. Query the World Model
    # ========================================
    print("5. Querying world model...")

    # Get relevant context about temperature
    subgraph = world_model.get_relevant_context(
        query="temperature",
        max_nodes=10,
    )
    print(f"   Found {subgraph.number_of_nodes()} nodes related to 'temperature'")
    print(f"   With {subgraph.number_of_edges()} edges")

    # Get world model statistics
    stats = world_model.get_stats()
    print(f"\n   World Model Statistics:")
    print(f"   - Total nodes: {stats['total_nodes']}")
    print(f"   - Total edges: {stats['total_edges']}")
    print(f"   - Node types: {stats['node_types']}")
    print(f"   - Edge types: {stats['edge_types']}")
    print()

    # ========================================
    # 6. Create an Orchestrator
    # ========================================
    print("6. Creating orchestrator...")
    orchestrator = Orchestrator(
        world_model=world_model,
        max_concurrent_tasks=3,
        default_budget=100.0,
    )
    print(f"   Created: {orchestrator}")
    print()

    # ========================================
    # 7. Spawn a Discovery Cycle
    # ========================================
    print("7. Spawning discovery cycle...")
    print("   (Note: This is a placeholder implementation)")
    print("   (Agents will be implemented in Phase 2)")

    cycle = await orchestrator.spawn_cycle(
        objective="Analyze climate change feedback mechanisms",
        max_tasks=5,
    )

    print(f"\n   Cycle completed: {cycle.cycle_id}")
    print(f"   Status: {cycle.status.value}")
    print(f"   Tasks created: {len(cycle.tasks)}")

    # Show task details
    for i, task in enumerate(cycle.tasks, 1):
        print(f"   - Task {i}: {task.task_type.value} ({task.status.value})")

    # Get cycle summary
    summary = orchestrator.get_cycle_summary(cycle.cycle_id)
    print(f"\n   Cycle Summary:")
    print(f"   - Total tasks: {summary['total_tasks']}")
    print(f"   - Task status breakdown: {summary['task_status']}")
    print()

    # ========================================
    # 8. Check Active Hypotheses
    # ========================================
    print("8. Checking active hypotheses...")
    active_hypotheses = orchestrator.get_active_hypotheses()
    print(f"   Found {len(active_hypotheses)} active hypotheses:")
    for hyp in active_hypotheses:
        conf = hyp.get('confidence')
        conf_str = f" (confidence: {conf:.2f})" if conf else ""
        print(f"   - {hyp['text'][:60]}...{conf_str}")
    print()

    # ========================================
    # 9. Save World Model to Database
    # ========================================
    print("9. Saving world model to database...")
    db_path = Path("outputs/world_model.db")
    db_path.parent.mkdir(exist_ok=True)

    world_model.save(db_path)
    print(f"   Saved to: {db_path}")
    print(f"   File size: {db_path.stat().st_size} bytes")
    print()

    # ========================================
    # 10. Load World Model from Database
    # ========================================
    print("10. Loading world model from database...")
    loaded_model = WorldModel.load(db_path)
    print(f"    Loaded: {loaded_model}")

    # Verify data was loaded correctly
    loaded_stats = loaded_model.get_stats()
    print(f"    Verified: {loaded_stats['total_nodes']} nodes, {loaded_stats['total_edges']} edges")
    print()

    # ========================================
    # Summary
    # ========================================
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("- Run examples/basic_usage.py for data analysis")
    print("- Run examples/advanced_usage.py for world model integration")
    print("- Run examples/literature_search.py for literature review")
    print()


if __name__ == "__main__":
    asyncio.run(main())
