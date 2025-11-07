"""
End-to-End Integration Tests for Full Discovery Loop.

Tests the complete discovery pipeline with real dataset.
"""

import pytest
import asyncio
import os
import tempfile
from pathlib import Path
import pandas as pd

from src.orchestrator.cycle_manager import Orchestrator
from src.world_model.graph import WorldModel, NodeType
from src.orchestrator.checkpoint_manager import CheckpointManager
from src.orchestrator.budget_enforcer import BudgetEnforcer
from src.utils.structured_logger import get_logger, EventType


pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - required for E2E tests"
)


class TestEndToEndIntegration:
    """End-to-end integration tests for full discovery loop."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a sample dataset for testing."""
        # Create synthetic dataset with correlations
        import numpy as np

        np.random.seed(42)
        n_samples = 200

        # Generate correlated features
        x1 = np.random.normal(0, 1, n_samples)
        x2 = x1 * 0.8 + np.random.normal(0, 0.2, n_samples)
        x3 = np.random.normal(0, 1, n_samples)
        y = 2 * x1 + 3 * x2 + np.random.normal(0, 0.5, n_samples)

        df = pd.DataFrame({
            'feature_1': x1,
            'feature_2': x2,
            'feature_3': x3,
            'target': y
        })

        dataset_path = tmp_path / "test_dataset.csv"
        df.to_csv(dataset_path, index=False)

        return str(dataset_path)

    @pytest.fixture
    async def world_model(self, tmp_path):
        """Create world model instance."""
        db_path = tmp_path / "world_model.db"
        world_model = WorldModel(db_path=str(db_path))
        yield world_model
        # Cleanup
        await world_model.close()

    @pytest.mark.asyncio
    async def test_single_cycle_discovery(self, world_model, sample_dataset):
        """Test a single discovery cycle end-to-end."""

        # Create orchestrator
        orchestrator = Orchestrator(
            world_model=world_model,
            max_concurrent_tasks=2,
            max_cycle_budget=5.0,
            max_total_budget=10.0
        )

        # Add dataset to world model
        await world_model.add_dataset(
            dataset_id="test_dataset",
            path=sample_dataset,
            description="Test dataset with correlated features"
        )

        # Run a single cycle
        objective = "Analyze the test dataset and discover relationships between features and target"

        cycle = await orchestrator.spawn_cycle(
            objective=objective,
            max_tasks=3
        )

        assert cycle is not None, "Cycle should be created"
        assert len(cycle.tasks) > 0, "Cycle should have tasks"

        # Check that findings were added to world model
        findings = world_model.query_nodes(NodeType.FINDING)
        assert len(findings) > 0, "Should have findings in world model"

        # Check budget tracking
        assert cycle.budget_used > 0, "Should have used budget"
        assert cycle.budget_used <= 5.0, "Should not exceed cycle budget"

        print(f"\n✓ Single cycle completed:")
        print(f"  Tasks: {len(cycle.tasks)}")
        print(f"  Findings: {len(findings)}")
        print(f"  Budget used: ${cycle.budget_used:.2f}")

    @pytest.mark.asyncio
    async def test_multi_cycle_discovery(self, world_model, sample_dataset):
        """Test multiple discovery cycles with hypothesis generation and testing."""

        orchestrator = Orchestrator(
            world_model=world_model,
            max_concurrent_tasks=2,
            max_cycle_budget=3.0,
            max_total_budget=10.0,
            auto_synthesize=True
        )

        # Add dataset
        await world_model.add_dataset(
            dataset_id="test_dataset",
            path=sample_dataset,
            description="Test dataset"
        )

        # Run multiple cycles
        results = await orchestrator.run_discovery_loop(
            objective="Discover and test hypotheses about feature relationships",
            max_cycles=3,
            max_time=300  # 5 minutes
        )

        assert results["cycles_completed"] >= 2, "Should complete multiple cycles"

        # Check world model state
        findings = world_model.query_nodes(NodeType.FINDING)
        hypotheses = world_model.query_nodes(NodeType.HYPOTHESIS)

        assert len(findings) > 0, "Should have findings"
        assert len(hypotheses) > 0, "Should have hypotheses"

        # Check for hypothesis testing
        tested_hypotheses = [
            h for h in hypotheses
            if h.get("metadata", {}).get("tested", False)
        ]

        print(f"\n✓ Multi-cycle discovery completed:")
        print(f"  Cycles: {results['cycles_completed']}")
        print(f"  Findings: {len(findings)}")
        print(f"  Hypotheses: {len(hypotheses)}")
        print(f"  Tested hypotheses: {len(tested_hypotheses)}")
        print(f"  Total cost: ${results['total_cost']:.2f}")

    @pytest.mark.asyncio
    async def test_discovery_with_checkpointing(self, world_model, sample_dataset, tmp_path):
        """Test discovery with checkpointing and recovery."""

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            auto_checkpoint=True,
            checkpoint_interval=1
        )

        orchestrator = Orchestrator(
            world_model=world_model,
            max_concurrent_tasks=2,
            max_cycle_budget=3.0,
            max_total_budget=10.0
        )

        # Add dataset
        await world_model.add_dataset(
            dataset_id="test_dataset",
            path=sample_dataset,
            description="Test dataset"
        )

        # Run first cycle
        cycle1 = await orchestrator.spawn_cycle(
            objective="Analyze dataset",
            max_tasks=2
        )

        # Create checkpoint
        checkpoint = checkpoint_manager.create_orchestrator_checkpoint(
            checkpoint_id="test_checkpoint_1",
            research_objective="Test objective",
            dataset_path=sample_dataset,
            total_cycles=1,
            total_budget_used=orchestrator.total_budget_used,
            discovery_complete=False,
            cycles=[],
            current_cycle_number=1,
            world_model_db_path=str(world_model.db_path),
            config={}
        )

        checkpoint_path = await checkpoint_manager.save_checkpoint(checkpoint)

        assert checkpoint_path.exists(), "Checkpoint should be saved"

        # Load checkpoint
        loaded_checkpoint = await checkpoint_manager.load_checkpoint(checkpoint_path)

        assert loaded_checkpoint is not None, "Checkpoint should load"
        assert loaded_checkpoint.total_cycles == 1, "Should restore cycle count"

        print(f"\n✓ Checkpointing test passed:")
        print(f"  Checkpoint saved: {checkpoint_path}")
        print(f"  Budget at checkpoint: ${loaded_checkpoint.total_budget_used:.2f}")

    @pytest.mark.asyncio
    async def test_discovery_with_budget_enforcement(self, world_model, sample_dataset):
        """Test that budget enforcement stops discovery when limit exceeded."""

        budget_enforcer = BudgetEnforcer(
            max_cycle_budget=1.0,  # Very small budget
            max_total_budget=2.0,
            enforce_hard_limits=True
        )

        orchestrator = Orchestrator(
            world_model=world_model,
            max_concurrent_tasks=2,
            max_cycle_budget=1.0,
            max_total_budget=2.0
        )

        # Add dataset
        await world_model.add_dataset(
            dataset_id="test_dataset",
            path=sample_dataset,
            description="Test dataset"
        )

        # Run discovery loop (should stop due to budget)
        results = await orchestrator.run_discovery_loop(
            objective="Analyze dataset",
            max_cycles=10,  # Request many cycles
            max_time=300
        )

        # Should stop early due to budget
        assert results["cycles_completed"] < 10, "Should stop before max_cycles due to budget"
        assert results["total_cost"] <= 2.5, "Should respect budget limit (with small tolerance)"

        print(f"\n✓ Budget enforcement test passed:")
        print(f"  Cycles completed: {results['cycles_completed']}")
        print(f"  Total cost: ${results['total_cost']:.2f}")
        print(f"  Budget limit: $2.00")

    @pytest.mark.asyncio
    async def test_discovery_with_structured_logging(self, world_model, sample_dataset, tmp_path):
        """Test discovery with structured logging."""

        log_dir = tmp_path / "logs"
        logger = get_logger("test_discovery", log_dir=str(log_dir))

        orchestrator = Orchestrator(
            world_model=world_model,
            max_concurrent_tasks=2,
            max_cycle_budget=3.0,
            max_total_budget=10.0
        )

        # Add dataset
        await world_model.add_dataset(
            dataset_id="test_dataset",
            path=sample_dataset,
            description="Test dataset"
        )

        # Log cycle start
        logger.log_cycle_start(cycle_number=1, objective="Test discovery")

        # Run cycle
        cycle = await orchestrator.spawn_cycle(
            objective="Analyze dataset",
            max_tasks=2
        )

        # Log cycle end
        logger.log_cycle_end(
            cycle_number=1,
            duration_ms=1000,
            budget_used=cycle.budget_used,
            tasks_completed=len(cycle.tasks)
        )

        # Check logs were created
        log_files = list(log_dir.glob("*.jsonl"))
        assert len(log_files) > 0, "Should create log files"

        # Export metrics
        metrics_path = logger.export_metrics()
        assert metrics_path.exists(), "Should export metrics"

        metrics = logger.get_metrics()
        assert metrics["total_cost"] > 0, "Should track costs"

        print(f"\n✓ Structured logging test passed:")
        print(f"  Log files: {len(log_files)}")
        print(f"  Metrics exported: {metrics_path}")

    @pytest.mark.asyncio
    async def test_full_discovery_pipeline(self, world_model, sample_dataset, tmp_path):
        """Test complete discovery pipeline with all components."""

        # Setup components
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            auto_checkpoint=True
        )

        budget_enforcer = BudgetEnforcer(
            max_cycle_budget=5.0,
            max_total_budget=15.0,
            enforce_hard_limits=True
        )

        logger = get_logger("full_pipeline", log_dir=str(tmp_path / "logs"))

        # Create orchestrator
        orchestrator = Orchestrator(
            world_model=world_model,
            max_concurrent_tasks=3,
            max_cycle_budget=5.0,
            max_total_budget=15.0,
            auto_synthesize=True,
            synthesis_interval=2
        )

        # Add dataset
        await world_model.add_dataset(
            dataset_id="test_dataset",
            path=sample_dataset,
            description="Test dataset with correlations"
        )

        # Run discovery
        logger.info(EventType.CYCLE_START, "Starting full pipeline test")

        results = await orchestrator.run_discovery_loop(
            objective="Discover, generate, and test hypotheses about feature relationships",
            max_cycles=3,
            max_time=600  # 10 minutes
        )

        logger.info(EventType.CYCLE_END, "Completed full pipeline test")

        # Verify results
        assert results["cycles_completed"] > 0, "Should complete cycles"

        findings = world_model.query_nodes(NodeType.FINDING)
        hypotheses = world_model.query_nodes(NodeType.HYPOTHESIS)

        assert len(findings) >= 2, "Should have multiple findings"
        assert len(hypotheses) >= 1, "Should generate hypotheses"

        # Check synthesis reports
        if results.get("synthesis_reports"):
            assert len(results["synthesis_reports"]) > 0, "Should generate synthesis reports"

        # Export final metrics
        metrics = logger.get_metrics()
        budget_report = budget_enforcer.get_budget_report()

        print(f"\n✓ Full pipeline test passed:")
        print(f"  Cycles: {results['cycles_completed']}")
        print(f"  Findings: {len(findings)}")
        print(f"  Hypotheses: {len(hypotheses)}")
        print(f"  Synthesis reports: {len(results.get('synthesis_reports', []))}")
        print(f"  Total cost: ${results['total_cost']:.2f}")
        print(f"  API calls: {metrics.get('total_api_calls', 0)}")

    @pytest.mark.asyncio
    async def test_discovery_with_hypothesis_ranking(self, world_model, sample_dataset):
        """Test discovery with hypothesis ranking."""

        from src.orchestrator.hypothesis_ranker import HypothesisRanker

        orchestrator = Orchestrator(
            world_model=world_model,
            max_concurrent_tasks=2,
            max_cycle_budget=5.0,
            max_total_budget=15.0
        )

        # Add dataset
        await world_model.add_dataset(
            dataset_id="test_dataset",
            path=sample_dataset,
            description="Test dataset"
        )

        # Run discovery to generate hypotheses
        results = await orchestrator.run_discovery_loop(
            objective="Generate and test hypotheses about features",
            max_cycles=2,
            max_time=300
        )

        # Get hypotheses
        hypotheses = world_model.query_nodes(NodeType.HYPOTHESIS)

        if len(hypotheses) > 0:
            # Rank hypotheses
            ranker = HypothesisRanker(
                world_model=world_model,
                research_objective="Analyze feature relationships"
            )

            ranked_hypotheses = ranker.rank_hypotheses(top_k=5)

            assert len(ranked_hypotheses) > 0, "Should rank hypotheses"
            assert all(h.rank is not None for h in ranked_hypotheses), "All should have ranks"

            # Check scores are valid
            for hyp in ranked_hypotheses:
                assert 0 <= hyp.composite_score <= 1, "Score should be 0-1"
                assert 0 <= hyp.information_gain <= 1, "Info gain should be 0-1"
                assert 0 <= hyp.novelty <= 1, "Novelty should be 0-1"

            print(f"\n✓ Hypothesis ranking test passed:")
            print(f"  Total hypotheses: {len(hypotheses)}")
            print(f"  Ranked: {len(ranked_hypotheses)}")
            if ranked_hypotheses:
                top = ranked_hypotheses[0]
                print(f"  Top hypothesis score: {top.composite_score:.3f}")
        else:
            print(f"\n⚠ No hypotheses generated - skipping ranking test")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
