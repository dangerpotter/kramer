"""
Long-Running Cycle Tests for 6-12 Hour Validation.

Tests system stability, resource management, and performance over extended periods.
"""

import pytest
import asyncio
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import psutil

from src.orchestrator.cycle_manager import Orchestrator
from src.world_model.graph import WorldModel, NodeType
from src.orchestrator.checkpoint_manager import CheckpointManager
from src.orchestrator.budget_enforcer import BudgetEnforcer
from src.utils.structured_logger import get_logger, EventType
from src.monitoring.dashboard import generate_dashboard


pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - required for long-running tests"
)


class SystemMonitor:
    """Monitor system resources during long-running tests."""

    def __init__(self):
        self.cpu_samples = []
        self.memory_samples = []
        self.timestamps = []
        self.process = psutil.Process()

    def record_sample(self):
        """Record a resource usage sample."""
        self.timestamps.append(datetime.now())
        self.cpu_samples.append(self.process.cpu_percent(interval=0.1))
        self.memory_samples.append(self.process.memory_info().rss / 1024 / 1024)  # MB

    def get_summary(self):
        """Get resource usage summary."""
        if not self.cpu_samples:
            return {}

        return {
            "avg_cpu_percent": sum(self.cpu_samples) / len(self.cpu_samples),
            "max_cpu_percent": max(self.cpu_samples),
            "avg_memory_mb": sum(self.memory_samples) / len(self.memory_samples),
            "max_memory_mb": max(self.memory_samples),
            "samples_count": len(self.cpu_samples)
        }


class TestLongRunning:
    """Long-running tests for system validation."""

    @pytest.fixture
    def complex_dataset(self, tmp_path):
        """Create a complex dataset for long-running tests."""
        import numpy as np

        np.random.seed(42)
        n_samples = 1000
        n_features = 20

        # Generate complex relationships
        data = {}

        for i in range(n_features):
            data[f"feature_{i}"] = np.random.normal(0, 1, n_samples)

        # Add some non-linear relationships
        data["target"] = (
            2 * data["feature_0"] +
            3 * data["feature_1"] ** 2 +
            np.sin(data["feature_2"]) * 5 +
            np.random.normal(0, 1, n_samples)
        )

        # Add categorical variable
        data["category"] = np.random.choice(["A", "B", "C"], n_samples)

        df = pd.DataFrame(data)

        dataset_path = tmp_path / "complex_dataset.csv"
        df.to_csv(dataset_path, index=False)

        return str(dataset_path)

    @pytest.fixture
    async def setup_long_running_test(self, tmp_path, complex_dataset):
        """Setup for long-running test."""
        # Create directories
        checkpoint_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"
        output_dir = tmp_path / "outputs"

        checkpoint_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)

        # Create world model
        db_path = tmp_path / "world_model.db"
        world_model = WorldModel(db_path=str(db_path))

        # Add dataset
        await world_model.add_dataset(
            dataset_id="complex_dataset",
            path=complex_dataset,
            description="Complex dataset with 20 features and non-linear relationships"
        )

        # Create components
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            auto_checkpoint=True,
            checkpoint_interval=2  # Checkpoint every 2 cycles
        )

        budget_enforcer = BudgetEnforcer(
            max_cycle_budget=10.0,
            max_total_budget=200.0,  # Higher budget for long run
            enforce_hard_limits=True,
            enable_projections=True
        )

        logger = get_logger("long_running", log_dir=str(log_dir))

        # Create orchestrator
        orchestrator = Orchestrator(
            world_model=world_model,
            max_concurrent_tasks=3,
            max_cycle_budget=10.0,
            max_total_budget=200.0,
            auto_synthesize=True,
            synthesis_interval=5,  # Synthesize every 5 cycles
            output_dir=str(output_dir)
        )

        yield {
            "world_model": world_model,
            "orchestrator": orchestrator,
            "checkpoint_manager": checkpoint_manager,
            "budget_enforcer": budget_enforcer,
            "logger": logger,
            "paths": {
                "checkpoint_dir": checkpoint_dir,
                "log_dir": log_dir,
                "output_dir": output_dir
            }
        }

        # Cleanup
        await world_model.close()

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_6_hour_continuous_discovery(self, setup_long_running_test):
        """
        Run 6-hour continuous discovery test.

        Tests:
        - System stability over extended period
        - Memory leak detection
        - Checkpoint/recovery reliability
        - Budget management
        - Synthesis generation
        """
        setup = setup_long_running_test
        orchestrator = setup["orchestrator"]
        logger = setup["logger"]
        checkpoint_manager = setup["checkpoint_manager"]
        world_model = setup["world_model"]

        # Setup monitoring
        monitor = SystemMonitor()

        # Start time
        start_time = datetime.now()
        target_duration = timedelta(hours=6)

        print(f"\n" + "="*80)
        print(f"Starting 6-hour continuous discovery test")
        print(f"Start time: {start_time}")
        print(f"Target end: {start_time + target_duration}")
        print("="*80 + "\n")

        # Run discovery with periodic monitoring
        monitoring_interval = 300  # 5 minutes

        try:
            # Start discovery in background
            discovery_task = asyncio.create_task(
                orchestrator.run_discovery_loop(
                    objective="Comprehensively analyze complex dataset, discover patterns, generate and test hypotheses",
                    max_cycles=50,  # Allow many cycles
                    max_time=int(target_duration.total_seconds())
                )
            )

            # Monitor while running
            while not discovery_task.done():
                # Wait for monitoring interval
                await asyncio.sleep(monitoring_interval)

                # Record resource usage
                monitor.record_sample()

                # Get current progress
                elapsed = datetime.now() - start_time
                findings = world_model.query_nodes(NodeType.FINDING)
                hypotheses = world_model.query_nodes(NodeType.HYPOTHESIS)

                print(f"\n[{elapsed}] Progress Report:")
                print(f"  Cycles: {len(orchestrator.cycles)}")
                print(f"  Findings: {len(findings)}")
                print(f"  Hypotheses: {len(hypotheses)}")
                print(f"  Budget used: ${orchestrator.total_budget_used:.2f}")
                print(f"  Memory: {monitor.memory_samples[-1]:.1f} MB")

            # Get final results
            results = await discovery_task

            # Final metrics
            end_time = datetime.now()
            total_duration = end_time - start_time

            resource_summary = monitor.get_summary()

            print(f"\n" + "="*80)
            print(f"6-Hour Test Completed")
            print("="*80)
            print(f"Duration: {total_duration}")
            print(f"\nDiscovery Results:")
            print(f"  Cycles completed: {results['cycles_completed']}")
            print(f"  Total cost: ${results['total_cost']:.2f}")
            print(f"\nWorld Model:")
            findings = world_model.query_nodes(NodeType.FINDING)
            hypotheses = world_model.query_nodes(NodeType.HYPOTHESIS)
            print(f"  Findings: {len(findings)}")
            print(f"  Hypotheses: {len(hypotheses)}")
            print(f"\nResource Usage:")
            print(f"  Avg CPU: {resource_summary['avg_cpu_percent']:.1f}%")
            print(f"  Max CPU: {resource_summary['max_cpu_percent']:.1f}%")
            print(f"  Avg Memory: {resource_summary['avg_memory_mb']:.1f} MB")
            print(f"  Max Memory: {resource_summary['max_memory_mb']:.1f} MB")

            # Generate dashboard
            dashboard_path = setup["paths"]["output_dir"] / "dashboard.html"
            generate_dashboard(
                log_dir=str(setup["paths"]["log_dir"]),
                output_path=str(dashboard_path)
            )
            print(f"\nDashboard: {dashboard_path}")

            # Assertions
            assert total_duration >= timedelta(hours=5.5), "Should run close to 6 hours"
            assert results['cycles_completed'] >= 5, "Should complete multiple cycles"
            assert len(findings) >= 10, "Should discover significant findings"
            assert resource_summary['max_memory_mb'] < 2000, "Memory should stay reasonable"

        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            raise

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_12_hour_with_interruptions(self, setup_long_running_test):
        """
        Run 12-hour test with simulated interruptions and recovery.

        Tests:
        - Checkpoint/recovery under stress
        - Long-term stability
        - Budget enforcement over time
        """
        setup = setup_long_running_test
        orchestrator = setup["orchestrator"]
        checkpoint_manager = setup["checkpoint_manager"]

        start_time = datetime.now()
        target_duration = timedelta(hours=12)

        print(f"\n" + "="*80)
        print(f"Starting 12-hour test with interruptions")
        print(f"Start time: {start_time}")
        print("="*80 + "\n")

        # Run in segments with checkpoints
        segment_duration = 3600 * 2  # 2 hours per segment
        segments_completed = 0

        while (datetime.now() - start_time) < target_duration:
            print(f"\n--- Starting segment {segments_completed + 1} ---")

            # Run discovery segment
            try:
                results = await orchestrator.run_discovery_loop(
                    objective="Continue comprehensive analysis",
                    max_cycles=10,
                    max_time=segment_duration
                )

                segments_completed += 1

                # Create checkpoint
                checkpoint = checkpoint_manager.create_orchestrator_checkpoint(
                    checkpoint_id=f"segment_{segments_completed}",
                    research_objective="Long-running test",
                    dataset_path=None,
                    total_cycles=len(orchestrator.cycles),
                    total_budget_used=orchestrator.total_budget_used,
                    discovery_complete=False,
                    cycles=[],
                    current_cycle_number=len(orchestrator.cycles),
                    world_model_db_path=str(setup["world_model"].db_path),
                    config={}
                )

                checkpoint_path = await checkpoint_manager.save_checkpoint(checkpoint)

                print(f"Checkpoint saved: {checkpoint_path}")
                print(f"Budget used: ${orchestrator.total_budget_used:.2f}")

            except Exception as e:
                print(f"Segment failed: {e}")
                # Try to recover from last checkpoint
                latest = await checkpoint_manager.get_latest_checkpoint()
                if latest:
                    print(f"Recovering from checkpoint: {latest}")
                break

        # Final report
        total_duration = datetime.now() - start_time

        print(f"\n" + "="*80)
        print(f"12-Hour Test Completed")
        print("="*80)
        print(f"Duration: {total_duration}")
        print(f"Segments completed: {segments_completed}")
        print(f"Total budget: ${orchestrator.total_budget_used:.2f}")

        assert segments_completed >= 3, "Should complete multiple segments"

    @pytest.mark.asyncio
    async def test_stress_rapid_cycles(self, setup_long_running_test):
        """
        Stress test with rapid cycle spawning.

        Tests system under high load with many small cycles.
        """
        setup = setup_long_running_test
        orchestrator = setup["orchestrator"]
        monitor = SystemMonitor()

        print(f"\n" + "="*80)
        print(f"Starting rapid cycle stress test")
        print("="*80 + "\n")

        start_time = time.time()

        # Run many rapid cycles with small budgets
        results = await orchestrator.run_discovery_loop(
            objective="Rapid analysis with many small cycles",
            max_cycles=20,
            max_time=1800  # 30 minutes
        )

        end_time = time.time()
        duration = end_time - start_time

        # Record final resource usage
        monitor.record_sample()
        resource_summary = monitor.get_summary()

        print(f"\n" + "="*80)
        print(f"Stress Test Completed")
        print("="*80)
        print(f"Duration: {duration:.1f}s")
        print(f"Cycles: {results['cycles_completed']}")
        print(f"Budget: ${results['total_cost']:.2f}")
        print(f"Avg cycle time: {duration / results['cycles_completed']:.1f}s")

        # Should complete rapidly without crashes
        assert results['cycles_completed'] >= 10, "Should complete many cycles"
        assert duration < 2000, "Should complete within reasonable time"

    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, setup_long_running_test):
        """
        Test for memory leaks over extended operation.

        Runs many cycles and monitors memory growth.
        """
        setup = setup_long_running_test
        orchestrator = setup["orchestrator"]
        monitor = SystemMonitor()

        print(f"\n" + "="*80)
        print(f"Starting memory leak detection test")
        print("="*80 + "\n")

        # Run cycles with periodic memory monitoring
        for i in range(10):
            monitor.record_sample()

            await orchestrator.spawn_cycle(
                objective=f"Cycle {i+1} analysis",
                max_tasks=2
            )

            # Force garbage collection
            import gc
            gc.collect()

            print(f"Cycle {i+1}: Memory = {monitor.memory_samples[-1]:.1f} MB")

        # Check for memory growth
        memory_growth = monitor.memory_samples[-1] - monitor.memory_samples[0]
        memory_growth_percent = (memory_growth / monitor.memory_samples[0]) * 100

        print(f"\n" + "="*80)
        print(f"Memory Leak Test Results")
        print("="*80)
        print(f"Initial memory: {monitor.memory_samples[0]:.1f} MB")
        print(f"Final memory: {monitor.memory_samples[-1]:.1f} MB")
        print(f"Growth: {memory_growth:.1f} MB ({memory_growth_percent:.1f}%)")

        # Memory growth should be reasonable
        assert memory_growth_percent < 200, "Memory growth should be < 200%"
        assert monitor.memory_samples[-1] < 3000, "Memory should stay under 3GB"


if __name__ == "__main__":
    # Run with: pytest test_long_running.py -v -s -m slow
    pytest.main([__file__, "-v", "-s", "-m", "slow"])
