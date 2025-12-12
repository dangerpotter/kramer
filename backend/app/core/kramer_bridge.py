"""
Bridge between FastAPI and existing Kramer code.

This module provides a wrapper around the existing CycleManager and WorldModel
to make them accessible via the web API.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Callable, Dict, Optional
from datetime import datetime
import asyncio

# Add parent directory to path to import Kramer modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.orchestrator.cycle_manager import Orchestrator
from src.world_model.graph import WorldModel, NodeType, EdgeType
from app.core.events import Event, EventType, create_event
from app.services.persistence_service import get_persistence_service

logger = logging.getLogger(__name__)


class KramerBridge:
    """
    Bridge between FastAPI and existing Kramer components.

    This class manages the lifecycle of Orchestrator instances and provides
    methods to interact with them via the web API.
    """

    def __init__(self):
        """Initialize the bridge."""
        self.orchestrators: Dict[str, Orchestrator] = {}
        self.world_models: Dict[str, WorldModel] = {}
        self.event_callbacks: Dict[str, Callable] = {}
        self.discovery_configs: Dict[str, dict] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._persistence = get_persistence_service()
        self._initialized = False

    async def startup(self) -> None:
        """Initialize bridge and load existing discoveries from database."""
        if self._initialized:
            return

        logger.info("Loading existing discoveries from database...")
        try:
            discoveries = await self._persistence.list_discoveries()
            for discovery in discoveries:
                if discovery.status in ("running", "pending"):
                    # Don't auto-resume, just mark as stopped
                    await self._persistence.save_discovery(
                        discovery_id=discovery.id,
                        config={
                            "objective": discovery.objective,
                            "dataset_path": discovery.dataset_path,
                            "model": discovery.model,
                            "max_cycles": discovery.max_cycles,
                            "max_total_budget": discovery.max_total_budget,
                            "max_parallel_tasks": discovery.max_parallel_tasks,
                        },
                        status="stopped",
                        current_cycle=discovery.current_cycle,
                        total_cost=discovery.total_cost,
                    )
                logger.info(f"Found discovery {discovery.id}: {discovery.status}")
            logger.info(f"Loaded {len(discoveries)} discoveries from database")
        except Exception as e:
            logger.warning(f"Failed to load discoveries from database: {e}")

        self._initialized = True

    async def initialize_discovery(
        self,
        discovery_id: str,
        config: dict,
        event_callback: Optional[Callable[[Event], None]] = None,
    ) -> None:
        """
        Initialize a new discovery session.

        Args:
            discovery_id: Unique identifier for this discovery
            config: Configuration dictionary with keys:
                - objective: Research objective
                - dataset_path: Path to dataset
                - max_cycles: Maximum cycles to run
                - max_total_budget: Budget in USD
                - max_parallel_tasks: Max parallel tasks
            event_callback: Optional callback for events
        """
        # Store config
        self.discovery_configs[discovery_id] = config

        # Set model from config if specified, otherwise use env var
        if config.get("model"):
            os.environ["CLAUDE_MODEL"] = config["model"]
        elif not os.getenv("CLAUDE_MODEL"):
            raise ValueError("CLAUDE_MODEL must be set either in config or environment")

        # Store event callback
        if event_callback:
            self.event_callbacks[discovery_id] = event_callback

        # Create world model (in-memory, will sync to PostgreSQL)
        world_model = WorldModel()
        self.world_models[discovery_id] = world_model

        # Create orchestrator
        # Compute sensible default cycle budget from total budget and max cycles
        total_budget = config.get("max_total_budget", 100.0)
        max_cycles = config.get("max_cycles", 20)
        default_cycle_budget = total_budget / max(max_cycles, 1)

        orchestrator = Orchestrator(
            world_model=world_model,
            max_concurrent_tasks=config.get("max_parallel_tasks", 3),
            default_budget=total_budget,
            max_cycle_budget=config.get("max_cycle_budget", default_cycle_budget),
            max_total_budget=total_budget,
        )

        # Set discovery context for cycle report persistence
        orchestrator.set_discovery_context(
            discovery_id=discovery_id,
            persistence_service=self._persistence,
        )

        self.orchestrators[discovery_id] = orchestrator

        # Save to PostgreSQL
        await self._persistence.save_discovery(
            discovery_id=discovery_id,
            config=config,
            status="pending",
        )

        # Emit discovery started event
        if event_callback:
            event = create_event(
                EventType.DISCOVERY_STARTED,
                discovery_id,
                {"objective": config["objective"]},
            )
            await self._emit_event(discovery_id, event)

    async def run_discovery(self, discovery_id: str) -> dict:
        """
        Run discovery in background.

        Args:
            discovery_id: Discovery to run

        Returns:
            Final result dictionary
        """
        orchestrator = self.orchestrators.get(discovery_id)
        if not orchestrator:
            raise ValueError(f"Discovery {discovery_id} not found")

        try:
            # Update status to running
            config = self.discovery_configs.get(discovery_id, {})
            await self._persistence.save_discovery(
                discovery_id=discovery_id,
                config=config,
                status="running",
                started_at=datetime.utcnow(),
            )

            # Attach event hooks to Orchestrator
            self._attach_event_hooks(discovery_id, orchestrator)

            # Run the discovery with periodic persistence
            result = await self._run_with_persistence(
                discovery_id=discovery_id,
                orchestrator=orchestrator,
                config=config,
            )

            # Update status to completed
            await self._persistence.save_discovery(
                discovery_id=discovery_id,
                config=config,
                status="completed",
                current_cycle=len(orchestrator.cycles),
                total_cost=orchestrator.total_budget_used,
                completed_at=datetime.utcnow(),
            )

            # Final sync of world model
            await self._sync_world_model(discovery_id)

            # Emit completion event
            event = create_event(
                EventType.DISCOVERY_COMPLETED,
                discovery_id,
                {"result": result},
            )
            await self._emit_event(discovery_id, event)

            return result

        except Exception as e:
            # Update status to failed
            config = self.discovery_configs.get(discovery_id, {})
            await self._persistence.save_discovery(
                discovery_id=discovery_id,
                config=config,
                status="failed",
                current_cycle=len(orchestrator.cycles) if orchestrator else 0,
                total_cost=orchestrator.total_budget_used if orchestrator else 0,
            )

            # Emit failure event
            event = create_event(
                EventType.DISCOVERY_FAILED,
                discovery_id,
                {"error": str(e)},
            )
            await self._emit_event(discovery_id, event)
            raise

    async def _run_with_persistence(
        self,
        discovery_id: str,
        orchestrator: Orchestrator,
        config: dict,
    ) -> dict:
        """Run discovery with periodic persistence of state."""
        last_cycle_count = 0

        # Run the discovery
        result = await orchestrator.run_cycle(
            objective=config.get("objective", "Research objective"),
            max_cycles=config.get("max_cycles", 20),
            budget=config.get("max_total_budget", 100.0)
        )

        # Persist cycles and tasks after completion
        for i, cycle in enumerate(orchestrator.cycles.values()):
            await self._persistence.save_cycle(
                discovery_id=discovery_id,
                cycle_id=cycle.cycle_id,
                cycle_number=i + 1,
                objective=cycle.objective,
                status=cycle.status.value,
                budget_used=cycle.budget_used,
                started_at=cycle.started_at,
                completed_at=cycle.completed_at,
            )

            # Save tasks for this cycle
            for task in cycle.tasks:
                await self._persistence.save_task(
                    cycle_id=cycle.cycle_id,
                    task_id=task.task_id,
                    task_type=task.task_type.value,
                    objective=task.objective,
                    context=task.context,
                    status=task.status.value,
                    result=task.result,
                    error=task.error,
                    cost=task.cost if hasattr(task, 'cost') else 0.0,
                    started_at=task.started_at,
                    completed_at=task.completed_at,
                )

        return result

    async def _sync_world_model(self, discovery_id: str) -> None:
        """Sync world model to PostgreSQL."""
        world_model = self.world_models.get(discovery_id)
        if not world_model:
            return

        try:
            await self._persistence.sync_world_model_to_db(discovery_id, world_model)
            logger.info(f"Synced world model for discovery {discovery_id}")
        except Exception as e:
            logger.error(f"Failed to sync world model: {e}")

    def _attach_event_hooks(self, discovery_id: str, orchestrator: Orchestrator) -> None:
        """Attach event hooks to Orchestrator for real-time updates."""
        # Note: This would require modifying Orchestrator to support callbacks
        # For now, we'll emit events at key points
        pass

    async def _emit_event(self, discovery_id: str, event: Event) -> None:
        """Emit an event if callback is registered."""
        callback = self.event_callbacks.get(discovery_id)
        if callback:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)

    async def stop_discovery(self, discovery_id: str) -> None:
        """Stop a running discovery."""
        task = self._tasks.get(discovery_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Update status in database
        config = self.discovery_configs.get(discovery_id, {})
        orchestrator = self.orchestrators.get(discovery_id)
        await self._persistence.save_discovery(
            discovery_id=discovery_id,
            config=config,
            status="stopped",
            current_cycle=len(orchestrator.cycles) if orchestrator else 0,
            total_cost=orchestrator.total_budget_used if orchestrator else 0,
        )

        # Sync world model before stopping
        await self._sync_world_model(discovery_id)

        # Emit stopped event
        event = create_event(
            EventType.DISCOVERY_STOPPED,
            discovery_id,
            {},
        )
        await self._emit_event(discovery_id, event)

    async def load_discovery(self, discovery_id: str) -> bool:
        """
        Load a discovery from the database into memory.

        Args:
            discovery_id: Discovery ID to load

        Returns:
            True if loaded successfully, False if not found
        """
        discovery = await self._persistence.get_discovery(discovery_id)
        if not discovery:
            return False

        # Reconstruct config
        config = {
            "objective": discovery.objective,
            "dataset_path": discovery.dataset_path,
            "model": discovery.model,
            "max_cycles": discovery.max_cycles,
            "max_total_budget": discovery.max_total_budget,
            "max_parallel_tasks": discovery.max_parallel_tasks,
        }
        self.discovery_configs[discovery_id] = config

        # Set model
        if config.get("model"):
            os.environ["CLAUDE_MODEL"] = config["model"]

        # Create world model and load from database
        world_model = WorldModel()
        await self._persistence.load_world_model_from_db(discovery_id, world_model)
        self.world_models[discovery_id] = world_model

        # Create orchestrator
        total_budget = config.get("max_total_budget", 100.0)
        max_cycles = config.get("max_cycles", 20)
        default_cycle_budget = total_budget / max(max_cycles, 1)

        orchestrator = Orchestrator(
            world_model=world_model,
            max_concurrent_tasks=config.get("max_parallel_tasks", 3),
            default_budget=total_budget,
            max_cycle_budget=config.get("max_cycle_budget", default_cycle_budget),
            max_total_budget=total_budget,
        )

        # Set discovery context for cycle report persistence
        orchestrator.set_discovery_context(
            discovery_id=discovery_id,
            persistence_service=self._persistence,
        )

        # Restore budget used from database
        orchestrator.total_budget_used = discovery.total_cost

        self.orchestrators[discovery_id] = orchestrator

        logger.info(f"Loaded discovery {discovery_id} from database")
        return True

    def get_orchestrator(self, discovery_id: str) -> Optional[Orchestrator]:
        """Get Orchestrator instance for a discovery."""
        return self.orchestrators.get(discovery_id)

    def get_world_model(self, discovery_id: str) -> Optional[WorldModel]:
        """Get WorldModel instance for a discovery."""
        return self.world_models.get(discovery_id)

    def get_cycle_manager(self, discovery_id: str) -> Optional[Orchestrator]:
        """Get the orchestrator (cycle manager) for a discovery session."""
        return self.orchestrators.get(discovery_id)

    def get_discovery_status(self, discovery_id: str) -> dict:
        """
        Get current status of a discovery.

        Returns:
            Status dictionary with current metrics
        """
        orchestrator = self.orchestrators.get(discovery_id)
        if not orchestrator:
            return {"status": "not_found"}

        world_model = self.world_models.get(discovery_id)

        # Count nodes by type
        findings_count = 0
        hypotheses_count = 0
        papers_count = 0

        if world_model:
            for node_id in world_model.graph.nodes():
                node_data = world_model.graph.nodes[node_id]
                node_type = node_data.get("node_type")
                if node_type == NodeType.FINDING.value:
                    findings_count += 1
                elif node_type == NodeType.HYPOTHESIS.value:
                    hypotheses_count += 1
                elif node_type == NodeType.PAPER.value:
                    papers_count += 1

        return {
            "discovery_id": discovery_id,
            "status": "running" if discovery_id in self._tasks else "idle",
            "current_cycle": len(orchestrator.cycles),
            "total_cost": orchestrator.total_budget_used,
            "findings_count": findings_count,
            "hypotheses_count": hypotheses_count,
            "papers_count": papers_count,
        }

    def cleanup_discovery(self, discovery_id: str) -> None:
        """Clean up resources for a discovery."""
        self.orchestrators.pop(discovery_id, None)
        self.world_models.pop(discovery_id, None)
        self.event_callbacks.pop(discovery_id, None)
        self.discovery_configs.pop(discovery_id, None)
        self._tasks.pop(discovery_id, None)


# Global bridge instance
_bridge: Optional[KramerBridge] = None


def get_bridge() -> KramerBridge:
    """Get or create the global KramerBridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = KramerBridge()
    return _bridge
