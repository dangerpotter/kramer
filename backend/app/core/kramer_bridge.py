"""
Bridge between FastAPI and existing Kramer code.

This module provides a wrapper around the existing CycleManager and WorldModel
to make them accessible via the web API.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, Optional
import asyncio

# Add parent directory to path to import Kramer modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.orchestrator.cycle_manager import CycleManager
from src.world_model.graph import WorldModel, NodeType, EdgeType
from app.core.events import Event, EventType, create_event


class KramerBridge:
    """
    Bridge between FastAPI and existing Kramer components.

    This class manages the lifecycle of CycleManager instances and provides
    methods to interact with them via the web API.
    """

    def __init__(self):
        """Initialize the bridge."""
        self.cycle_managers: Dict[str, CycleManager] = {}
        self.world_models: Dict[str, WorldModel] = {}
        self.event_callbacks: Dict[str, Callable] = {}
        self.discovery_configs: Dict[str, dict] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

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

        # Store event callback
        if event_callback:
            self.event_callbacks[discovery_id] = event_callback

        # Create world model with persistent database
        db_path = Path(f"../data/discoveries/{discovery_id}.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        world_model = WorldModel(db_path=db_path)
        self.world_models[discovery_id] = world_model

        # Create cycle manager
        cycle_manager = CycleManager(
            objective=config["objective"],
            dataset_path=config.get("dataset_path"),
            max_cycles=config.get("max_cycles", 20),
            max_total_budget=config.get("max_total_budget", 100.0),
            world_model=world_model,
            enable_checkpointing=config.get("enable_checkpointing", True),
            checkpoint_interval=config.get("checkpoint_interval", 5),
        )
        self.cycle_managers[discovery_id] = cycle_manager

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
        cycle_manager = self.cycle_managers.get(discovery_id)
        if not cycle_manager:
            raise ValueError(f"Discovery {discovery_id} not found")

        try:
            # Attach event hooks to CycleManager
            self._attach_event_hooks(discovery_id, cycle_manager)

            # Run the discovery
            result = await cycle_manager.run()

            # Emit completion event
            event = create_event(
                EventType.DISCOVERY_COMPLETED,
                discovery_id,
                {"result": result},
            )
            await self._emit_event(discovery_id, event)

            return result

        except Exception as e:
            # Emit failure event
            event = create_event(
                EventType.DISCOVERY_FAILED,
                discovery_id,
                {"error": str(e)},
            )
            await self._emit_event(discovery_id, event)
            raise

    def _attach_event_hooks(self, discovery_id: str, cycle_manager: CycleManager) -> None:
        """Attach event hooks to CycleManager for real-time updates."""
        # Note: This would require modifying CycleManager to support callbacks
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

        # Emit stopped event
        event = create_event(
            EventType.DISCOVERY_STOPPED,
            discovery_id,
            {},
        )
        await self._emit_event(discovery_id, event)

    def get_cycle_manager(self, discovery_id: str) -> Optional[CycleManager]:
        """Get CycleManager instance for a discovery."""
        return self.cycle_managers.get(discovery_id)

    def get_world_model(self, discovery_id: str) -> Optional[WorldModel]:
        """Get WorldModel instance for a discovery."""
        return self.world_models.get(discovery_id)

    def get_discovery_status(self, discovery_id: str) -> dict:
        """
        Get current status of a discovery.

        Returns:
            Status dictionary with current metrics
        """
        cycle_manager = self.cycle_managers.get(discovery_id)
        if not cycle_manager:
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
            "current_cycle": len(cycle_manager.cycles),
            "total_cost": sum(c.budget_used for c in cycle_manager.cycles),
            "findings_count": findings_count,
            "hypotheses_count": hypotheses_count,
            "papers_count": papers_count,
        }

    def cleanup_discovery(self, discovery_id: str) -> None:
        """Clean up resources for a discovery."""
        self.cycle_managers.pop(discovery_id, None)
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
