"""
Discovery service for managing discovery sessions.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from app.core.kramer_bridge import get_bridge
from app.core.events import Event
from app.services.websocket_manager import get_connection_manager
from app.services.persistence_service import get_persistence_service
from app.models.discovery import (
    DiscoveryConfig,
    DiscoveryDetail,
    DiscoveryStatus,
    MetricsResponse,
    CycleInfo,
    TaskStatusInfo,
)


class DiscoveryService:
    """Service for managing discovery sessions."""

    def __init__(self):
        """Initialize discovery service."""
        self.bridge = get_bridge()
        self.ws_manager = get_connection_manager()
        self.persistence = get_persistence_service()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.discovery_metadata: Dict[str, dict] = {}

    async def create_discovery(self, config: DiscoveryConfig) -> str:
        """
        Create a new discovery session.

        Args:
            config: Discovery configuration

        Returns:
            Discovery ID
        """
        discovery_id = str(uuid4())

        # Store metadata
        self.discovery_metadata[discovery_id] = {
            "created_at": datetime.utcnow(),
            "status": DiscoveryStatus.PENDING,
            "config": config.model_dump(),
        }

        # Initialize via bridge
        await self.bridge.initialize_discovery(
            discovery_id=discovery_id,
            config=config.model_dump(),
            event_callback=self._handle_event,
        )

        return discovery_id

    async def start_discovery(self, discovery_id: str):
        """
        Start running a discovery in the background.

        Args:
            discovery_id: Discovery to start
        """
        if discovery_id in self.running_tasks:
            raise ValueError("Discovery is already running")

        # Update status
        self.discovery_metadata[discovery_id]["status"] = DiscoveryStatus.RUNNING
        self.discovery_metadata[discovery_id]["started_at"] = datetime.utcnow()

        # Create background task
        task = asyncio.create_task(self._run_discovery(discovery_id))
        self.running_tasks[discovery_id] = task

    async def _run_discovery(self, discovery_id: str):
        """Run discovery (internal). Supports tree search mode."""
        try:
            config = self.discovery_metadata[discovery_id].get("config", {})

            if config.get("use_tree_search"):
                result = await self._run_tree_search_discovery(discovery_id, config)
            else:
                result = await self.bridge.run_discovery(discovery_id)

            self.discovery_metadata[discovery_id]["status"] = DiscoveryStatus.COMPLETED
            self.discovery_metadata[discovery_id]["completed_at"] = datetime.utcnow()
            self.discovery_metadata[discovery_id]["result"] = result
        except asyncio.CancelledError:
            self.discovery_metadata[discovery_id]["status"] = DiscoveryStatus.STOPPED
        except Exception as e:
            self.discovery_metadata[discovery_id]["status"] = DiscoveryStatus.FAILED
            self.discovery_metadata[discovery_id]["error"] = str(e)
        finally:
            self.running_tasks.pop(discovery_id, None)

    async def _run_tree_search_discovery(self, discovery_id: str, config: dict) -> dict:
        """Run discovery using tree search over hypotheses."""
        from src.orchestrator.tree_search import TreeSearchConfig, TreeSearchOrchestrator

        orchestrator = self.bridge.get_orchestrator(discovery_id)
        if not orchestrator:
            raise ValueError(f"Orchestrator not found for discovery {discovery_id}")

        tree_config = TreeSearchConfig(**config.get("tree_search_config", {}))
        tree_orchestrator = TreeSearchOrchestrator(orchestrator, tree_config)
        return await tree_orchestrator.run(objective=config.get("objective", ""))

    async def stop_discovery(self, discovery_id: str):
        """
        Stop a running discovery.

        Args:
            discovery_id: Discovery to stop
        """
        task = self.running_tasks.get(discovery_id)
        if task and not task.done():
            await self.bridge.stop_discovery(discovery_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.discovery_metadata[discovery_id]["status"] = DiscoveryStatus.STOPPED

    async def get_status(self, discovery_id: str) -> dict:
        """
        Get current status of a discovery.

        Args:
            discovery_id: Discovery ID

        Returns:
            Status dictionary
        """
        # Try to load from database if not in memory
        if discovery_id not in self.discovery_metadata:
            loaded = await self.bridge.load_discovery(discovery_id)
            if loaded:
                # Get from database
                db_discovery = await self.persistence.get_discovery(discovery_id)
                if db_discovery:
                    self.discovery_metadata[discovery_id] = {
                        "created_at": db_discovery.created_at,
                        "status": db_discovery.status,
                        "config": {
                            "objective": db_discovery.objective,
                            "model": db_discovery.model,
                            "max_cycles": db_discovery.max_cycles,
                            "max_total_budget": db_discovery.max_total_budget,
                        },
                        "started_at": db_discovery.started_at,
                        "completed_at": db_discovery.completed_at,
                    }
            else:
                raise ValueError(f"Discovery {discovery_id} not found")

        metadata = self.discovery_metadata[discovery_id]
        bridge_status = self.bridge.get_discovery_status(discovery_id)

        return {
            **metadata,
            **bridge_status,
        }

    async def get_cycles(self, discovery_id: str) -> List[CycleInfo]:
        """
        Get all cycles for a discovery.

        Args:
            discovery_id: Discovery ID

        Returns:
            List of cycle information
        """
        cycle_manager = self.bridge.get_cycle_manager(discovery_id)
        if not cycle_manager:
            # Try to load from database
            await self.bridge.load_discovery(discovery_id)
            cycle_manager = self.bridge.get_cycle_manager(discovery_id)

        if not cycle_manager:
            # Return cycles from database directly
            db_cycles = await self.persistence.get_cycles(discovery_id)
            return [
                CycleInfo(
                    cycle_id=c.id,
                    cycle_number=c.cycle_number,
                    status=c.status,
                    tasks=[
                        TaskStatusInfo(
                            task_id=t.id,
                            task_type=t.task_type,
                            status=t.status,
                            objective=t.objective,
                            created_at=t.created_at,
                            started_at=t.started_at,
                            completed_at=t.completed_at,
                        )
                        for t in c.tasks
                    ],
                    budget_used=c.budget_used,
                    findings_generated=sum(
                        len(t.result.get("findings", [])) if t.result else 0
                        for t in c.tasks
                    ),
                    hypotheses_generated=sum(
                        len([f for f in t.result.get("findings", []) if f.get("type") == "hypothesis"])
                        if t.result else 0
                        for t in c.tasks
                    ),
                    created_at=c.created_at,
                    started_at=c.started_at,
                    completed_at=c.completed_at,
                )
                for c in db_cycles
            ]

        cycles = []
        for i, cycle in enumerate(cycle_manager.cycles.values()):
            tasks = [
                TaskStatusInfo(
                    task_id=task.task_id,
                    task_type=task.task_type.value,
                    status=task.status.value,
                    objective=task.objective,
                    created_at=task.created_at,
                    started_at=task.started_at,
                    completed_at=task.completed_at,
                )
                for task in cycle.tasks
            ]

            cycle_info = CycleInfo(
                cycle_id=cycle.cycle_id,
                cycle_number=i + 1,
                status=cycle.status.value,
                tasks=tasks,
                budget_used=cycle.budget_used,
                findings_generated=sum(
                    len(t.result.get("findings", [])) for t in cycle.tasks if t.result
                ),
                hypotheses_generated=sum(
                    len([f for f in t.result.get("findings", []) if f.get("type") == "hypothesis"])
                    for t in cycle.tasks if t.result
                ),
                created_at=cycle.created_at,
                started_at=cycle.started_at,
                completed_at=cycle.completed_at,
            )
            cycles.append(cycle_info)

        return cycles

    async def get_metrics(self, discovery_id: str) -> MetricsResponse:
        """
        Get real-time metrics for a discovery.

        Args:
            discovery_id: Discovery ID

        Returns:
            Metrics response
        """
        status = await self.get_status(discovery_id)
        cycles = await self.get_cycles(discovery_id)

        cost_per_cycle = [c.budget_used for c in cycles]
        findings_per_cycle = [c.findings_generated for c in cycles]
        hypotheses_per_cycle = [c.hypotheses_generated for c in cycles]

        # Calculate task statistics
        all_tasks = [task for cycle in cycles for task in cycle.tasks]
        tasks_completed = sum(1 for t in all_tasks if t.status == "completed")
        tasks_pending = sum(1 for t in all_tasks if t.status == "pending")
        tasks_running = sum(1 for t in all_tasks if t.status == "running")

        # Calculate average task duration
        completed_tasks = [
            t for t in all_tasks
            if t.completed_at and t.started_at
        ]
        if completed_tasks:
            durations = [
                (t.completed_at - t.started_at).total_seconds()
                for t in completed_tasks
            ]
            avg_task_duration = sum(durations) / len(durations)
        else:
            avg_task_duration = 0.0

        return MetricsResponse(
            discovery_id=discovery_id,
            current_cycle=status.get("current_cycle", 0),
            total_cost=status.get("total_cost", 0.0),
            cost_per_cycle=cost_per_cycle,
            findings_per_cycle=findings_per_cycle,
            hypotheses_per_cycle=hypotheses_per_cycle,
            tasks_completed=tasks_completed,
            tasks_pending=tasks_pending,
            tasks_running=tasks_running,
            avg_task_duration=avg_task_duration,
        )

    async def list_discoveries(self) -> List[dict]:
        """
        List all discoveries from database.

        Returns:
            List of discovery metadata
        """
        try:
            discoveries = await self.persistence.list_discoveries()
            return [
                {
                    "discovery_id": d.id,
                    "objective": d.objective,
                    "model": d.model,
                    "status": d.status,
                    "current_cycle": d.current_cycle,
                    "total_cost": d.total_cost,
                    "created_at": d.created_at,
                    "started_at": d.started_at,
                    "completed_at": d.completed_at,
                }
                for d in discoveries
            ]
        except Exception:
            # Fallback to in-memory if database fails
            return list(self.discovery_metadata.values())

    async def _handle_event(self, event: Event):
        """Handle events from the bridge and broadcast via WebSocket."""
        await self.ws_manager.broadcast_event(event)
