"""
Persistence service for saving and loading discoveries from PostgreSQL.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import async_session_maker
from app.models.db_models import (
    Discovery, Cycle, Task, WorldModelNode, WorldModelEdge, CycleReport,
    DiscoveryStatus, CycleStatus, TaskStatus
)


class PersistenceService:
    """Service for persisting and loading discovery data from PostgreSQL."""

    async def save_discovery(
        self,
        discovery_id: str,
        config: dict,
        status: str = "pending",
        current_cycle: int = 0,
        total_cost: float = 0.0,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> Discovery:
        """Save or update a discovery record."""
        async with async_session_maker() as session:
            # Check if exists
            result = await session.execute(
                select(Discovery).where(Discovery.id == discovery_id)
            )
            discovery = result.scalar_one_or_none()

            if discovery:
                # Update existing
                discovery.objective = config.get("objective", discovery.objective)
                discovery.status = status
                discovery.current_cycle = current_cycle
                discovery.total_cost = total_cost
                if started_at:
                    discovery.started_at = started_at
                if completed_at:
                    discovery.completed_at = completed_at
            else:
                # Create new
                discovery = Discovery(
                    id=discovery_id,
                    objective=config.get("objective", ""),
                    dataset_path=config.get("dataset_path"),
                    model=config.get("model", "claude-sonnet-4-20250514"),
                    max_cycles=config.get("max_cycles", 20),
                    max_total_budget=config.get("max_total_budget", 100.0),
                    max_parallel_tasks=config.get("max_parallel_tasks", 4),
                    status=status,
                    current_cycle=current_cycle,
                    total_cost=total_cost,
                    started_at=started_at,
                    completed_at=completed_at,
                )
                session.add(discovery)

            await session.commit()
            await session.refresh(discovery)
            return discovery

    async def get_discovery(self, discovery_id: str) -> Optional[Discovery]:
        """Get a discovery by ID."""
        async with async_session_maker() as session:
            result = await session.execute(
                select(Discovery)
                .options(selectinload(Discovery.cycles).selectinload(Cycle.tasks))
                .where(Discovery.id == discovery_id)
            )
            return result.scalar_one_or_none()

    async def list_discoveries(self) -> List[Discovery]:
        """List all discoveries."""
        async with async_session_maker() as session:
            result = await session.execute(
                select(Discovery).order_by(Discovery.created_at.desc())
            )
            return list(result.scalars().all())

    async def delete_discovery(self, discovery_id: str) -> bool:
        """Delete a discovery and all related data."""
        async with async_session_maker() as session:
            result = await session.execute(
                delete(Discovery).where(Discovery.id == discovery_id)
            )
            await session.commit()
            return result.rowcount > 0

    async def save_cycle(
        self,
        discovery_id: str,
        cycle_id: str,
        cycle_number: int,
        objective: str,
        status: str = "pending",
        budget_used: float = 0.0,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> Cycle:
        """Save or update a cycle record."""
        async with async_session_maker() as session:
            result = await session.execute(
                select(Cycle).where(Cycle.id == cycle_id)
            )
            cycle = result.scalar_one_or_none()

            if cycle:
                cycle.status = status
                cycle.budget_used = budget_used
                if started_at:
                    cycle.started_at = started_at
                if completed_at:
                    cycle.completed_at = completed_at
            else:
                cycle = Cycle(
                    id=cycle_id,
                    discovery_id=discovery_id,
                    cycle_number=cycle_number,
                    objective=objective,
                    status=status,
                    budget_used=budget_used,
                    started_at=started_at,
                    completed_at=completed_at,
                )
                session.add(cycle)

            await session.commit()
            await session.refresh(cycle)
            return cycle

    async def get_cycles(self, discovery_id: str) -> List[Cycle]:
        """Get all cycles for a discovery."""
        async with async_session_maker() as session:
            result = await session.execute(
                select(Cycle)
                .options(selectinload(Cycle.tasks))
                .where(Cycle.discovery_id == discovery_id)
                .order_by(Cycle.cycle_number)
            )
            return list(result.scalars().all())

    async def save_cycle_report(
        self,
        discovery_id: str,
        cycle_id: str,
        summary: str,
        full_content: str,
        tasks_completed: int = 0,
        findings_count: int = 0,
        hypotheses_count: int = 0,
        papers_count: int = 0,
        budget_used: float = 0.0,
        generation_cost: float = 0.0,
    ) -> CycleReport:
        """Save or update a cycle report."""
        async with async_session_maker() as session:
            result = await session.execute(
                select(CycleReport).where(CycleReport.cycle_id == cycle_id)
            )
            report = result.scalar_one_or_none()

            if report:
                # Update existing
                report.summary = summary
                report.full_content = full_content
                report.tasks_completed = tasks_completed
                report.findings_count = findings_count
                report.hypotheses_count = hypotheses_count
                report.papers_count = papers_count
                report.budget_used = budget_used
                report.generation_cost = generation_cost
            else:
                # Create new
                report = CycleReport(
                    discovery_id=discovery_id,
                    cycle_id=cycle_id,
                    summary=summary,
                    full_content=full_content,
                    tasks_completed=tasks_completed,
                    findings_count=findings_count,
                    hypotheses_count=hypotheses_count,
                    papers_count=papers_count,
                    budget_used=budget_used,
                    generation_cost=generation_cost,
                )
                session.add(report)

            await session.commit()
            await session.refresh(report)
            return report

    async def save_cycle_with_report(
        self,
        discovery_id: str,
        cycle_id: str,
        cycle_number: int,
        objective: str,
        cycle_status: str,
        budget_used: float,
        started_at: Optional[datetime],
        completed_at: Optional[datetime],
        report_summary: str,
        report_full_content: str,
        tasks_completed: int = 0,
        findings_count: int = 0,
        hypotheses_count: int = 0,
        papers_count: int = 0,
        generation_cost: float = 0.0,
    ) -> tuple[Cycle, CycleReport]:
        """
        Save cycle and report in the same transaction to avoid FK violations.

        This ensures the cycle exists before the report references it.
        """
        async with async_session_maker() as session:
            # Check if cycle exists
            result = await session.execute(
                select(Cycle).where(Cycle.id == cycle_id)
            )
            cycle = result.scalar_one_or_none()

            if cycle:
                # Update existing cycle
                cycle.status = cycle_status
                cycle.budget_used = budget_used
                if started_at:
                    cycle.started_at = started_at
                if completed_at:
                    cycle.completed_at = completed_at
            else:
                # Create new cycle
                cycle = Cycle(
                    id=cycle_id,
                    discovery_id=discovery_id,
                    cycle_number=cycle_number,
                    objective=objective,
                    status=cycle_status,
                    budget_used=budget_used,
                    started_at=started_at,
                    completed_at=completed_at,
                )
                session.add(cycle)

            # Flush to ensure cycle has an ID before creating report
            await session.flush()

            # Check if report exists
            result = await session.execute(
                select(CycleReport).where(CycleReport.cycle_id == cycle_id)
            )
            report = result.scalar_one_or_none()

            if report:
                # Update existing report
                report.summary = report_summary
                report.full_content = report_full_content
                report.tasks_completed = tasks_completed
                report.findings_count = findings_count
                report.hypotheses_count = hypotheses_count
                report.papers_count = papers_count
                report.budget_used = budget_used
                report.generation_cost = generation_cost
            else:
                # Create new report
                report = CycleReport(
                    discovery_id=discovery_id,
                    cycle_id=cycle_id,
                    summary=report_summary,
                    full_content=report_full_content,
                    tasks_completed=tasks_completed,
                    findings_count=findings_count,
                    hypotheses_count=hypotheses_count,
                    papers_count=papers_count,
                    budget_used=budget_used,
                    generation_cost=generation_cost,
                )
                session.add(report)

            # Commit both in the same transaction
            await session.commit()
            await session.refresh(cycle)
            await session.refresh(report)
            return cycle, report

    async def get_cycle_reports(self, discovery_id: str) -> List[CycleReport]:
        """Get all cycle reports for a discovery."""
        async with async_session_maker() as session:
            result = await session.execute(
                select(CycleReport)
                .where(CycleReport.discovery_id == discovery_id)
                .order_by(CycleReport.created_at)
            )
            return list(result.scalars().all())

    async def get_cycle_report(self, cycle_id: str) -> Optional[CycleReport]:
        """Get a cycle report by cycle ID."""
        async with async_session_maker() as session:
            result = await session.execute(
                select(CycleReport).where(CycleReport.cycle_id == cycle_id)
            )
            return result.scalar_one_or_none()

    async def get_recent_cycle_summaries(
        self,
        discovery_id: str,
        limit: int = 3
    ) -> List[str]:
        """Get compact summaries from recent cycle reports for LLM context."""
        async with async_session_maker() as session:
            result = await session.execute(
                select(CycleReport)
                .where(CycleReport.discovery_id == discovery_id)
                .order_by(CycleReport.created_at.desc())
                .limit(limit)
            )
            reports = list(result.scalars().all())
            # Reverse to get chronological order (oldest first)
            return [r.summary for r in reversed(reports)]

    async def save_task(
        self,
        cycle_id: str,
        task_id: str,
        task_type: str,
        objective: str,
        context: Optional[dict] = None,
        status: str = "pending",
        result: Optional[dict] = None,
        error: Optional[str] = None,
        cost: float = 0.0,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> Task:
        """Save or update a task record."""
        async with async_session_maker() as session:
            db_result = await session.execute(
                select(Task).where(Task.id == task_id)
            )
            task = db_result.scalar_one_or_none()

            if task:
                task.status = status
                task.result = result
                task.error = error
                task.cost = cost
                if started_at:
                    task.started_at = started_at
                if completed_at:
                    task.completed_at = completed_at
            else:
                task = Task(
                    id=task_id,
                    cycle_id=cycle_id,
                    task_type=task_type,
                    objective=objective,
                    context=context or {},
                    status=status,
                    result=result,
                    error=error,
                    cost=cost,
                    started_at=started_at,
                    completed_at=completed_at,
                )
                session.add(task)

            await session.commit()
            await session.refresh(task)
            return task

    async def save_world_model_node(
        self,
        discovery_id: str,
        node_id: str,
        node_type: str,
        text: str,
        confidence: Optional[float] = None,
        provenance: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> WorldModelNode:
        """Save or update a world model node."""
        async with async_session_maker() as session:
            result = await session.execute(
                select(WorldModelNode).where(WorldModelNode.id == node_id)
            )
            node = result.scalar_one_or_none()

            if node:
                node.text = text
                node.confidence = confidence
                node.provenance = provenance
                node.extra_data = metadata or {}
                node.updated_at = datetime.utcnow()
            else:
                node = WorldModelNode(
                    id=node_id,
                    discovery_id=discovery_id,
                    node_type=node_type,
                    text=text,
                    confidence=confidence,
                    provenance=provenance,
                    extra_data=metadata or {},
                )
                session.add(node)

            await session.commit()
            await session.refresh(node)
            return node

    async def save_world_model_edge(
        self,
        discovery_id: str,
        edge_id: str,
        source_id: str,
        target_id: str,
        edge_type: str,
        metadata: Optional[dict] = None,
    ) -> WorldModelEdge:
        """Save or update a world model edge."""
        async with async_session_maker() as session:
            result = await session.execute(
                select(WorldModelEdge).where(WorldModelEdge.id == edge_id)
            )
            edge = result.scalar_one_or_none()

            if edge:
                edge.extra_data = metadata or {}
            else:
                edge = WorldModelEdge(
                    id=edge_id,
                    discovery_id=discovery_id,
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=edge_type,
                    extra_data=metadata or {},
                )
                session.add(edge)

            await session.commit()
            await session.refresh(edge)
            return edge

    async def get_world_model_nodes(self, discovery_id: str) -> List[WorldModelNode]:
        """Get all world model nodes for a discovery."""
        async with async_session_maker() as session:
            result = await session.execute(
                select(WorldModelNode)
                .where(WorldModelNode.discovery_id == discovery_id)
            )
            return list(result.scalars().all())

    async def get_world_model_edges(self, discovery_id: str) -> List[WorldModelEdge]:
        """Get all world model edges for a discovery."""
        async with async_session_maker() as session:
            result = await session.execute(
                select(WorldModelEdge)
                .where(WorldModelEdge.discovery_id == discovery_id)
            )
            return list(result.scalars().all())

    async def sync_world_model_to_db(
        self,
        discovery_id: str,
        world_model: Any,  # WorldModel from src.world_model.graph
    ) -> None:
        """Sync entire world model graph to database."""
        # Save all nodes
        for node_id in world_model.graph.nodes():
            node_data = world_model.graph.nodes[node_id]
            await self.save_world_model_node(
                discovery_id=discovery_id,
                node_id=node_id,
                node_type=node_data.get("node_type", "unknown"),
                text=node_data.get("text", ""),
                confidence=node_data.get("confidence"),
                provenance=node_data.get("provenance"),
                metadata=node_data.get("metadata", {}),
            )

        # Save all edges
        for source, target, edge_data in world_model.graph.edges(data=True):
            # Use UUID for edge ID instead of composite key
            edge_id = str(uuid.uuid4())
            await self.save_world_model_edge(
                discovery_id=discovery_id,
                edge_id=edge_id,
                source_id=source,
                target_id=target,
                edge_type=edge_data.get("edge_type", "unknown"),
                metadata=edge_data.get("metadata", {}),
            )

    async def load_world_model_from_db(
        self,
        discovery_id: str,
        world_model: Any,  # WorldModel from src.world_model.graph
    ) -> None:
        """Load world model graph from database."""
        # Load nodes
        nodes = await self.get_world_model_nodes(discovery_id)
        for node in nodes:
            world_model.graph.add_node(
                node.id,
                node_type=node.node_type,
                text=node.text,
                confidence=node.confidence,
                provenance=node.provenance,
                metadata=node.extra_data,
                created_at=node.created_at.isoformat(),
                updated_at=node.updated_at.isoformat(),
            )

        # Load edges
        edges = await self.get_world_model_edges(discovery_id)
        for edge in edges:
            if world_model.graph.has_node(edge.source_id) and world_model.graph.has_node(edge.target_id):
                world_model.graph.add_edge(
                    edge.source_id,
                    edge.target_id,
                    edge_type=edge.edge_type,
                    metadata=edge.extra_data,
                    created_at=edge.created_at.isoformat(),
                )


# Global instance
_persistence_service: Optional[PersistenceService] = None


def get_persistence_service() -> PersistenceService:
    """Get the global persistence service instance."""
    global _persistence_service
    if _persistence_service is None:
        _persistence_service = PersistenceService()
    return _persistence_service
