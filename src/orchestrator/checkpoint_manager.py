"""
Checkpoint Manager for Error Recovery and Restart.

Saves and restores orchestrator state for long-running discovery cycles.
Enables graceful recovery from crashes, timeouts, and interruptions.
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


@dataclass
class TaskCheckpoint:
    """Checkpoint data for a single task."""
    task_id: str
    task_type: str
    status: str
    objective: str
    dependencies: List[str]
    result: Optional[Dict[str, Any]] = None
    cost: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    error: Optional[str] = None


@dataclass
class CycleCheckpoint:
    """Checkpoint data for a discovery cycle."""
    cycle_number: int
    status: str
    objective: str
    tasks: List[TaskCheckpoint]
    budget_used: float = 0.0
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    synthesis_generated: bool = False


@dataclass
class OrchestratorCheckpoint:
    """Complete checkpoint of orchestrator state."""
    checkpoint_id: str
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Orchestrator state
    research_objective: str = ""
    dataset_path: Optional[str] = None
    total_cycles: int = 0
    total_budget_used: float = 0.0
    discovery_complete: bool = False

    # Cycles
    cycles: List[CycleCheckpoint] = field(default_factory=list)
    current_cycle_number: int = 0

    # World model path (for separate persistence)
    world_model_db_path: Optional[str] = None

    # Configuration snapshot
    config: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())


class CheckpointManager:
    """
    Manages checkpoints for orchestrator state.

    Features:
    - Automatic checkpoint creation on cycle completion
    - Manual checkpoint creation
    - State restoration from checkpoint
    - Checkpoint versioning
    - Checkpoint cleanup (keep last N checkpoints)
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        auto_checkpoint: bool = True,
        max_checkpoints: int = 10,
        checkpoint_interval: int = 1  # Checkpoint every N cycles
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            auto_checkpoint: Whether to auto-checkpoint after cycles
            max_checkpoints: Maximum number of checkpoints to keep
            checkpoint_interval: Checkpoint every N cycles
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.auto_checkpoint = auto_checkpoint
        self.max_checkpoints = max_checkpoints
        self.checkpoint_interval = checkpoint_interval

        self._lock = asyncio.Lock()

        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")

    async def save_checkpoint(
        self,
        checkpoint: OrchestratorCheckpoint,
        checkpoint_name: Optional[str] = None
    ) -> Path:
        """
        Save orchestrator checkpoint to disk.

        Args:
            checkpoint: Checkpoint data
            checkpoint_name: Optional custom name (default: timestamp-based)

        Returns:
            Path to saved checkpoint file
        """
        async with self._lock:
            if checkpoint_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"checkpoint_{timestamp}.json"

            checkpoint_path = self.checkpoint_dir / checkpoint_name

            # Update metadata
            checkpoint.last_activity = datetime.now().isoformat()

            # Convert to dict and save
            checkpoint_dict = asdict(checkpoint)

            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_dict, f, indent=2)

            logger.info(f"Checkpoint saved: {checkpoint_path}")

            # Cleanup old checkpoints
            await self._cleanup_old_checkpoints()

            return checkpoint_path

    async def load_checkpoint(
        self,
        checkpoint_path: Optional[Path] = None
    ) -> Optional[OrchestratorCheckpoint]:
        """
        Load orchestrator checkpoint from disk.

        Args:
            checkpoint_path: Path to checkpoint file (default: latest)

        Returns:
            Loaded checkpoint or None if not found
        """
        async with self._lock:
            if checkpoint_path is None:
                checkpoint_path = await self.get_latest_checkpoint()

            if checkpoint_path is None or not checkpoint_path.exists():
                logger.warning("No checkpoint found to load")
                return None

            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint_dict = json.load(f)

                # Reconstruct checkpoint object
                checkpoint = self._dict_to_checkpoint(checkpoint_dict)

                logger.info(f"Checkpoint loaded: {checkpoint_path}")
                return checkpoint

            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
                return None

    async def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to the most recent checkpoint file."""
        checkpoint_files = sorted(
            self.checkpoint_dir.glob("checkpoint_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        return checkpoint_files[0] if checkpoint_files else None

    async def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint metadata (name, size, created_at)
        """
        checkpoints = []

        for checkpoint_file in sorted(
            self.checkpoint_dir.glob("checkpoint_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        ):
            stat = checkpoint_file.stat()
            checkpoints.append({
                "name": checkpoint_file.name,
                "path": str(checkpoint_file),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

        return checkpoints

    async def delete_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Delete a specific checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if deleted, False otherwise
        """
        async with self._lock:
            try:
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    logger.info(f"Checkpoint deleted: {checkpoint_path}")
                    return True
                return False
            except Exception as e:
                logger.error(f"Failed to delete checkpoint: {e}")
                return False

    async def should_checkpoint(self, cycle_number: int) -> bool:
        """
        Determine if checkpoint should be created for given cycle.

        Args:
            cycle_number: Current cycle number

        Returns:
            True if checkpoint should be created
        """
        if not self.auto_checkpoint:
            return False

        return cycle_number % self.checkpoint_interval == 0

    def create_task_checkpoint(
        self,
        task_id: str,
        task_type: str,
        status: str,
        objective: str,
        dependencies: List[str],
        result: Optional[Dict[str, Any]] = None,
        cost: float = 0.0,
        error: Optional[str] = None
    ) -> TaskCheckpoint:
        """Create a task checkpoint from task data."""
        return TaskCheckpoint(
            task_id=task_id,
            task_type=task_type,
            status=status,
            objective=objective,
            dependencies=dependencies,
            result=result,
            cost=cost,
            completed_at=datetime.now().isoformat() if status == "completed" else None,
            error=error
        )

    def create_cycle_checkpoint(
        self,
        cycle_number: int,
        status: str,
        objective: str,
        tasks: List[TaskCheckpoint],
        budget_used: float = 0.0,
        synthesis_generated: bool = False
    ) -> CycleCheckpoint:
        """Create a cycle checkpoint from cycle data."""
        return CycleCheckpoint(
            cycle_number=cycle_number,
            status=status,
            objective=objective,
            tasks=tasks,
            budget_used=budget_used,
            synthesis_generated=synthesis_generated,
            end_time=datetime.now().isoformat() if status == "completed" else None
        )

    def create_orchestrator_checkpoint(
        self,
        checkpoint_id: str,
        research_objective: str,
        dataset_path: Optional[str],
        total_cycles: int,
        total_budget_used: float,
        discovery_complete: bool,
        cycles: List[CycleCheckpoint],
        current_cycle_number: int,
        world_model_db_path: Optional[str],
        config: Dict[str, Any]
    ) -> OrchestratorCheckpoint:
        """Create complete orchestrator checkpoint."""
        return OrchestratorCheckpoint(
            checkpoint_id=checkpoint_id,
            research_objective=research_objective,
            dataset_path=dataset_path,
            total_cycles=total_cycles,
            total_budget_used=total_budget_used,
            discovery_complete=discovery_complete,
            cycles=cycles,
            current_cycle_number=current_cycle_number,
            world_model_db_path=world_model_db_path,
            config=config
        )

    async def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints limit."""
        checkpoint_files = sorted(
            self.checkpoint_dir.glob("checkpoint_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if len(checkpoint_files) > self.max_checkpoints:
            for checkpoint_file in checkpoint_files[self.max_checkpoints:]:
                try:
                    checkpoint_file.unlink()
                    logger.debug(f"Cleaned up old checkpoint: {checkpoint_file}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup checkpoint {checkpoint_file}: {e}")

    def _dict_to_checkpoint(self, checkpoint_dict: Dict[str, Any]) -> OrchestratorCheckpoint:
        """Convert dictionary to OrchestratorCheckpoint object."""

        # Convert cycles
        cycles = []
        for cycle_dict in checkpoint_dict.get("cycles", []):
            # Convert tasks
            tasks = [
                TaskCheckpoint(**task_dict)
                for task_dict in cycle_dict.get("tasks", [])
            ]
            cycle_dict["tasks"] = tasks
            cycles.append(CycleCheckpoint(**cycle_dict))

        checkpoint_dict["cycles"] = cycles

        return OrchestratorCheckpoint(**checkpoint_dict)


class CheckpointError(Exception):
    """Base exception for checkpoint operations."""
    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when checkpoint file is not found."""
    pass


class CheckpointCorruptedError(CheckpointError):
    """Raised when checkpoint file is corrupted."""
    pass


# Utility functions for integration with orchestrator

async def save_orchestrator_state(
    orchestrator,
    checkpoint_manager: CheckpointManager,
    checkpoint_name: Optional[str] = None
) -> Path:
    """
    Save complete orchestrator state to checkpoint.

    Args:
        orchestrator: Orchestrator instance
        checkpoint_manager: CheckpointManager instance
        checkpoint_name: Optional custom checkpoint name

    Returns:
        Path to saved checkpoint
    """
    # Convert orchestrator state to checkpoint
    cycle_checkpoints = []

    for cycle in orchestrator.cycles:
        # Convert tasks
        task_checkpoints = [
            checkpoint_manager.create_task_checkpoint(
                task_id=task.task_id,
                task_type=task.task_type,
                status=task.status,
                objective=task.objective,
                dependencies=task.dependencies,
                result=task.result,
                cost=task.cost,
                error=task.error
            )
            for task in cycle.tasks
        ]

        cycle_checkpoint = checkpoint_manager.create_cycle_checkpoint(
            cycle_number=cycle.cycle_number,
            status=cycle.status,
            objective=cycle.objective,
            tasks=task_checkpoints,
            budget_used=cycle.budget_used,
            synthesis_generated=cycle.synthesis_generated
        )
        cycle_checkpoints.append(cycle_checkpoint)

    # Create orchestrator checkpoint
    checkpoint = checkpoint_manager.create_orchestrator_checkpoint(
        checkpoint_id=f"orch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        research_objective=orchestrator.research_objective,
        dataset_path=orchestrator.dataset_path,
        total_cycles=len(orchestrator.cycles),
        total_budget_used=orchestrator.total_budget_used,
        discovery_complete=orchestrator.discovery_complete,
        cycles=cycle_checkpoints,
        current_cycle_number=orchestrator.current_cycle_number,
        world_model_db_path=orchestrator.world_model.db_path if orchestrator.world_model else None,
        config=orchestrator.config.__dict__ if hasattr(orchestrator, 'config') else {}
    )

    # Save checkpoint
    checkpoint_path = await checkpoint_manager.save_checkpoint(checkpoint, checkpoint_name)

    logger.info(f"Orchestrator state saved to checkpoint: {checkpoint_path}")

    return checkpoint_path


async def restore_orchestrator_state(
    orchestrator,
    checkpoint_manager: CheckpointManager,
    checkpoint_path: Optional[Path] = None
) -> bool:
    """
    Restore orchestrator state from checkpoint.

    Args:
        orchestrator: Orchestrator instance to restore into
        checkpoint_manager: CheckpointManager instance
        checkpoint_path: Optional specific checkpoint path (default: latest)

    Returns:
        True if restored successfully, False otherwise
    """
    # Load checkpoint
    checkpoint = await checkpoint_manager.load_checkpoint(checkpoint_path)

    if checkpoint is None:
        logger.warning("No checkpoint available to restore")
        return False

    try:
        # Restore orchestrator state
        orchestrator.research_objective = checkpoint.research_objective
        orchestrator.dataset_path = checkpoint.dataset_path
        orchestrator.total_budget_used = checkpoint.total_budget_used
        orchestrator.discovery_complete = checkpoint.discovery_complete
        orchestrator.current_cycle_number = checkpoint.current_cycle_number

        # Restore cycles (convert back to orchestrator's Cycle objects)
        # Note: This requires the orchestrator to have a method to reconstruct cycles
        # from checkpoint data. Implementation will vary based on orchestrator design.

        logger.info(f"Orchestrator state restored from checkpoint: {checkpoint.checkpoint_id}")
        logger.info(f"  - Total cycles: {checkpoint.total_cycles}")
        logger.info(f"  - Budget used: ${checkpoint.total_budget_used:.2f}")
        logger.info(f"  - Discovery complete: {checkpoint.discovery_complete}")

        return True

    except Exception as e:
        logger.error(f"Failed to restore orchestrator state: {e}", exc_info=True)
        return False
