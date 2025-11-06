"""Priority Queue for task management"""

import heapq
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .world_model import Task, Priority


@dataclass(order=True)
class PrioritizedTask:
    """Wrapper for tasks with priority ordering"""
    priority: int = field(compare=True)
    timestamp: float = field(compare=True)
    task: Task = field(compare=False)

    def __init__(self, task: Task):
        # Negate priority for max-heap behavior (higher priority = smaller number)
        self.priority = -task.priority.value
        self.timestamp = task.created_at.timestamp()
        self.task = task


class TaskPriorityQueue:
    """Priority queue for managing research tasks"""

    def __init__(self):
        self._heap: List[PrioritizedTask] = []
        self._task_ids = set()  # Track IDs to prevent duplicates

    def push(self, task: Task) -> bool:
        """Add a task to the queue. Returns True if added, False if duplicate."""
        if task.id in self._task_ids:
            return False

        prioritized = PrioritizedTask(task)
        heapq.heappush(self._heap, prioritized)
        self._task_ids.add(task.id)
        return True

    def pop(self) -> Optional[Task]:
        """Remove and return the highest priority task"""
        if not self._heap:
            return None

        prioritized = heapq.heappop(self._heap)
        self._task_ids.discard(prioritized.task.id)
        return prioritized.task

    def peek(self) -> Optional[Task]:
        """View the highest priority task without removing it"""
        if not self._heap:
            return None
        return self._heap[0].task

    def pop_batch(self, n: int) -> List[Task]:
        """Remove and return up to n highest priority tasks"""
        tasks = []
        for _ in range(min(n, len(self._heap))):
            task = self.pop()
            if task:
                tasks.append(task)
        return tasks

    def size(self) -> int:
        """Return the number of tasks in the queue"""
        return len(self._heap)

    def is_empty(self) -> bool:
        """Check if the queue is empty"""
        return len(self._heap) == 0

    def clear(self):
        """Remove all tasks from the queue"""
        self._heap.clear()
        self._task_ids.clear()

    def get_tasks_by_priority(self, priority: Priority) -> List[Task]:
        """Get all tasks with a specific priority (without removing them)"""
        return [pt.task for pt in self._heap
                if pt.task.priority == priority]

    def get_stats(self) -> dict:
        """Get statistics about the queue"""
        by_priority = {
            Priority.HIGH: 0,
            Priority.MEDIUM: 0,
            Priority.LOW: 0
        }
        by_type = {}

        for pt in self._heap:
            task = pt.task
            by_priority[task.priority] += 1
            by_type[task.type] = by_type.get(task.type, 0) + 1

        return {
            "total": len(self._heap),
            "by_priority": {
                "high": by_priority[Priority.HIGH],
                "medium": by_priority[Priority.MEDIUM],
                "low": by_priority[Priority.LOW]
            },
            "by_type": by_type
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"TaskPriorityQueue(total={stats['total']}, high={stats['by_priority']['high']}, medium={stats['by_priority']['medium']}, low={stats['by_priority']['low']})"
