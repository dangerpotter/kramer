"""
Budget Enforcer - Hard budget limits and enforcement for discovery cycles.

Provides:
- Hard budget enforcement with automatic cycle termination
- Budget warnings at configurable thresholds
- Budget tracking and reporting
- Cost predictions and projections
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class BudgetStatus(Enum):
    """Budget status indicators."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EXCEEDED = "exceeded"


@dataclass
class BudgetLimit:
    """Represents a budget limit."""
    amount: float
    warning_threshold: float = 0.8  # Warning at 80%
    critical_threshold: float = 0.95  # Critical at 95%
    enforce_hard_limit: bool = True


@dataclass
class BudgetUsage:
    """Tracks budget usage."""
    used: float = 0.0
    limit: float = 0.0
    remaining: float = 0.0
    percentage: float = 0.0
    status: BudgetStatus = BudgetStatus.NORMAL
    projected_total: Optional[float] = None


class BudgetExceededException(Exception):
    """Raised when budget limit is exceeded."""

    def __init__(self, message: str, budget_type: str, used: float, limit: float):
        super().__init__(message)
        self.budget_type = budget_type
        self.used = used
        self.limit = limit


class BudgetEnforcer:
    """
    Enforces hard budget limits for discovery cycles.

    Features:
    - Per-cycle budget enforcement
    - Global budget enforcement
    - Per-task budget enforcement
    - Automatic warnings at thresholds
    - Budget projection and prediction
    """

    def __init__(
        self,
        max_cycle_budget: float = 10.0,
        max_total_budget: float = 100.0,
        max_task_budget: Optional[float] = None,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95,
        enforce_hard_limits: bool = True,
        enable_projections: bool = True
    ):
        """
        Initialize budget enforcer.

        Args:
            max_cycle_budget: Maximum budget per cycle ($)
            max_total_budget: Maximum total budget across all cycles ($)
            max_task_budget: Optional maximum budget per task ($)
            warning_threshold: Threshold for warning (0.0-1.0)
            critical_threshold: Threshold for critical warning (0.0-1.0)
            enforce_hard_limits: If True, raise exception when exceeded
            enable_projections: If True, calculate budget projections
        """
        self.cycle_limit = BudgetLimit(
            amount=max_cycle_budget,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            enforce_hard_limit=enforce_hard_limits
        )

        self.total_limit = BudgetLimit(
            amount=max_total_budget,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            enforce_hard_limit=enforce_hard_limits
        )

        self.task_limit = BudgetLimit(
            amount=max_task_budget,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            enforce_hard_limit=enforce_hard_limits
        ) if max_task_budget else None

        self.enable_projections = enable_projections

        # Tracking
        self.total_used: float = 0.0
        self.cycle_usage: Dict[str, float] = {}
        self.task_usage: Dict[str, float] = {}

        # History for projections
        self.cycle_costs: List[float] = []
        self.task_costs: List[float] = []

        logger.info(
            f"BudgetEnforcer initialized: "
            f"cycle=${max_cycle_budget}, total=${max_total_budget}, "
            f"enforce={enforce_hard_limits}"
        )

    def check_can_start_cycle(self, cycle_id: str) -> bool:
        """
        Check if a new cycle can be started within budget.

        Args:
            cycle_id: Cycle identifier

        Returns:
            True if cycle can be started

        Raises:
            BudgetExceededException: If budget exceeded and enforcement enabled
        """
        # Check total budget
        total_usage = self.get_total_usage()

        if total_usage.status == BudgetStatus.EXCEEDED:
            message = (
                f"Cannot start cycle {cycle_id}: "
                f"Total budget exceeded (${total_usage.used:.2f} / ${total_usage.limit:.2f})"
            )

            if self.total_limit.enforce_hard_limit:
                logger.error(message)
                raise BudgetExceededException(
                    message,
                    "total",
                    total_usage.used,
                    total_usage.limit
                )
            else:
                logger.warning(message)
                return False

        # Estimate if we have enough budget for average cycle
        if self.enable_projections and self.cycle_costs:
            avg_cycle_cost = sum(self.cycle_costs) / len(self.cycle_costs)
            if self.total_used + avg_cycle_cost > self.total_limit.amount:
                logger.warning(
                    f"Starting cycle {cycle_id} may exceed total budget "
                    f"(projected: ${self.total_used + avg_cycle_cost:.2f})"
                )

        # Initialize cycle tracking
        self.cycle_usage[cycle_id] = 0.0

        logger.info(f"Cycle {cycle_id} approved to start (budget available)")
        return True

    def check_can_start_task(
        self,
        task_id: str,
        cycle_id: str,
        estimated_cost: Optional[float] = None
    ) -> bool:
        """
        Check if a new task can be started within budget.

        Args:
            task_id: Task identifier
            cycle_id: Cycle identifier
            estimated_cost: Optional estimated cost for this task

        Returns:
            True if task can be started

        Raises:
            BudgetExceededException: If budget exceeded and enforcement enabled
        """
        # Check total budget
        total_usage = self.get_total_usage()
        if total_usage.status == BudgetStatus.EXCEEDED:
            message = (
                f"Cannot start task {task_id}: "
                f"Total budget exceeded (${total_usage.used:.2f} / ${total_usage.limit:.2f})"
            )

            if self.total_limit.enforce_hard_limit:
                logger.error(message)
                raise BudgetExceededException(
                    message,
                    "total",
                    total_usage.used,
                    total_usage.limit
                )
            else:
                logger.warning(message)
                return False

        # Check cycle budget
        cycle_usage = self.get_cycle_usage(cycle_id)
        if cycle_usage.status == BudgetStatus.EXCEEDED:
            message = (
                f"Cannot start task {task_id}: "
                f"Cycle {cycle_id} budget exceeded (${cycle_usage.used:.2f} / ${cycle_usage.limit:.2f})"
            )

            if self.cycle_limit.enforce_hard_limit:
                logger.error(message)
                raise BudgetExceededException(
                    message,
                    "cycle",
                    cycle_usage.used,
                    cycle_usage.limit
                )
            else:
                logger.warning(message)
                return False

        # Check if estimated cost would exceed limits
        if estimated_cost:
            if self.total_used + estimated_cost > self.total_limit.amount:
                logger.warning(
                    f"Task {task_id} estimated cost (${estimated_cost:.4f}) "
                    f"may exceed total budget"
                )

            cycle_used = self.cycle_usage.get(cycle_id, 0.0)
            if cycle_used + estimated_cost > self.cycle_limit.amount:
                logger.warning(
                    f"Task {task_id} estimated cost (${estimated_cost:.4f}) "
                    f"may exceed cycle budget"
                )

        # Initialize task tracking
        self.task_usage[task_id] = 0.0

        logger.debug(f"Task {task_id} approved to start")
        return True

    def record_cost(
        self,
        cost: float,
        cycle_id: Optional[str] = None,
        task_id: Optional[str] = None
    ):
        """
        Record a cost and update budgets.

        Args:
            cost: Cost to record ($)
            cycle_id: Optional cycle identifier
            task_id: Optional task identifier

        Raises:
            BudgetExceededException: If budget exceeded and enforcement enabled
        """
        # Update total
        self.total_used += cost

        # Update cycle
        if cycle_id:
            if cycle_id not in self.cycle_usage:
                self.cycle_usage[cycle_id] = 0.0
            self.cycle_usage[cycle_id] += cost

        # Update task
        if task_id:
            if task_id not in self.task_usage:
                self.task_usage[task_id] = 0.0
            self.task_usage[task_id] += cost

            # Check task limit
            if self.task_limit:
                task_usage = self.get_task_usage(task_id)
                if task_usage.status == BudgetStatus.EXCEEDED:
                    message = (
                        f"Task {task_id} exceeded budget "
                        f"(${task_usage.used:.4f} / ${task_usage.limit:.4f})"
                    )

                    if self.task_limit.enforce_hard_limit:
                        logger.error(message)
                        raise BudgetExceededException(
                            message,
                            "task",
                            task_usage.used,
                            task_usage.limit
                        )
                    else:
                        logger.warning(message)

        # Check limits
        self._check_and_warn()

        logger.debug(
            f"Recorded cost: ${cost:.4f} "
            f"(total: ${self.total_used:.2f}, cycle: {cycle_id}, task: {task_id})"
        )

    def complete_cycle(self, cycle_id: str):
        """
        Mark cycle as complete and update projections.

        Args:
            cycle_id: Cycle identifier
        """
        if cycle_id in self.cycle_usage:
            cycle_cost = self.cycle_usage[cycle_id]
            self.cycle_costs.append(cycle_cost)

            logger.info(
                f"Cycle {cycle_id} completed: ${cycle_cost:.2f} "
                f"(avg cycle cost: ${sum(self.cycle_costs) / len(self.cycle_costs):.2f})"
            )

    def complete_task(self, task_id: str):
        """
        Mark task as complete and update projections.

        Args:
            task_id: Task identifier
        """
        if task_id in self.task_usage:
            task_cost = self.task_usage[task_id]
            self.task_costs.append(task_cost)

            logger.debug(f"Task {task_id} completed: ${task_cost:.4f}")

    def get_total_usage(self) -> BudgetUsage:
        """Get current total budget usage."""
        return self._calculate_usage(
            self.total_used,
            self.total_limit
        )

    def get_cycle_usage(self, cycle_id: str) -> BudgetUsage:
        """Get budget usage for a specific cycle."""
        used = self.cycle_usage.get(cycle_id, 0.0)
        return self._calculate_usage(used, self.cycle_limit)

    def get_task_usage(self, task_id: str) -> BudgetUsage:
        """Get budget usage for a specific task."""
        if self.task_limit is None:
            return BudgetUsage()

        used = self.task_usage.get(task_id, 0.0)
        return self._calculate_usage(used, self.task_limit)

    def get_projected_total_cost(self, remaining_cycles: int) -> Optional[float]:
        """
        Project total cost based on historical averages.

        Args:
            remaining_cycles: Number of cycles remaining

        Returns:
            Projected total cost or None if insufficient data
        """
        if not self.enable_projections or not self.cycle_costs:
            return None

        avg_cycle_cost = sum(self.cycle_costs) / len(self.cycle_costs)
        projected = self.total_used + (avg_cycle_cost * remaining_cycles)

        return projected

    def get_budget_report(self) -> Dict[str, Any]:
        """
        Get comprehensive budget report.

        Returns:
            Dictionary with budget statistics
        """
        total_usage = self.get_total_usage()

        report = {
            "total": {
                "used": total_usage.used,
                "limit": total_usage.limit,
                "remaining": total_usage.remaining,
                "percentage": total_usage.percentage,
                "status": total_usage.status.value
            },
            "cycles": {
                "count": len(self.cycle_usage),
                "average_cost": (
                    sum(self.cycle_costs) / len(self.cycle_costs)
                    if self.cycle_costs else 0.0
                ),
                "max_cost": max(self.cycle_costs) if self.cycle_costs else 0.0,
                "min_cost": min(self.cycle_costs) if self.cycle_costs else 0.0,
            },
            "tasks": {
                "count": len(self.task_usage),
                "average_cost": (
                    sum(self.task_costs) / len(self.task_costs)
                    if self.task_costs else 0.0
                ),
            },
        }

        # Add projections
        if self.enable_projections and self.cycle_costs:
            cycles_possible = int(total_usage.remaining / report["cycles"]["average_cost"])
            report["projections"] = {
                "cycles_possible": cycles_possible,
                "projected_10_cycles": self.get_projected_total_cost(10),
            }

        return report

    def print_budget_report(self):
        """Print human-readable budget report."""
        report = self.get_budget_report()

        print("\n" + "="*60)
        print("BUDGET REPORT")
        print("="*60)

        total = report["total"]
        print(f"\nTotal Budget:")
        print(f"  Used:      ${total['used']:.2f}")
        print(f"  Limit:     ${total['limit']:.2f}")
        print(f"  Remaining: ${total['remaining']:.2f}")
        print(f"  Status:    {total['status'].upper()} ({total['percentage']:.1f}%)")

        cycles = report["cycles"]
        print(f"\nCycles ({cycles['count']}):")
        print(f"  Average:   ${cycles['average_cost']:.2f}")
        if cycles['max_cost'] > 0:
            print(f"  Max:       ${cycles['max_cost']:.2f}")
            print(f"  Min:       ${cycles['min_cost']:.2f}")

        if "projections" in report:
            proj = report["projections"]
            print(f"\nProjections:")
            print(f"  Cycles possible: {proj['cycles_possible']}")
            if proj['projected_10_cycles']:
                print(f"  10 cycles cost:  ${proj['projected_10_cycles']:.2f}")

        print("="*60 + "\n")

    def _calculate_usage(
        self,
        used: float,
        limit: BudgetLimit
    ) -> BudgetUsage:
        """Calculate budget usage statistics."""
        remaining = limit.amount - used
        percentage = (used / limit.amount * 100) if limit.amount > 0 else 0

        # Determine status
        if used >= limit.amount:
            status = BudgetStatus.EXCEEDED
        elif percentage >= limit.critical_threshold * 100:
            status = BudgetStatus.CRITICAL
        elif percentage >= limit.warning_threshold * 100:
            status = BudgetStatus.WARNING
        else:
            status = BudgetStatus.NORMAL

        return BudgetUsage(
            used=used,
            limit=limit.amount,
            remaining=remaining,
            percentage=percentage,
            status=status
        )

    def _check_and_warn(self):
        """Check budgets and emit warnings if needed."""
        total_usage = self.get_total_usage()

        if total_usage.status == BudgetStatus.EXCEEDED:
            logger.error(
                f"BUDGET EXCEEDED: Total budget exceeded "
                f"(${total_usage.used:.2f} / ${total_usage.limit:.2f})"
            )

            if self.total_limit.enforce_hard_limit:
                raise BudgetExceededException(
                    "Total budget exceeded",
                    "total",
                    total_usage.used,
                    total_usage.limit
                )

        elif total_usage.status == BudgetStatus.CRITICAL:
            logger.warning(
                f"BUDGET CRITICAL: {total_usage.percentage:.1f}% of total budget used "
                f"(${total_usage.used:.2f} / ${total_usage.limit:.2f})"
            )

        elif total_usage.status == BudgetStatus.WARNING:
            logger.warning(
                f"Budget warning: {total_usage.percentage:.1f}% of total budget used "
                f"(${total_usage.used:.2f} / ${total_usage.limit:.2f})"
            )
