"""
Tests for the BudgetEnforcer module.
"""

import pytest
from src.orchestrator.budget_enforcer import (
    BudgetEnforcer,
    BudgetStatus,
    BudgetLimit,
    BudgetUsage,
    BudgetExceededException,
)


class TestBudgetStatusEnum:
    """Test BudgetStatus enumeration."""

    def test_status_values(self):
        """Test all status values are defined."""
        assert BudgetStatus.NORMAL.value == "normal"
        assert BudgetStatus.WARNING.value == "warning"
        assert BudgetStatus.CRITICAL.value == "critical"
        assert BudgetStatus.EXCEEDED.value == "exceeded"


class TestBudgetLimitDataclass:
    """Test BudgetLimit dataclass."""

    def test_create_budget_limit_defaults(self):
        """Test creating budget limit with defaults."""
        limit = BudgetLimit(amount=100.0)

        assert limit.amount == 100.0
        assert limit.warning_threshold == 0.8
        assert limit.critical_threshold == 0.95
        assert limit.enforce_hard_limit is True

    def test_create_budget_limit_custom(self):
        """Test creating budget limit with custom values."""
        limit = BudgetLimit(
            amount=50.0,
            warning_threshold=0.7,
            critical_threshold=0.9,
            enforce_hard_limit=False
        )

        assert limit.amount == 50.0
        assert limit.warning_threshold == 0.7
        assert limit.critical_threshold == 0.9
        assert limit.enforce_hard_limit is False


class TestBudgetUsageDataclass:
    """Test BudgetUsage dataclass."""

    def test_create_budget_usage_defaults(self):
        """Test creating budget usage with defaults."""
        usage = BudgetUsage()

        assert usage.used == 0.0
        assert usage.limit == 0.0
        assert usage.remaining == 0.0
        assert usage.percentage == 0.0
        assert usage.status == BudgetStatus.NORMAL
        assert usage.projected_total is None

    def test_create_budget_usage_custom(self):
        """Test creating budget usage with custom values."""
        usage = BudgetUsage(
            used=50.0,
            limit=100.0,
            remaining=50.0,
            percentage=50.0,
            status=BudgetStatus.WARNING,
            projected_total=75.0
        )

        assert usage.used == 50.0
        assert usage.limit == 100.0
        assert usage.remaining == 50.0
        assert usage.percentage == 50.0
        assert usage.status == BudgetStatus.WARNING
        assert usage.projected_total == 75.0


class TestBudgetExceededException:
    """Test BudgetExceededException."""

    def test_exception_creation(self):
        """Test creating budget exception."""
        exc = BudgetExceededException(
            "Budget exceeded",
            budget_type="total",
            used=110.0,
            limit=100.0
        )

        assert str(exc) == "Budget exceeded"
        assert exc.budget_type == "total"
        assert exc.used == 110.0
        assert exc.limit == 100.0


class TestBudgetEnforcerBasics:
    """Test basic BudgetEnforcer functionality."""

    def test_create_enforcer_defaults(self):
        """Test creating enforcer with defaults."""
        enforcer = BudgetEnforcer()

        assert enforcer.cycle_limit.amount == 10.0
        assert enforcer.total_limit.amount == 100.0
        assert enforcer.task_limit is None
        assert enforcer.enable_projections is True
        assert enforcer.total_used == 0.0

    def test_create_enforcer_custom(self):
        """Test creating enforcer with custom values."""
        enforcer = BudgetEnforcer(
            max_cycle_budget=5.0,
            max_total_budget=50.0,
            max_task_budget=1.0,
            warning_threshold=0.7,
            critical_threshold=0.9,
            enforce_hard_limits=False,
            enable_projections=False
        )

        assert enforcer.cycle_limit.amount == 5.0
        assert enforcer.total_limit.amount == 50.0
        assert enforcer.task_limit is not None
        assert enforcer.task_limit.amount == 1.0
        assert enforcer.cycle_limit.warning_threshold == 0.7
        assert enforcer.cycle_limit.critical_threshold == 0.9
        assert enforcer.cycle_limit.enforce_hard_limit is False
        assert enforcer.enable_projections is False


class TestCycleManagement:
    """Test cycle budget management."""

    def test_check_can_start_cycle(self):
        """Test checking if cycle can start."""
        enforcer = BudgetEnforcer(max_total_budget=100.0)

        result = enforcer.check_can_start_cycle("cycle-1")

        assert result is True
        assert "cycle-1" in enforcer.cycle_usage
        assert enforcer.cycle_usage["cycle-1"] == 0.0

    def test_check_can_start_cycle_budget_exceeded(self):
        """Test cycle start blocked when budget exceeded."""
        enforcer = BudgetEnforcer(
            max_total_budget=10.0,
            enforce_hard_limits=True
        )

        # Use up all budget
        enforcer.total_used = 11.0

        with pytest.raises(BudgetExceededException):
            enforcer.check_can_start_cycle("cycle-1")

    def test_check_can_start_cycle_soft_limit(self):
        """Test cycle start with soft limit."""
        enforcer = BudgetEnforcer(
            max_total_budget=10.0,
            enforce_hard_limits=False
        )

        # Use up all budget
        enforcer.total_used = 11.0

        result = enforcer.check_can_start_cycle("cycle-1")

        assert result is False

    def test_complete_cycle(self):
        """Test completing a cycle."""
        enforcer = BudgetEnforcer()

        enforcer.check_can_start_cycle("cycle-1")
        enforcer.record_cost(5.0, cycle_id="cycle-1")
        enforcer.complete_cycle("cycle-1")

        assert len(enforcer.cycle_costs) == 1
        assert enforcer.cycle_costs[0] == 5.0


class TestTaskManagement:
    """Test task budget management."""

    def test_check_can_start_task(self):
        """Test checking if task can start."""
        enforcer = BudgetEnforcer()

        enforcer.check_can_start_cycle("cycle-1")
        result = enforcer.check_can_start_task("task-1", "cycle-1")

        assert result is True
        assert "task-1" in enforcer.task_usage
        assert enforcer.task_usage["task-1"] == 0.0

    def test_check_can_start_task_total_exceeded(self):
        """Test task blocked when total budget exceeded."""
        enforcer = BudgetEnforcer(
            max_total_budget=10.0,
            enforce_hard_limits=True
        )

        enforcer.total_used = 11.0

        with pytest.raises(BudgetExceededException):
            enforcer.check_can_start_task("task-1", "cycle-1")

    def test_check_can_start_task_cycle_exceeded(self):
        """Test task blocked when cycle budget exceeded."""
        enforcer = BudgetEnforcer(
            max_cycle_budget=5.0,
            enforce_hard_limits=True
        )

        enforcer.check_can_start_cycle("cycle-1")
        enforcer.cycle_usage["cycle-1"] = 6.0  # Exceed cycle budget

        with pytest.raises(BudgetExceededException):
            enforcer.check_can_start_task("task-1", "cycle-1")

    def test_check_can_start_task_with_estimate(self):
        """Test task start with cost estimate."""
        enforcer = BudgetEnforcer(
            max_total_budget=10.0,
            max_cycle_budget=5.0
        )

        enforcer.check_can_start_cycle("cycle-1")

        # Task should start (warnings logged but not blocked)
        result = enforcer.check_can_start_task(
            "task-1", "cycle-1",
            estimated_cost=8.0  # Would exceed budget
        )

        assert result is True  # Still starts, just warns

    def test_complete_task(self):
        """Test completing a task."""
        enforcer = BudgetEnforcer()

        enforcer.check_can_start_cycle("cycle-1")
        enforcer.check_can_start_task("task-1", "cycle-1")
        enforcer.record_cost(0.5, cycle_id="cycle-1", task_id="task-1")
        enforcer.complete_task("task-1")

        assert len(enforcer.task_costs) == 1
        assert enforcer.task_costs[0] == 0.5


class TestCostRecording:
    """Test cost recording functionality."""

    def test_record_cost_total_only(self):
        """Test recording cost to total only."""
        enforcer = BudgetEnforcer()

        enforcer.record_cost(5.0)

        assert enforcer.total_used == 5.0

    def test_record_cost_with_cycle(self):
        """Test recording cost with cycle."""
        enforcer = BudgetEnforcer()

        enforcer.check_can_start_cycle("cycle-1")
        enforcer.record_cost(5.0, cycle_id="cycle-1")

        assert enforcer.total_used == 5.0
        assert enforcer.cycle_usage["cycle-1"] == 5.0

    def test_record_cost_with_task(self):
        """Test recording cost with task."""
        enforcer = BudgetEnforcer()

        enforcer.check_can_start_cycle("cycle-1")
        enforcer.check_can_start_task("task-1", "cycle-1")
        enforcer.record_cost(0.5, cycle_id="cycle-1", task_id="task-1")

        assert enforcer.total_used == 0.5
        assert enforcer.cycle_usage["cycle-1"] == 0.5
        assert enforcer.task_usage["task-1"] == 0.5

    def test_record_cost_creates_missing_cycle(self):
        """Test that recording cost creates missing cycle entry."""
        enforcer = BudgetEnforcer()

        enforcer.record_cost(5.0, cycle_id="new-cycle")

        assert enforcer.cycle_usage["new-cycle"] == 5.0

    def test_record_cost_exceeds_total(self):
        """Test recording cost that exceeds total budget."""
        enforcer = BudgetEnforcer(
            max_total_budget=10.0,
            enforce_hard_limits=True
        )

        enforcer.total_used = 5.0

        with pytest.raises(BudgetExceededException):
            enforcer.record_cost(6.0)  # Exceeds 10.0 total

    def test_record_cost_exceeds_task(self):
        """Test recording cost that exceeds task budget."""
        enforcer = BudgetEnforcer(
            max_task_budget=1.0,
            enforce_hard_limits=True
        )

        enforcer.check_can_start_cycle("cycle-1")
        enforcer.check_can_start_task("task-1", "cycle-1")

        with pytest.raises(BudgetExceededException):
            enforcer.record_cost(1.5, cycle_id="cycle-1", task_id="task-1")


class TestBudgetUsageQueries:
    """Test budget usage query methods."""

    def test_get_total_usage(self):
        """Test getting total budget usage."""
        enforcer = BudgetEnforcer(max_total_budget=100.0)

        enforcer.total_used = 50.0

        usage = enforcer.get_total_usage()

        assert usage.used == 50.0
        assert usage.limit == 100.0
        assert usage.remaining == 50.0
        assert usage.percentage == 50.0
        assert usage.status == BudgetStatus.NORMAL

    def test_get_cycle_usage(self):
        """Test getting cycle budget usage."""
        enforcer = BudgetEnforcer(max_cycle_budget=10.0)

        enforcer.check_can_start_cycle("cycle-1")
        enforcer.record_cost(5.0, cycle_id="cycle-1")

        usage = enforcer.get_cycle_usage("cycle-1")

        assert usage.used == 5.0
        assert usage.limit == 10.0
        assert usage.remaining == 5.0
        assert usage.percentage == 50.0

    def test_get_cycle_usage_missing(self):
        """Test getting usage for missing cycle."""
        enforcer = BudgetEnforcer()

        usage = enforcer.get_cycle_usage("nonexistent")

        assert usage.used == 0.0

    def test_get_task_usage(self):
        """Test getting task budget usage."""
        enforcer = BudgetEnforcer(max_task_budget=1.0)

        enforcer.check_can_start_cycle("cycle-1")
        enforcer.check_can_start_task("task-1", "cycle-1")
        enforcer.record_cost(0.5, cycle_id="cycle-1", task_id="task-1")

        usage = enforcer.get_task_usage("task-1")

        assert usage.used == 0.5
        assert usage.limit == 1.0
        assert usage.remaining == 0.5
        assert usage.percentage == 50.0

    def test_get_task_usage_no_limit(self):
        """Test getting task usage when no task limit set."""
        enforcer = BudgetEnforcer(max_task_budget=None)

        usage = enforcer.get_task_usage("task-1")

        assert usage.used == 0.0
        assert usage.limit == 0.0


class TestBudgetStatusCalculation:
    """Test budget status calculation."""

    def test_status_normal(self):
        """Test normal status calculation."""
        enforcer = BudgetEnforcer(max_total_budget=100.0)

        enforcer.total_used = 50.0
        usage = enforcer.get_total_usage()

        assert usage.status == BudgetStatus.NORMAL

    def test_status_warning(self):
        """Test warning status calculation."""
        enforcer = BudgetEnforcer(
            max_total_budget=100.0,
            warning_threshold=0.8
        )

        enforcer.total_used = 85.0
        usage = enforcer.get_total_usage()

        assert usage.status == BudgetStatus.WARNING

    def test_status_critical(self):
        """Test critical status calculation."""
        enforcer = BudgetEnforcer(
            max_total_budget=100.0,
            critical_threshold=0.95
        )

        enforcer.total_used = 96.0
        usage = enforcer.get_total_usage()

        assert usage.status == BudgetStatus.CRITICAL

    def test_status_exceeded(self):
        """Test exceeded status calculation."""
        enforcer = BudgetEnforcer(
            max_total_budget=100.0,
            enforce_hard_limits=False  # Disable to avoid exception
        )

        enforcer.total_used = 110.0
        usage = enforcer.get_total_usage()

        assert usage.status == BudgetStatus.EXCEEDED


class TestProjections:
    """Test budget projections."""

    def test_get_projected_total_cost(self):
        """Test getting projected total cost."""
        enforcer = BudgetEnforcer(enable_projections=True)

        # Complete some cycles with costs
        enforcer.cycle_costs = [5.0, 6.0, 7.0]  # Average: 6.0
        enforcer.total_used = 18.0

        projected = enforcer.get_projected_total_cost(remaining_cycles=5)

        # Expected: 18 + (6 * 5) = 48
        assert projected == 48.0

    def test_get_projected_total_cost_no_data(self):
        """Test projection with no historical data."""
        enforcer = BudgetEnforcer(enable_projections=True)

        projected = enforcer.get_projected_total_cost(remaining_cycles=5)

        assert projected is None

    def test_get_projected_total_cost_disabled(self):
        """Test projection when disabled."""
        enforcer = BudgetEnforcer(enable_projections=False)

        enforcer.cycle_costs = [5.0, 6.0, 7.0]

        projected = enforcer.get_projected_total_cost(remaining_cycles=5)

        assert projected is None


class TestBudgetReport:
    """Test budget reporting."""

    def test_get_budget_report_empty(self):
        """Test getting report from empty enforcer."""
        enforcer = BudgetEnforcer(
            max_total_budget=100.0,
            max_cycle_budget=10.0
        )

        report = enforcer.get_budget_report()

        assert report["total"]["used"] == 0.0
        assert report["total"]["limit"] == 100.0
        assert report["total"]["remaining"] == 100.0
        assert report["total"]["percentage"] == 0.0
        assert report["total"]["status"] == "normal"

        assert report["cycles"]["count"] == 0
        assert report["cycles"]["average_cost"] == 0.0

        assert report["tasks"]["count"] == 0

    def test_get_budget_report_with_data(self):
        """Test getting report with data."""
        enforcer = BudgetEnforcer(
            max_total_budget=100.0,
            enable_projections=True
        )

        # Simulate some usage
        enforcer.check_can_start_cycle("cycle-1")
        enforcer.record_cost(5.0, cycle_id="cycle-1")
        enforcer.complete_cycle("cycle-1")

        enforcer.check_can_start_cycle("cycle-2")
        enforcer.record_cost(7.0, cycle_id="cycle-2")
        enforcer.complete_cycle("cycle-2")

        report = enforcer.get_budget_report()

        assert report["total"]["used"] == 12.0
        assert report["cycles"]["count"] == 2
        assert report["cycles"]["average_cost"] == 6.0
        assert report["cycles"]["max_cost"] == 7.0
        assert report["cycles"]["min_cost"] == 5.0

        # Check projections
        assert "projections" in report
        assert report["projections"]["cycles_possible"] > 0

    def test_print_budget_report(self, capsys):
        """Test printing budget report."""
        enforcer = BudgetEnforcer(
            max_total_budget=100.0,
            enable_projections=True
        )

        enforcer.check_can_start_cycle("cycle-1")
        enforcer.record_cost(5.0, cycle_id="cycle-1")
        enforcer.complete_cycle("cycle-1")

        enforcer.print_budget_report()

        captured = capsys.readouterr()
        assert "BUDGET REPORT" in captured.out
        assert "Total Budget:" in captured.out
        assert "Used:" in captured.out
        assert "Limit:" in captured.out


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_zero_budget_limit(self):
        """Test behavior with zero budget limit."""
        enforcer = BudgetEnforcer(max_total_budget=0.0)

        usage = enforcer.get_total_usage()

        assert usage.percentage == 0.0

    def test_multiple_cycles_parallel(self):
        """Test tracking multiple cycles."""
        enforcer = BudgetEnforcer()

        enforcer.check_can_start_cycle("cycle-1")
        enforcer.check_can_start_cycle("cycle-2")
        enforcer.check_can_start_cycle("cycle-3")

        enforcer.record_cost(5.0, cycle_id="cycle-1")
        enforcer.record_cost(3.0, cycle_id="cycle-2")
        enforcer.record_cost(2.0, cycle_id="cycle-3")

        assert enforcer.total_used == 10.0
        assert enforcer.cycle_usage["cycle-1"] == 5.0
        assert enforcer.cycle_usage["cycle-2"] == 3.0
        assert enforcer.cycle_usage["cycle-3"] == 2.0

    def test_incremental_cost_recording(self):
        """Test recording costs incrementally."""
        enforcer = BudgetEnforcer()

        enforcer.check_can_start_cycle("cycle-1")

        enforcer.record_cost(1.0, cycle_id="cycle-1")
        enforcer.record_cost(2.0, cycle_id="cycle-1")
        enforcer.record_cost(3.0, cycle_id="cycle-1")

        assert enforcer.cycle_usage["cycle-1"] == 6.0
        assert enforcer.total_used == 6.0

    def test_soft_limit_allows_overage(self):
        """Test that soft limits allow going over budget."""
        enforcer = BudgetEnforcer(
            max_total_budget=10.0,
            enforce_hard_limits=False
        )

        enforcer.total_used = 9.0

        # This should work (soft limit)
        enforcer.record_cost(5.0)

        assert enforcer.total_used == 14.0  # Over budget but allowed

    def test_cycle_projections_warning(self):
        """Test that cycle start warns about projections."""
        enforcer = BudgetEnforcer(
            max_total_budget=20.0,
            enable_projections=True
        )

        # Complete a cycle with high cost
        enforcer.cycle_costs = [15.0]
        enforcer.total_used = 15.0

        # Should still start (just warns)
        result = enforcer.check_can_start_cycle("cycle-2")

        assert result is True
