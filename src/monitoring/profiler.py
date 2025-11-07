"""
Performance Profiling for Bottleneck Identification and Optimization.

Provides:
- Function-level profiling
- Cost analysis per component
- Performance metrics aggregation
- Bottleneck identification
- Optimization recommendations
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from functools import wraps
import cProfile
import pstats
from io import StringIO

logger = logging.getLogger(__name__)


@dataclass
class ProfileEntry:
    """Single profiling entry."""
    component: str
    operation: str
    duration_ms: float
    cost: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentProfile:
    """Aggregate profile for a component."""
    component_name: str
    total_calls: int = 0
    total_duration_ms: float = 0.0
    total_cost: float = 0.0
    avg_duration_ms: float = 0.0
    avg_cost: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    bottleneck_score: float = 0.0


@dataclass
class BottleneckReport:
    """Report identifying performance bottlenecks."""
    bottlenecks: List[ComponentProfile]
    total_time_ms: float
    total_cost: float
    recommendations: List[str]


class PerformanceProfiler:
    """
    Performance profiler for identifying bottlenecks.

    Features:
    - Automatic timing of function calls
    - Cost tracking per component
    - Bottleneck identification
    - Performance recommendations
    """

    def __init__(
        self,
        enable_profiling: bool = True,
        enable_cprofile: bool = False
    ):
        """
        Initialize performance profiler.

        Args:
            enable_profiling: Enable profiling
            enable_cprofile: Enable Python cProfile for detailed analysis
        """
        self.enable_profiling = enable_profiling
        self.enable_cprofile = enable_cprofile

        # Storage
        self.entries: List[ProfileEntry] = []
        self.component_profiles: Dict[str, ComponentProfile] = {}

        # cProfile
        self.profiler = cProfile.Profile() if enable_cprofile else None

        logger.info(f"PerformanceProfiler initialized (profiling={'enabled' if enable_profiling else 'disabled'})")

    def profile(
        self,
        component: str,
        operation: str
    ):
        """
        Decorator for profiling functions.

        Args:
            component: Component name (e.g., "DataAnalysisAgent")
            operation: Operation name (e.g., "analyze_dataset")

        Usage:
            @profiler.profile("MyAgent", "my_operation")
            def my_function():
                pass
        """
        def decorator(func: Callable):
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    if not self.enable_profiling:
                        return await func(*args, **kwargs)

                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    finally:
                        duration_ms = (time.time() - start_time) * 1000
                        self.record(component, operation, duration_ms)

                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    if not self.enable_profiling:
                        return func(*args, **kwargs)

                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        return result
                    finally:
                        duration_ms = (time.time() - start_time) * 1000
                        self.record(component, operation, duration_ms)

                return sync_wrapper

        return decorator

    def record(
        self,
        component: str,
        operation: str,
        duration_ms: float,
        cost: float = 0.0,
        tokens_input: int = 0,
        tokens_output: int = 0,
        **metadata
    ):
        """
        Record a profiling entry.

        Args:
            component: Component name
            operation: Operation name
            duration_ms: Duration in milliseconds
            cost: Cost in dollars
            tokens_input: Input tokens
            tokens_output: Output tokens
            **metadata: Additional metadata
        """
        if not self.enable_profiling:
            return

        entry = ProfileEntry(
            component=component,
            operation=operation,
            duration_ms=duration_ms,
            cost=cost,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            metadata=metadata
        )

        self.entries.append(entry)

        # Update component profile
        self._update_component_profile(component, entry)

    def start_cprofile(self):
        """Start cProfile profiling."""
        if self.profiler:
            self.profiler.enable()

    def stop_cprofile(self):
        """Stop cProfile profiling."""
        if self.profiler:
            self.profiler.disable()

    def get_cprofile_stats(self, top_n: int = 20) -> str:
        """Get cProfile statistics."""
        if not self.profiler:
            return "cProfile not enabled"

        stream = StringIO()
        stats = pstats.Stats(self.profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(top_n)

        return stream.getvalue()

    def get_component_profile(self, component: str) -> Optional[ComponentProfile]:
        """Get profile for a specific component."""
        return self.component_profiles.get(component)

    def get_all_profiles(self) -> List[ComponentProfile]:
        """Get all component profiles."""
        return list(self.component_profiles.values())

    def identify_bottlenecks(
        self,
        threshold_ms: float = 1000.0,
        threshold_cost: float = 1.0,
        top_n: int = 5
    ) -> BottleneckReport:
        """
        Identify performance bottlenecks.

        Args:
            threshold_ms: Duration threshold for bottleneck (ms)
            threshold_cost: Cost threshold for bottleneck ($)
            top_n: Number of top bottlenecks to return

        Returns:
            BottleneckReport with identified bottlenecks
        """
        # Compute bottleneck scores
        for component in self.component_profiles.values():
            # Score based on total time and cost
            time_score = component.total_duration_ms / 1000.0  # Convert to seconds
            cost_score = component.total_cost * 10  # Weight cost heavily

            component.bottleneck_score = time_score + cost_score

        # Sort by bottleneck score
        sorted_components = sorted(
            self.component_profiles.values(),
            key=lambda c: c.bottleneck_score,
            reverse=True
        )

        # Identify bottlenecks
        bottlenecks = []
        for component in sorted_components[:top_n]:
            if (component.avg_duration_ms >= threshold_ms or
                component.avg_cost >= threshold_cost):
                bottlenecks.append(component)

        # Generate recommendations
        recommendations = self._generate_recommendations(bottlenecks)

        # Compute totals
        total_time_ms = sum(c.total_duration_ms for c in self.component_profiles.values())
        total_cost = sum(c.total_cost for c in self.component_profiles.values())

        return BottleneckReport(
            bottlenecks=bottlenecks,
            total_time_ms=total_time_ms,
            total_cost=total_cost,
            recommendations=recommendations
        )

    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by component."""
        return {
            component: profile.total_cost
            for component, profile in self.component_profiles.items()
        }

    def get_time_breakdown(self) -> Dict[str, float]:
        """Get time breakdown by component."""
        return {
            component: profile.total_duration_ms
            for component, profile in self.component_profiles.items()
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        total_calls = sum(c.total_calls for c in self.component_profiles.values())
        total_time_ms = sum(c.total_duration_ms for c in self.component_profiles.values())
        total_cost = sum(c.total_cost for c in self.component_profiles.values())

        return {
            "total_calls": total_calls,
            "total_time_ms": total_time_ms,
            "total_time_s": total_time_ms / 1000.0,
            "total_cost": total_cost,
            "components": len(self.component_profiles),
            "entries": len(self.entries)
        }

    def print_report(self, top_n: int = 10):
        """Print human-readable profiling report."""
        summary = self.get_summary()

        print("\n" + "="*80)
        print("PERFORMANCE PROFILING REPORT")
        print("="*80)

        print(f"\nSummary:")
        print(f"  Total calls: {summary['total_calls']:,}")
        print(f"  Total time: {summary['total_time_s']:.2f}s")
        print(f"  Total cost: ${summary['total_cost']:.2f}")
        print(f"  Components: {summary['components']}")

        print(f"\nTop {top_n} Components by Time:")
        print("-"*80)

        time_sorted = sorted(
            self.component_profiles.values(),
            key=lambda c: c.total_duration_ms,
            reverse=True
        )

        for i, component in enumerate(time_sorted[:top_n], 1):
            print(f"{i}. {component.component_name}")
            print(f"   Calls: {component.total_calls:,}")
            print(f"   Total time: {component.total_duration_ms/1000:.2f}s")
            print(f"   Avg time: {component.avg_duration_ms:.1f}ms")
            print(f"   Cost: ${component.total_cost:.2f}")
            print()

        # Bottleneck analysis
        bottlenecks = self.identify_bottlenecks(top_n=5)

        if bottlenecks.bottlenecks:
            print(f"\nBottlenecks Identified:")
            print("-"*80)

            for i, bottleneck in enumerate(bottlenecks.bottlenecks, 1):
                print(f"{i}. {bottleneck.component_name}")
                print(f"   Score: {bottleneck.bottleneck_score:.2f}")
                print(f"   Avg duration: {bottleneck.avg_duration_ms:.1f}ms")
                print(f"   Total cost: ${bottleneck.total_cost:.2f}")
                print()

        if bottlenecks.recommendations:
            print(f"\nOptimization Recommendations:")
            print("-"*80)
            for i, rec in enumerate(bottlenecks.recommendations, 1):
                print(f"{i}. {rec}")

        print("="*80 + "\n")

    def reset(self):
        """Reset profiling data."""
        self.entries.clear()
        self.component_profiles.clear()

        if self.profiler:
            self.profiler = cProfile.Profile()

        logger.info("Profiling data reset")

    def _update_component_profile(self, component: str, entry: ProfileEntry):
        """Update aggregate profile for a component."""
        if component not in self.component_profiles:
            self.component_profiles[component] = ComponentProfile(
                component_name=component
            )

        profile = self.component_profiles[component]

        profile.total_calls += 1
        profile.total_duration_ms += entry.duration_ms
        profile.total_cost += entry.cost

        profile.avg_duration_ms = profile.total_duration_ms / profile.total_calls
        profile.avg_cost = profile.total_cost / profile.total_calls

        profile.min_duration_ms = min(profile.min_duration_ms, entry.duration_ms)
        profile.max_duration_ms = max(profile.max_duration_ms, entry.duration_ms)

    def _generate_recommendations(self, bottlenecks: List[ComponentProfile]) -> List[str]:
        """Generate optimization recommendations based on bottlenecks."""
        recommendations = []

        for bottleneck in bottlenecks:
            component = bottleneck.component_name

            # High cost recommendations
            if bottleneck.avg_cost > 0.5:
                recommendations.append(
                    f"Consider caching or reducing API calls in {component} "
                    f"(avg cost: ${bottleneck.avg_cost:.2f})"
                )

            # High duration recommendations
            if bottleneck.avg_duration_ms > 5000:
                recommendations.append(
                    f"Optimize {component} for faster execution "
                    f"(avg duration: {bottleneck.avg_duration_ms/1000:.1f}s)"
                )

            # Many calls recommendations
            if bottleneck.total_calls > 100:
                recommendations.append(
                    f"Consider batching operations in {component} "
                    f"({bottleneck.total_calls} calls)"
                )

        return recommendations


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler(enable_profiling: bool = True) -> PerformanceProfiler:
    """
    Get or create global profiler instance.

    Args:
        enable_profiling: Enable profiling

    Returns:
        PerformanceProfiler instance
    """
    global _global_profiler

    if _global_profiler is None:
        _global_profiler = PerformanceProfiler(enable_profiling=enable_profiling)

    return _global_profiler


def profile(component: str, operation: str):
    """
    Convenience decorator for profiling.

    Usage:
        @profile("MyComponent", "my_operation")
        def my_function():
            pass
    """
    profiler = get_profiler()
    return profiler.profile(component, operation)
