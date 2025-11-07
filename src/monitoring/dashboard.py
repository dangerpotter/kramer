"""
Monitoring Dashboard for Kramer Discovery System.

Generates HTML dashboard with metrics visualization from structured logs.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from collections import defaultdict


class MonitoringDashboard:
    """Generate monitoring dashboard from logs and metrics."""

    def __init__(self, log_dir: str = "logs"):
        """
        Initialize dashboard generator.

        Args:
            log_dir: Directory containing log files
        """
        self.log_dir = Path(log_dir)

    def generate_dashboard(self, output_path: str = "dashboard.html") -> Path:
        """
        Generate HTML dashboard from logs.

        Args:
            output_path: Path to output HTML file

        Returns:
            Path to generated dashboard
        """
        # Load log data
        log_entries = self._load_json_logs()

        # Compute statistics
        stats = self._compute_statistics(log_entries)

        # Generate HTML
        html = self._generate_html(stats, log_entries)

        # Write to file
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            f.write(html)

        return output_file

    def _load_json_logs(self) -> List[Dict[str, Any]]:
        """Load all JSON log entries."""
        entries = []

        for log_file in self.log_dir.glob("*.jsonl"):
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            entries.append(json.loads(line))
            except Exception as e:
                print(f"Warning: Failed to load {log_file}: {e}")

        # Sort by timestamp
        entries.sort(key=lambda e: e.get("timestamp", ""))

        return entries

    def _compute_statistics(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute dashboard statistics from log entries."""

        stats = {
            "total_events": len(entries),
            "event_counts": defaultdict(int),
            "level_counts": defaultdict(int),
            "total_cost": 0.0,
            "total_api_calls": 0,
            "total_errors": 0,
            "cycles_completed": 0,
            "tasks_completed": 0,
            "hypotheses_generated": 0,
            "hypotheses_tested": 0,
            "findings_added": 0,
            "avg_cycle_duration_ms": 0.0,
            "avg_task_duration_ms": 0.0,
            "cycle_timeline": [],
            "cost_timeline": [],
            "error_list": [],
        }

        cycle_durations = []
        task_durations = []

        for entry in entries:
            # Count events by type
            event_type = entry.get("event_type", "unknown")
            stats["event_counts"][event_type] += 1

            # Count by level
            level = entry.get("level", "INFO")
            stats["level_counts"][level] += 1

            # Track costs
            if entry.get("cost"):
                stats["total_cost"] += entry["cost"]
                stats["cost_timeline"].append({
                    "timestamp": entry.get("timestamp"),
                    "cost": entry["cost"],
                    "event": event_type
                })

            # Track API calls
            if event_type in ["agent_call", "api_call"]:
                stats["total_api_calls"] += 1

            # Track errors
            if level == "ERROR":
                stats["total_errors"] += 1
                stats["error_list"].append({
                    "timestamp": entry.get("timestamp"),
                    "message": entry.get("message"),
                    "error_type": entry.get("error_type"),
                })

            # Track cycles
            if event_type == "cycle_end":
                stats["cycles_completed"] += 1
                if entry.get("duration_ms"):
                    cycle_durations.append(entry["duration_ms"])
                    stats["cycle_timeline"].append({
                        "cycle_number": entry.get("cycle_number"),
                        "duration_ms": entry["duration_ms"],
                        "cost": entry.get("cost", 0.0)
                    })

            # Track tasks
            if event_type == "task_end":
                stats["tasks_completed"] += 1
                if entry.get("duration_ms"):
                    task_durations.append(entry["duration_ms"])

            # Track discoveries
            if event_type == "hypothesis_generated":
                stats["hypotheses_generated"] += 1
            if event_type == "hypothesis_tested":
                stats["hypotheses_tested"] += 1
            if event_type == "finding_added":
                stats["findings_added"] += 1

        # Compute averages
        if cycle_durations:
            stats["avg_cycle_duration_ms"] = sum(cycle_durations) / len(cycle_durations)
        if task_durations:
            stats["avg_task_duration_ms"] = sum(task_durations) / len(task_durations)

        return stats

    def _generate_html(self, stats: Dict[str, Any], entries: List[Dict[str, Any]]) -> str:
        """Generate HTML dashboard."""

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kramer Discovery System - Monitoring Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f7fa;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        header {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}

        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}

        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .metric-card h3 {{
            color: #7f8c8d;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 10px;
            text-transform: uppercase;
        }}

        .metric-value {{
            font-size: 32px;
            font-weight: 700;
            color: #2c3e50;
        }}

        .metric-value.success {{
            color: #27ae60;
        }}

        .metric-value.warning {{
            color: #f39c12;
        }}

        .metric-value.danger {{
            color: #e74c3c;
        }}

        .section {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .section h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
        }}

        th {{
            background: #f8f9fa;
            color: #2c3e50;
            font-weight: 600;
        }}

        tr:hover {{
            background: #f8f9fa;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}

        .badge.success {{
            background: #d4edda;
            color: #155724;
        }}

        .badge.warning {{
            background: #fff3cd;
            color: #856404;
        }}

        .badge.danger {{
            background: #f8d7da;
            color: #721c24;
        }}

        .chart-container {{
            margin: 20px 0;
            height: 300px;
        }}

        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #ecf0f1;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }}

        .progress-fill {{
            height: 100%;
            background: #3498db;
            transition: width 0.3s ease;
        }}

        .timeline {{
            position: relative;
            padding-left: 30px;
        }}

        .timeline-item {{
            position: relative;
            padding-bottom: 20px;
        }}

        .timeline-item::before {{
            content: '';
            position: absolute;
            left: -24px;
            top: 0;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #3498db;
        }}

        .timeline-item::after {{
            content: '';
            position: absolute;
            left: -18px;
            top: 12px;
            width: 2px;
            height: 100%;
            background: #ecf0f1;
        }}

        .timeline-item:last-child::after {{
            display: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 Kramer Discovery System</h1>
            <p class="timestamp">Dashboard generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </header>

        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Total Events</h3>
                <div class="metric-value">{stats['total_events']:,}</div>
            </div>

            <div class="metric-card">
                <h3>Total Cost</h3>
                <div class="metric-value success">${stats['total_cost']:.2f}</div>
            </div>

            <div class="metric-card">
                <h3>Cycles Completed</h3>
                <div class="metric-value">{stats['cycles_completed']}</div>
            </div>

            <div class="metric-card">
                <h3>Tasks Completed</h3>
                <div class="metric-value">{stats['tasks_completed']}</div>
            </div>

            <div class="metric-card">
                <h3>Hypotheses Generated</h3>
                <div class="metric-value">{stats['hypotheses_generated']}</div>
            </div>

            <div class="metric-card">
                <h3>Hypotheses Tested</h3>
                <div class="metric-value">{stats['hypotheses_tested']}</div>
            </div>

            <div class="metric-card">
                <h3>Findings Added</h3>
                <div class="metric-value">{stats['findings_added']}</div>
            </div>

            <div class="metric-card">
                <h3>Errors</h3>
                <div class="metric-value {('danger' if stats['total_errors'] > 0 else 'success')}">{stats['total_errors']}</div>
            </div>

            <div class="metric-card">
                <h3>API Calls</h3>
                <div class="metric-value">{stats['total_api_calls']}</div>
            </div>

            <div class="metric-card">
                <h3>Avg Cycle Duration</h3>
                <div class="metric-value">{stats['avg_cycle_duration_ms']/1000:.1f}s</div>
            </div>

            <div class="metric-card">
                <h3>Avg Task Duration</h3>
                <div class="metric-value">{stats['avg_task_duration_ms']/1000:.1f}s</div>
            </div>
        </div>

        <div class="section">
            <h2>📊 Event Distribution</h2>
            <table>
                <thead>
                    <tr>
                        <th>Event Type</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_event_table_rows(stats['event_counts'], stats['total_events'])}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>📈 Cycle Timeline</h2>
            <table>
                <thead>
                    <tr>
                        <th>Cycle</th>
                        <th>Duration</th>
                        <th>Cost</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_cycle_timeline_rows(stats['cycle_timeline'])}
                </tbody>
            </table>
        </div>

        {self._generate_errors_section(stats['error_list']) if stats['error_list'] else ''}

        <div class="section">
            <h2>📝 Recent Events</h2>
            <div class="timeline">
                {self._generate_recent_events_timeline(entries[-20:])}
            </div>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _generate_event_table_rows(self, event_counts: Dict[str, int], total: int) -> str:
        """Generate HTML table rows for event distribution."""
        rows = []
        for event_type, count in sorted(event_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            rows.append(f"""
                <tr>
                    <td>{event_type}</td>
                    <td>{count:,}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
            """)
        return "\n".join(rows)

    def _generate_cycle_timeline_rows(self, timeline: List[Dict[str, Any]]) -> str:
        """Generate HTML table rows for cycle timeline."""
        rows = []
        for item in timeline:
            rows.append(f"""
                <tr>
                    <td>Cycle {item.get('cycle_number', 'N/A')}</td>
                    <td>{item.get('duration_ms', 0)/1000:.1f}s</td>
                    <td>${item.get('cost', 0):.4f}</td>
                </tr>
            """)
        return "\n".join(rows) if rows else "<tr><td colspan='3'>No cycle data available</td></tr>"

    def _generate_errors_section(self, error_list: List[Dict[str, Any]]) -> str:
        """Generate HTML section for errors."""
        rows = []
        for error in error_list[-10:]:  # Show last 10 errors
            rows.append(f"""
                <tr>
                    <td>{error.get('timestamp', 'N/A')}</td>
                    <td>{error.get('error_type', 'Unknown')}</td>
                    <td>{error.get('message', 'N/A')}</td>
                </tr>
            """)

        return f"""
        <div class="section">
            <h2>⚠️ Recent Errors</h2>
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Type</th>
                        <th>Message</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """

    def _generate_recent_events_timeline(self, entries: List[Dict[str, Any]]) -> str:
        """Generate HTML timeline for recent events."""
        items = []
        for entry in reversed(entries):  # Most recent first
            timestamp = entry.get("timestamp", "")
            event_type = entry.get("event_type", "unknown")
            message = entry.get("message", "")
            level = entry.get("level", "INFO")

            badge_class = {
                "ERROR": "danger",
                "WARNING": "warning",
                "INFO": "success"
            }.get(level, "success")

            items.append(f"""
                <div class="timeline-item">
                    <strong>{timestamp}</strong>
                    <span class="badge {badge_class}">{event_type}</span>
                    <p>{message}</p>
                </div>
            """)

        return "\n".join(items) if items else "<p>No recent events</p>"


def generate_dashboard(log_dir: str = "logs", output_path: str = "dashboard.html") -> Path:
    """
    Convenience function to generate monitoring dashboard.

    Args:
        log_dir: Directory containing log files
        output_path: Path to output HTML file

    Returns:
        Path to generated dashboard
    """
    dashboard = MonitoringDashboard(log_dir)
    return dashboard.generate_dashboard(output_path)


if __name__ == "__main__":
    import sys

    log_dir = sys.argv[1] if len(sys.argv) > 1 else "logs"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "dashboard.html"

    dashboard_path = generate_dashboard(log_dir, output_path)
    print(f"Dashboard generated: {dashboard_path}")
