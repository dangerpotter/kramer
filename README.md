# Kramer

A pragmatic implementation of autonomous discovery cycles for scientific research.

## Overview

Kramer is a simplified but functional system that can run autonomous discovery cycles. It focuses on the core loop of:
1. Analyzing datasets
2. Generating hypotheses
3. Searching literature
4. Synthesizing findings
5. Generating cited reports

## Architecture

- **World Model**: NetworkX graph + SQLite for persistence
- **Orchestrator**: Async task queue for cycle management
- **Agents**: Data analysis and literature search agents
- **Reporting**: Markdown report generation with Jinja2

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## Usage

```python
from src.world_model.graph import WorldModel
from src.orchestrator.cycle_manager import Orchestrator

# Create a world model
world_model = WorldModel()

# Add findings
world_model.add_finding(
    text="Temperature increases correlate with CO2 levels",
    code_link="analysis/climate_data.py:42",
    confidence=0.95
)

# Create orchestrator
orchestrator = Orchestrator(world_model)

# Run discovery cycle
await orchestrator.spawn_cycle(
    objective="Analyze climate change trends",
    max_tasks=10
)
```

## Project Structure

```
kramer/
├── src/
│   ├── world_model/     # Graph-based knowledge store
│   ├── orchestrator/    # Task spawning and cycle management
│   ├── agents/          # Data analysis and literature agents
│   └── reporting/       # Report generation
├── tests/               # Test suite
├── data/                # Sample datasets
└── outputs/             # Generated reports
```

## Testing

```bash
pytest tests/
```

## Phase 1: Core Infrastructure ✓

- [x] Project structure
- [x] WorldModel with NetworkX + SQLite
- [x] Basic orchestrator
- [ ] Agent implementations (Phase 2)
- [ ] Report generation (Phase 2)

## License

See LICENSE file.
