# Kramer - Autonomous Research Discovery Loop

An autonomous research system that iteratively explores datasets, generates hypotheses, validates them through literature searches, and builds a knowledge base of findings.

## Architecture

- **WorldModel**: Stores hypotheses, findings, papers, and research state
- **PriorityQueue**: Manages tasks by priority (high: untested hypotheses, medium: follow-ups, low: exploration)
- **DataAgent**: Analyzes datasets and generates findings
- **LiteratureAgent**: Searches scientific literature to validate hypotheses
- **CycleManager**: Orchestrates the discovery loop

## Discovery Loop

1. Query world model for top priority items
2. Generate tasks (data analysis or literature search)
3. Spawn agents in parallel
4. Collect results and update world model
5. Generate new hypotheses from findings
6. Repeat until stopping condition

## Stopping Conditions

- Max cycles reached (default: 20)
- Max time reached (default: 6 hours)
- No new findings in last 3 cycles

## Usage

```bash
python -m kramer.main --dataset data/iris.csv --max-cycles 5
```

## Installation

```bash
pip install -r requirements.txt
```
