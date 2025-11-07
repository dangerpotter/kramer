# Production Readiness Guide

This document describes the production-ready features implemented for the Kramer autonomous discovery system.

## Overview

Kramer is now production-ready with comprehensive features for:
- **Error recovery and restart capabilities**
- **Structured logging and monitoring**
- **Hard budget enforcement**
- **Advanced hypothesis ranking**
- **Multi-evaluator consensus**
- **Computational analysis beyond basic statistics**
- **Comprehensive testing and validation**

---

## 🔄 Error Recovery & Restart

### Checkpoint Manager (`src/orchestrator/checkpoint_manager.py`)

**Purpose**: Save and restore orchestrator state for long-running discovery cycles.

**Features**:
- Automatic checkpoint creation on cycle completion
- Manual checkpoint creation
- State restoration from checkpoint
- Checkpoint versioning and cleanup

**Usage**:

```python
from src.orchestrator.checkpoint_manager import CheckpointManager

# Initialize
checkpoint_manager = CheckpointManager(
    checkpoint_dir="checkpoints",
    auto_checkpoint=True,
    max_checkpoints=10,
    checkpoint_interval=1  # Checkpoint every N cycles
)

# Create checkpoint
checkpoint = checkpoint_manager.create_orchestrator_checkpoint(
    checkpoint_id="checkpoint_1",
    research_objective="...",
    dataset_path="...",
    total_cycles=5,
    total_budget_used=12.50,
    discovery_complete=False,
    cycles=[...],
    current_cycle_number=5,
    world_model_db_path="...",
    config={...}
)

# Save checkpoint
checkpoint_path = await checkpoint_manager.save_checkpoint(checkpoint)

# Load checkpoint
checkpoint = await checkpoint_manager.load_checkpoint()
```

**Checkpoint Data**:
- Research objective
- Dataset paths
- Cycle history
- Task results
- Budget usage
- World model path
- Configuration snapshot

---

## 📊 Comprehensive Logging & Monitoring

### Structured Logger (`src/utils/structured_logger.py`)

**Purpose**: Structured JSON logging with metrics collection and monitoring support.

**Features**:
- JSON structured logging
- Automatic metrics aggregation
- Performance tracking
- Cost tracking
- Console and file outputs

**Usage**:

```python
from src.utils.structured_logger import get_logger, EventType

# Get logger
logger = get_logger("my_component", log_dir="logs")

# Log events
logger.log_cycle_start(cycle_number=1, objective="Analyze data")
logger.log_cycle_end(cycle_number=1, duration_ms=5000, budget_used=1.50, tasks_completed=3)
logger.log_hypothesis_generated(hypothesis_id="hyp_1", hypothesis_text="...", cycle_number=1)
logger.log_error("Error occurred", error_type="ValueError", stack_trace="...")

# Get metrics
metrics = logger.get_metrics()
print(f"Total cost: ${metrics['total_cost']:.2f}")
print(f"API calls: {metrics['total_api_calls']}")

# Export metrics
metrics_path = logger.export_metrics()
```

**Event Types**:
- `CYCLE_START` / `CYCLE_END`
- `TASK_START` / `TASK_END`
- `AGENT_CALL` / `API_CALL`
- `HYPOTHESIS_GENERATED` / `HYPOTHESIS_TESTED`
- `FINDING_ADDED`
- `SYNTHESIS`
- `CHECKPOINT`
- `ERROR`

### Monitoring Dashboard (`src/monitoring/dashboard.py`)

**Purpose**: Generate HTML monitoring dashboard from logs.

**Features**:
- Real-time metrics visualization
- Event distribution charts
- Cycle timeline
- Error tracking
- Cost analysis

**Usage**:

```python
from src.monitoring.dashboard import generate_dashboard

# Generate dashboard
dashboard_path = generate_dashboard(
    log_dir="logs",
    output_path="dashboard.html"
)

print(f"Dashboard: {dashboard_path}")
# Open dashboard.html in browser
```

**Dashboard Sections**:
- Key metrics (costs, cycles, hypotheses, errors)
- Event distribution table
- Cycle timeline with costs
- Recent errors
- Recent events timeline

---

## 💰 Hard Budget Enforcement

### Budget Enforcer (`src/orchestrator/budget_enforcer.py`)

**Purpose**: Enforce hard budget limits with automatic cycle termination.

**Features**:
- Per-cycle budget enforcement
- Global budget enforcement
- Per-task budget enforcement (optional)
- Automatic warnings at thresholds
- Budget projection and prediction
- Cost tracking

**Usage**:

```python
from src.orchestrator.budget_enforcer import BudgetEnforcer, BudgetExceededException

# Initialize
budget_enforcer = BudgetEnforcer(
    max_cycle_budget=10.0,      # $10 per cycle
    max_total_budget=100.0,     # $100 total
    warning_threshold=0.8,       # Warn at 80%
    critical_threshold=0.95,     # Critical at 95%
    enforce_hard_limits=True     # Raise exception when exceeded
)

# Check before starting cycle
try:
    can_start = budget_enforcer.check_can_start_cycle("cycle_1")
except BudgetExceededException as e:
    print(f"Cannot start cycle: {e}")

# Record costs
budget_enforcer.record_cost(
    cost=1.50,
    cycle_id="cycle_1",
    task_id="task_1"
)

# Get budget report
report = budget_enforcer.get_budget_report()
print(f"Total used: ${report['total']['used']:.2f}")
print(f"Remaining: ${report['total']['remaining']:.2f}")
print(f"Status: {report['total']['status']}")

# Print report
budget_enforcer.print_budget_report()
```

**Budget Levels**:
- **NORMAL**: < 80% used
- **WARNING**: 80-95% used
- **CRITICAL**: 95-100% used
- **EXCEEDED**: > 100% used

**Projections**:
- Average cycle cost
- Estimated cycles remaining
- Projected total cost

---

## 🔬 Computational Analysis Agent

### Computational Analysis Agent (`src/agents/computational_analysis_agent.py`)

**Purpose**: Advanced computational analysis beyond basic data analysis.

**Capabilities**:
- Differential equation modeling
- Monte Carlo simulations
- Statistical model fitting
- Sensitivity analysis
- Time series forecasting
- Agent-based modeling
- Parameter optimization

**Usage**:

```python
from src.agents.computational_analysis_agent import ComputationalAnalysisAgent, ModelingConfig

# Initialize
config = ModelingConfig(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-sonnet-4-20250514",
    timeout=300,
    use_extended_thinking=True
)

agent = ComputationalAnalysisAgent(config)

# Run differential equation model
result = await agent.run_differential_equation_model(
    system_description="SIR epidemic model",
    initial_conditions={"S": 990, "I": 10, "R": 0},
    time_span=(0, 100),
    dataset_path="epidemic_data.csv"
)

# Run Monte Carlo simulation
result = await agent.run_monte_carlo_simulation(
    process_description="Stock price simulation",
    parameters={"drift": 0.05, "volatility": 0.2},
    n_simulations=1000
)

# Fit statistical model
result = await agent.fit_statistical_model(
    dataset_path="data.csv",
    model_type="regression",
    target_variable="y",
    predictor_variables=["x1", "x2", "x3"]
)

# Time series forecasting
result = await agent.forecast_time_series(
    dataset_path="timeseries.csv",
    time_column="date",
    value_column="value",
    forecast_periods=30,
    method="auto"
)

# Parameter optimization
result = await agent.optimize_parameters(
    objective_description="Minimize error",
    parameters={"a": (0, 10, 5), "b": (0, 5, 2)},
    constraints=["a + b <= 10"],
    dataset_path="data.csv"
)
```

**Result Structure**:
```python
@dataclass
class SimulationResult:
    success: bool
    model_type: str
    code: str
    results: Dict[str, Any]
    plots: List[str]
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    interpretation: str
    error: Optional[str] = None
```

---

## 🎯 Advanced Hypothesis Ranking

### Hypothesis Ranker (`src/orchestrator/hypothesis_ranker.py`)

**Purpose**: Intelligent hypothesis prioritization with multi-criteria scoring.

**Features**:
- Information gain calculation
- Novelty scoring
- Testability assessment
- Cost-benefit analysis
- Strategic value alignment
- Uncertainty quantification

**Usage**:

```python
from src.orchestrator.hypothesis_ranker import HypothesisRanker, RankingWeights

# Initialize
ranker = HypothesisRanker(
    world_model=world_model,
    research_objective="Discover protein-protein interactions"
)

# Rank all hypotheses
ranked = ranker.rank_hypotheses(
    filter_tested=True,  # Exclude tested hypotheses
    top_k=10             # Top 10 hypotheses
)

# Get ranking report
report = ranker.get_ranking_report(top_k=10)
print(report)

# Score individual hypothesis
score = ranker.score_hypothesis("hypothesis_id")
print(f"Composite score: {score.composite_score:.3f}")
print(f"Information gain: {score.information_gain:.3f}")
print(f"Novelty: {score.novelty:.3f}")
print(f"Testability: {score.testability:.3f}")

# Custom weights
weights = RankingWeights(
    information_gain=0.4,
    novelty=0.3,
    testability=0.2,
    cost_benefit=0.1,
    strategic_value=0.0,
    uncertainty=0.0
)
ranker = HypothesisRanker(world_model, weights=weights)
```

**Ranking Criteria**:

1. **Information Gain** (0-1): Expected reduction in uncertainty
   - Related findings (fewer = higher gain)
   - Related hypotheses (more = higher gain)
   - Contradictions (higher gain)
   - Unexplored areas (higher gain)

2. **Novelty** (0-1): Uniqueness compared to existing hypotheses
   - Text similarity analysis
   - Concept coverage

3. **Testability** (0-1): Feasibility of testing
   - Data availability
   - Clear predictions
   - Measurable outcomes

4. **Cost-Benefit** (0-1): Expected value vs. cost
   - Estimated testing cost
   - Expected information value

5. **Strategic Value** (0-1): Alignment with research objective
   - Keyword overlap
   - Potential to unblock other work

6. **Uncertainty** (0-1): Current confidence level
   - Lower confidence = higher priority

**Composite Score**: Weighted combination of all criteria

---

## 👥 Multi-Evaluator Consensus

### Multi-Evaluator Consensus (`src/evaluation/multi_evaluator_consensus.py`)

**Purpose**: Aggregate evaluations from multiple experts with reliability metrics.

**Features**:
- Multiple consensus strategies
- Inter-rater reliability metrics (Cohen's Kappa, Fleiss' Kappa)
- Disagreement analysis
- Evaluator reliability tracking
- Confidence-weighted aggregation

**Usage**:

```python
from src.evaluation.multi_evaluator_consensus import (
    MultiEvaluatorConsensus,
    ConsensusStrategy
)

# Initialize
consensus = MultiEvaluatorConsensus(
    consensus_strategy=ConsensusStrategy.CONFIDENCE_WEIGHTED,
    min_evaluators=2,
    consensus_threshold=0.7
)

# Register evaluators
consensus.register_evaluator("expert_1", "Dr. Smith", expertise_level=1.0)
consensus.register_evaluator("expert_2", "Dr. Jones", expertise_level=0.9)
consensus.register_evaluator("expert_3", "Dr. Brown", expertise_level=0.8)

# Compute consensus for a claim
result = consensus.compute_consensus(
    claim_id="claim_1",
    evaluations=[eval1, eval2, eval3]
)

print(f"Consensus verdict: {result.consensus_verdict}")
print(f"Agreement: {result.agreement_level:.1%}")
print(f"Has consensus: {result.has_consensus}")

# Compute inter-rater reliability
reliability = consensus.compute_inter_rater_reliability(all_evaluations)

print(f"Fleiss' Kappa: {reliability.fleiss_kappa:.3f}")
print(f"Interpretation: {reliability.interpretation}")
print(f"Overall agreement: {reliability.overall_agreement:.1%}")

# Generate report
report = consensus.generate_consensus_report(
    consensus_results=[result1, result2, ...],
    reliability=reliability
)
print(report)
```

**Consensus Strategies**:

1. **MAJORITY_VOTE**: Simple majority wins
2. **CONFIDENCE_WEIGHTED**: Weight by evaluator confidence
3. **EXPERT_WEIGHTED**: Weight by expertise and reliability
4. **UNANIMOUS**: Require all evaluators to agree

**Inter-Rater Reliability Metrics**:
- **Cohen's Kappa** (2 raters): Agreement beyond chance
- **Fleiss' Kappa** (3+ raters): Multi-rater agreement
- **Overall Agreement**: Percentage of unanimous verdicts
- **Per-Category Agreement**: Agreement by verdict type

**Interpretation**:
- < 0: Poor (worse than chance)
- 0-0.2: Slight
- 0.2-0.4: Fair
- 0.4-0.6: Moderate
- 0.6-0.8: Substantial
- 0.8-1.0: Almost perfect

---

## ⚡ Performance Profiling

### Performance Profiler (`src/monitoring/profiler.py`)

**Purpose**: Identify bottlenecks and optimize performance.

**Features**:
- Function-level profiling
- Cost analysis per component
- Bottleneck identification
- Optimization recommendations
- cProfile integration

**Usage**:

```python
from src.monitoring.profiler import get_profiler, profile

# Get global profiler
profiler = get_profiler(enable_profiling=True)

# Decorator-based profiling
@profile("MyAgent", "analyze_data")
async def analyze_data(dataset):
    # ... implementation
    pass

# Manual profiling
profiler.record(
    component="DataAgent",
    operation="load_data",
    duration_ms=150.5,
    cost=0.05,
    tokens_input=100,
    tokens_output=200
)

# Identify bottlenecks
bottlenecks = profiler.identify_bottlenecks(
    threshold_ms=1000,
    threshold_cost=1.0,
    top_n=5
)

print(f"Found {len(bottlenecks.bottlenecks)} bottlenecks")
for rec in bottlenecks.recommendations:
    print(f"- {rec}")

# Print comprehensive report
profiler.print_report(top_n=10)

# Get cost breakdown
cost_breakdown = profiler.get_cost_breakdown()
for component, cost in cost_breakdown.items():
    print(f"{component}: ${cost:.2f}")

# cProfile integration
profiler.start_cprofile()
# ... run code ...
profiler.stop_cprofile()
stats = profiler.get_cprofile_stats(top_n=20)
print(stats)
```

**Bottleneck Score**: Combines execution time and cost
- Components with high scores are prioritized
- Automatic recommendations generated

---

## 🧪 Testing & Validation

### End-to-End Integration Tests (`tests/test_e2e_integration.py`)

**Purpose**: Validate full discovery pipeline with real datasets.

**Test Cases**:
- Single cycle discovery
- Multi-cycle discovery with hypothesis generation
- Discovery with checkpointing
- Budget enforcement validation
- Structured logging validation
- Hypothesis ranking integration
- Full pipeline integration

**Usage**:
```bash
# Run E2E tests (requires ANTHROPIC_API_KEY)
pytest tests/test_e2e_integration.py -v -s

# Skip integration tests
pytest tests/ -k "not integration"
```

### Long-Running Tests (`tests/test_long_running.py`)

**Purpose**: Validate system stability over 6-12 hours.

**Test Cases**:
- 6-hour continuous discovery
- 12-hour test with interruptions and recovery
- Stress test with rapid cycle spawning
- Memory leak detection

**Usage**:
```bash
# Run long-running tests (mark as slow)
pytest tests/test_long_running.py -v -s -m slow

# Run specific test
pytest tests/test_long_running.py::TestLongRunning::test_6_hour_continuous_discovery -v -s
```

**Monitoring**:
- CPU usage tracking
- Memory usage tracking
- Budget tracking
- Checkpoint creation/recovery
- Dashboard generation

### RAG Integration Tests (`tests/test_rag_integration.py`)

**Purpose**: Validate paper processor + RAG engine integration.

**Test Cases**:
- Full RAG pipeline with real paper
- RAG persistence across instances
- Multiple papers indexing
- Error handling
- Chunking quality
- Literature agent integration

**Usage**:
```bash
pytest tests/test_rag_integration.py -v -s
```

---

## 📈 Metrics & Monitoring

### Key Metrics Tracked

**Discovery Metrics**:
- Cycles completed
- Tasks completed
- Findings generated
- Hypotheses generated
- Hypotheses tested

**Performance Metrics**:
- Cycle duration (avg, min, max)
- Task duration (avg, min, max)
- Agent call duration
- API call count
- CPU usage
- Memory usage

**Cost Metrics**:
- Total cost
- Per-cycle cost
- Per-task cost
- Per-agent cost
- Budget utilization
- Cost projections

**Quality Metrics**:
- Hypothesis novelty scores
- Information gain scores
- Testability scores
- Inter-rater reliability (Kappa)
- Consensus agreement rates

### Monitoring Workflow

1. **Enable structured logging**:
   ```python
   logger = get_logger("discovery", log_dir="logs")
   ```

2. **Run discovery**:
   ```python
   results = await orchestrator.run_discovery_loop(...)
   ```

3. **Generate dashboard**:
   ```python
   generate_dashboard(log_dir="logs", output_path="dashboard.html")
   ```

4. **Review bottlenecks**:
   ```python
   profiler = get_profiler()
   profiler.print_report()
   ```

5. **Check budget**:
   ```python
   budget_enforcer.print_budget_report()
   ```

---

## 🚀 Production Deployment Checklist

### Configuration

- [ ] Set `ANTHROPIC_API_KEY` environment variable
- [ ] Configure budget limits (`max_cycle_budget`, `max_total_budget`)
- [ ] Enable checkpointing (`auto_checkpoint=True`)
- [ ] Configure log directory (`log_dir="logs"`)
- [ ] Set synthesis interval (`synthesis_interval=N`)
- [ ] Configure RAG if using full-text search

### Monitoring

- [ ] Enable structured logging
- [ ] Setup dashboard generation
- [ ] Configure performance profiling
- [ ] Setup budget alerts
- [ ] Monitor checkpoint creation

### Testing

- [ ] Run unit tests: `pytest tests/`
- [ ] Run integration tests: `pytest tests/test_e2e_integration.py`
- [ ] Run RAG tests: `pytest tests/test_rag_integration.py`
- [ ] Run long-running validation (optional): `pytest tests/test_long_running.py -m slow`

### Recovery

- [ ] Test checkpoint save/load
- [ ] Verify world model persistence
- [ ] Test budget enforcement
- [ ] Verify error handling

---

## 📚 Additional Resources

### Code Organization

```
kramer/
├── src/
│   ├── agents/
│   │   └── computational_analysis_agent.py
│   ├── evaluation/
│   │   └── multi_evaluator_consensus.py
│   ├── monitoring/
│   │   ├── dashboard.py
│   │   └── profiler.py
│   ├── orchestrator/
│   │   ├── budget_enforcer.py
│   │   ├── checkpoint_manager.py
│   │   └── hypothesis_ranker.py
│   └── utils/
│       └── structured_logger.py
└── tests/
    ├── test_e2e_integration.py
    ├── test_long_running.py
    └── test_rag_integration.py
```

### Example: Full Production Setup

```python
import asyncio
from src.orchestrator.cycle_manager import Orchestrator
from src.world_model.graph import WorldModel
from src.orchestrator.checkpoint_manager import CheckpointManager
from src.orchestrator.budget_enforcer import BudgetEnforcer
from src.utils.structured_logger import get_logger
from src.monitoring.profiler import get_profiler

async def main():
    # Setup
    world_model = WorldModel(db_path="production.db")
    logger = get_logger("production", log_dir="logs")
    profiler = get_profiler(enable_profiling=True)

    checkpoint_manager = CheckpointManager(
        checkpoint_dir="checkpoints",
        auto_checkpoint=True
    )

    budget_enforcer = BudgetEnforcer(
        max_cycle_budget=10.0,
        max_total_budget=100.0,
        enforce_hard_limits=True
    )

    orchestrator = Orchestrator(
        world_model=world_model,
        max_cycle_budget=10.0,
        max_total_budget=100.0,
        auto_synthesize=True
    )

    # Add dataset
    await world_model.add_dataset(
        dataset_id="dataset_1",
        path="data/dataset.csv",
        description="Production dataset"
    )

    # Run discovery
    logger.log_cycle_start(1, "Production discovery")

    results = await orchestrator.run_discovery_loop(
        objective="Discover insights from production data",
        max_cycles=20,
        max_time=3600 * 6  # 6 hours
    )

    # Generate reports
    profiler.print_report()
    budget_enforcer.print_budget_report()

    from src.monitoring.dashboard import generate_dashboard
    generate_dashboard("logs", "production_dashboard.html")

    await world_model.close()

asyncio.run(main())
```

---

## 🎓 Summary

Kramer is now production-ready with:

✅ **Error Recovery**: Checkpointing and state restoration
✅ **Logging & Monitoring**: Structured logs, metrics, dashboards
✅ **Budget Control**: Hard limits with automatic enforcement
✅ **Advanced Ranking**: Multi-criteria hypothesis prioritization
✅ **Consensus**: Multi-evaluator agreement with reliability metrics
✅ **Computational Analysis**: Advanced modeling beyond statistics
✅ **Testing**: E2E, long-running, and RAG integration tests
✅ **Profiling**: Bottleneck identification and optimization

The system is ready for:
- Long-running production workloads (6+ hours)
- Cost-controlled exploration ($100+ budgets)
- Multi-expert evaluation workflows
- Complex computational modeling
- Large-scale hypothesis testing

For questions or issues, refer to the test files for usage examples.
