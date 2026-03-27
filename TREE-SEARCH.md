# Tree Search Over Hypotheses — Implementation Guide

## Overview
Add a tree search orchestration mode that explores multiple hypotheses as parallel branches, scores them after each generation, prunes weak ones, and converges on the most promising line of inquiry.

## New File: `src/orchestrator/tree_search.py`

Create this file with the following components:

### Data Classes

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from uuid import uuid4

@dataclass
class Branch:
    branch_id: str = field(default_factory=lambda: str(uuid4()))
    hypothesis_id: str = ""
    hypothesis_text: str = ""
    parent_branch_id: Optional[str] = None
    status: str = "active"  # "active" | "pruned" | "completed"
    cycles_run: int = 0
    score: float = 0.0
    score_history: List[float] = field(default_factory=list)
    total_cost: float = 0.0
    findings_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "branch_id": self.branch_id,
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_text": self.hypothesis_text,
            "parent_branch_id": self.parent_branch_id,
            "status": self.status,
            "cycles_run": self.cycles_run,
            "score": self.score,
            "score_history": self.score_history,
            "total_cost": self.total_cost,
            "findings_ids": self.findings_ids,
        }

@dataclass
class TreeSearchConfig:
    max_branches: int = 3
    max_generations: int = 5
    min_branch_score: float = 0.3
    branch_budget: float = 2.0
    convergence_threshold: float = 0.8
    allow_sub_branching: bool = False  # disabled for v1
    tasks_per_branch_cycle: int = 5
```

### TreeSearchOrchestrator Class

```python
import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from src.utils.llm_client import get_llm_client
from src.utils.cost_tracker import CostTracker
from src.world_model.graph import NodeType, WorldModel
from src.orchestrator.cycle_manager import (
    Orchestrator, Cycle, Task, TaskType, TaskStatus, CycleType
)

class TreeSearchOrchestrator:
    """
    Tree search over hypotheses. Wraps the existing Orchestrator.
    
    Flow:
    1. Generate initial hypotheses (ideation cycle)
    2. Create a Branch per hypothesis (up to max_branches)
    3. Run one focused cycle per active branch (parallel)
    4. Score all branches
    5. Prune weakest
    6. Check convergence
    7. Repeat 3-6 until converged or max_generations hit
    8. Final synthesis merging all branch findings
    """
    
    def __init__(
        self,
        orchestrator: Orchestrator,
        config: Optional[TreeSearchConfig] = None,
    ):
        self.orchestrator = orchestrator
        self.world_model = orchestrator.world_model
        self.config = config or TreeSearchConfig()
        self.branches: Dict[str, Branch] = {}
        self.generation: int = 0
        self.total_cost: float = 0.0
        self.llm_client = get_llm_client()
```

### Key Methods to Implement

**`async def run(self, objective: str, initial_hypotheses: List[str] = None) -> Dict[str, Any]`**

Main entry point. Full tree search loop:

1. **Ideation phase** (if no initial_hypotheses provided):
   - Create a single cycle with tasks: SEARCH_LITERATURE + GENERATE_HYPOTHESIS focused on the objective
   - Use the orchestrator's existing spawn_cycle() for this
   - Collect generated hypothesis IDs from the world model (node_type == "hypothesis", untested)
   
2. **Create branches** — one Branch per hypothesis, up to max_branches. Store hypothesis_id and hypothesis_text from the world model node.

3. **Generation loop** (while generation < max_generations):
   a. Call `_run_generation()` — runs one cycle per active branch
   b. Call `_score_branches()` — evaluate each branch
   c. Call `_prune_branches()` — remove weak ones
   d. Call `_check_convergence()` — should we stop?
   e. Increment generation
   
4. **Final synthesis** — call `_synthesize_results()` to merge all findings and generate report

5. Return results dict with: branches (all, including pruned), generations_run, total_cost, final_report, winning_branch_id

**`async def _run_generation(self)`**

For each active branch, run a focused cycle IN PARALLEL using asyncio.gather():

```python
async def _run_branch_cycle(self, branch: Branch) -> None:
    """Run a single focused cycle for one branch."""
    # Create cycle with branch-focused objective
    cycle = self.orchestrator.create_cycle(
        objective=f"Investigate hypothesis: {branch.hypothesis_text}",
        max_tasks=self.config.tasks_per_branch_cycle,
    )
    cycle.cycle_type = CycleType.EXPLORATION
    cycle.status = TaskStatus.RUNNING
    cycle.started_at = datetime.utcnow()
    
    # Plan tasks using branch-filtered context
    tasks = self._plan_branch_tasks(branch, cycle)
    for task_type, task_objective, context in tasks:
        # Inject branch_id into every task's context
        context["branch_id"] = branch.branch_id
        self.orchestrator.create_task(
            cycle_id=cycle.cycle_id,
            task_type=task_type,
            objective=task_objective,
            context=context,
        )
    
    # Execute
    await self.orchestrator._execute_cycle(cycle)
    
    # Track cost and findings
    branch.total_cost += cycle.budget_used
    self.total_cost += cycle.budget_used
    branch.cycles_run += 1
    
    # Collect findings created during this cycle and tag them
    self._tag_branch_findings(branch, cycle)
```

Run all branch cycles in parallel:
```python
async def _run_generation(self):
    active = [b for b in self.branches.values() if b.status == "active"]
    await asyncio.gather(*[self._run_branch_cycle(b) for b in active])
    self.generation += 1
```

**`def _plan_branch_tasks(self, branch: Branch, cycle: Cycle) -> List[tuple]`**

Create tasks focused on THIS branch's hypothesis. Use branch-filtered world model context.

Standard set per branch cycle:
- TEST_HYPOTHESIS (if not yet tested) — with the branch's hypothesis_id
- SEARCH_LITERATURE — focused on the hypothesis topic
- ANALYZE_DATA — if dataset available, focused analysis
- Can also use LLM to plan (similar to existing _plan_initial_tasks but with branch context)

**`def _get_branch_context(self, branch_id: str) -> Dict[str, Any]`**

Filter world model to show only:
- Shared findings (no branch_id in metadata)
- This branch's findings (metadata.branch_id == branch_id)
- Exclude other branches' findings

```python
def _get_branch_context(self, branch_id: str) -> Dict[str, Any]:
    findings = []
    hypotheses = []
    papers = []
    
    for node_id, data in self.world_model.graph.nodes(data=True):
        meta = data.get("metadata", {})
        node_branch = meta.get("branch_id")
        
        # Include if: no branch tag (shared) OR same branch
        if node_branch is not None and node_branch != branch_id:
            continue
        
        node_type = data.get("node_type")
        if node_type == "finding":
            findings.append({"id": node_id, "text": data.get("text", ""), "confidence": data.get("confidence", 0.0)})
        elif node_type == "hypothesis":
            hypotheses.append({"id": node_id, "text": data.get("text", ""), "confidence": data.get("confidence", 0.0)})
        elif node_type == "paper":
            papers.append({"id": node_id, "text": data.get("text", ""), "title": data.get("metadata", {}).get("title", "")})
    
    return {"findings": findings, "hypotheses": hypotheses, "papers": papers}
```

**`def _tag_branch_findings(self, branch: Branch, cycle: Cycle)`**

After a branch cycle completes, find new nodes created during the cycle (by timestamp) and tag them with branch_id:

```python
def _tag_branch_findings(self, branch: Branch, cycle: Cycle):
    if not cycle.started_at:
        return
    for node_id, data in self.world_model.graph.nodes(data=True):
        created_at = data.get("created_at")
        if not created_at:
            continue
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except (ValueError, AttributeError):
                continue
        if created_at >= cycle.started_at:
            meta = data.get("metadata", {})
            if "branch_id" not in meta:
                meta["branch_id"] = branch.branch_id
                meta["generation"] = self.generation
                self.world_model.graph.nodes[node_id]["metadata"] = meta
                branch.findings_ids.append(node_id)
```

**`async def _score_branches(self)`**

Score each active branch on 4 dimensions. Use both heuristics AND an LLM comparative assessment.

Heuristic scoring per branch:
```python
def _compute_branch_score(self, branch: Branch) -> float:
    findings = [self.world_model.graph.nodes[nid] for nid in branch.findings_ids 
                if self.world_model.graph.has_node(nid)]
    
    if not findings:
        return 0.1
    
    # Evidence quality: avg confidence of findings
    evidence_quality = sum(f.get("confidence", 0.0) for f in findings) / len(findings)
    
    # Hypothesis support: check test outcome
    hyp_node = self.world_model.graph.nodes.get(branch.hypothesis_id, {})
    test_outcome = hyp_node.get("metadata", {}).get("test_outcome", "")
    hypothesis_support = {"supported": 1.0, "inconclusive": 0.4, "refuted": 0.1}.get(test_outcome, 0.5)
    
    # Novelty: avg novelty of findings
    novelty = sum(f.get("metadata", {}).get("novelty", 0.5) for f in findings) / len(findings)
    
    # Objective relevance: use embedding similarity if available, else 0.5
    objective_relevance = 0.5  # default, can enhance later
    
    score = (
        0.35 * evidence_quality +
        0.25 * hypothesis_support +
        0.20 * novelty +
        0.20 * objective_relevance
    )
    return round(score, 3)
```

Then do an LLM comparative assessment across all active branches:

```python
async def _score_branches(self):
    active = [b for b in self.branches.values() if b.status == "active"]
    
    # Compute heuristic scores
    for branch in active:
        branch.score = self._compute_branch_score(branch)
        branch.score_history.append(branch.score)
    
    # LLM comparative assessment (adjusts scores)
    if len(active) > 1:
        await self._llm_compare_branches(active)
```

The LLM comparison prompt:
```
You are evaluating parallel research branches. Each branch explores a different hypothesis.

Original objective: {objective}

Branch summaries:
{for each branch: hypothesis text, key findings (2-3), current score, cycles run}

For each branch, assess:
1. How promising is this line of inquiry?
2. Is it making progress toward the objective?
3. Should it continue, or has it exhausted its potential?

Return JSON array:
[
  {"branch_id": "...", "adjusted_score": 0.0-1.0, "reasoning": "...", "should_continue": true/false}
]
```

Apply adjusted scores (average of heuristic and LLM score).

**`def _prune_branches(self)`**

```python
def _prune_branches(self):
    active = [b for b in self.branches.values() if b.status == "active"]
    
    # Never prune the last branch
    if len(active) <= 1:
        return
    
    # Hard prune: below min score
    for branch in active:
        if branch.score < self.config.min_branch_score and len(active) > 1:
            branch.status = "pruned"
            print(f"  ✂️  Pruned branch {branch.branch_id[:8]} (score: {branch.score:.3f})")
            active = [b for b in self.branches.values() if b.status == "active"]
    
    # Soft prune: too many branches
    while len(active) > self.config.max_branches and len(active) > 1:
        worst = min(active, key=lambda b: b.score)
        worst.status = "pruned"
        print(f"  ✂️  Pruned branch {worst.branch_id[:8]} (score: {worst.score:.3f}, over max_branches)")
        active = [b for b in self.branches.values() if b.status == "active"]
```

**`def _check_convergence(self) -> bool`**

```python
def _check_convergence(self) -> bool:
    active = [b for b in self.branches.values() if b.status == "active"]
    
    if not active:
        return True  # everything pruned
    
    best = max(active, key=lambda b: b.score)
    
    # Score threshold met
    if best.score >= self.config.convergence_threshold:
        print(f"  🎯 Convergence: branch {best.branch_id[:8]} score {best.score:.3f} >= {self.config.convergence_threshold}")
        return True
    
    # Max generations reached
    if self.generation >= self.config.max_generations:
        print(f"  ⏰ Max generations ({self.config.max_generations}) reached")
        return True
    
    # Budget exhausted
    budget_remaining = self.orchestrator.max_total_budget - self.orchestrator.total_budget_used
    if budget_remaining < self.config.branch_budget:
        print(f"  💰 Budget exhausted (${budget_remaining:.2f} remaining)")
        return True
    
    return False
```

**`async def _synthesize_results(self, objective: str) -> Dict[str, Any]`**

Merge findings from all branches (pruned branches contribute with lower weight):

1. Collect all findings across all branches
2. Use LLM to synthesize: "Given these findings from multiple research branches exploring {objective}, synthesize the key conclusions. Branches that were pruned (marked) had weaker evidence."
3. Add synthesis findings to world model with metadata={"source": "tree_search_synthesis"}
4. Generate final report using existing ReportGenerator
5. Include tree visualization in the report: which branches explored what, scores per generation, why branches were pruned
6. Return dict: winning_branch, all_branches, report_path, total_cost

### Integration Points

**`src/orchestrator/cycle_manager.py` — Minor changes:**

In `create_task()`, allow `branch_id` to flow through in task context (already works since context is a dict, no changes needed).

In `_plan_initial_tasks()` and `_get_world_model_summary()`, add optional `branch_id` parameter:
- If branch_id is provided, filter nodes to only show shared + branch-specific findings
- Add a method signature: `_get_world_model_summary(self, branch_id: Optional[str] = None)`
- In the node iteration loop, skip nodes where metadata.branch_id exists and != branch_id

**`src/orchestrator/agent_coordinator.py` — Tag findings with branch_id:**

In `execute_literature_search()`, when adding papers to world model, pass `branch_id` from task.context into paper metadata.

In `_update_world_model_with_test_results()`, pass `branch_id` from task context into finding metadata.

In `execute_hypothesis_generation()`, the hypothesis agent already stores metadata — ensure branch_id flows through.

**`backend/app/services/discovery_service.py` — Add tree search option:**

When creating a discovery, accept `use_tree_search: bool = False` and `tree_search_config: dict = {}`.

If `use_tree_search` is True:
```python
tree_config = TreeSearchConfig(**tree_search_config)
tree_orchestrator = TreeSearchOrchestrator(orchestrator, tree_config)
result = await tree_orchestrator.run(objective=objective)
```

**`backend/app/api/v1/discovery.py` — Add to request model:**

Add `use_tree_search` and `tree_search_config` fields to the discovery creation endpoint.

**`backend/app/models/discovery.py` — Update model:**

Add the new fields to the Pydantic model.

### What NOT to implement (v1 boundaries):
- Sub-branching (config flag exists but keep `allow_sub_branching=False`)
- Dynamic budget allocation to branches (equal budget for now)
- Cross-branch pollination during exploration
- Tree visualization in the web UI (text-based tree in report is enough)

## Commit

After ALL changes:
```bash
git add -A && git commit -m 'feat: tree search over hypotheses with parallel branch exploration

- New TreeSearchOrchestrator with Branch, TreeSearchConfig
- Parallel branch cycles with per-branch world model filtering
- Heuristic + LLM branch scoring (evidence quality, support, novelty, relevance)
- Hard/soft pruning with minimum score threshold
- Convergence detection (score threshold, max generations, budget)
- Branch-tagged findings in world model
- Final synthesis merging all branch findings
- Backend API support for use_tree_search option
- v1: no sub-branching, equal branch budgets'
```
