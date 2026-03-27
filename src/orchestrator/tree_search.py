"""
Tree Search Over Hypotheses — Parallel branch exploration orchestrator.

Explores multiple hypotheses as parallel branches, scores them after each
generation, prunes weak ones, and converges on the most promising line of inquiry.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.orchestrator.cycle_manager import (
    CycleType,
    Orchestrator,
    TaskStatus,
    TaskType,
)
from src.orchestrator.sub_objectives import ObjectiveTracker
from src.reporting.report_generator import ReportGenerator
from src.utils.llm_client import get_llm_client
from src.world_model.graph import NodeType, WorldModel


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
        self._llm_client = None

    @property
    def llm_client(self):
        """Lazy-init LLM client."""
        if self._llm_client is None:
            from src.utils.llm_client import get_llm_client
            self._llm_client = get_llm_client()
        return self._llm_client

        self._objective: str = ""
        self.objective_tracker: Optional[ObjectiveTracker] = None

    async def run(
        self,
        objective: str,
        initial_hypotheses: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point. Full tree search loop.

        Args:
            objective: The research objective to explore
            initial_hypotheses: Optional pre-defined hypotheses to branch on

        Returns:
            Results dict with branches, generations_run, total_cost,
            final_report, winning_branch_id
        """
        self._objective = objective
        print(f"\n🌳 Tree Search starting: {objective[:80]}")

        # 1. Ideation phase
        if initial_hypotheses:
            hypothesis_nodes = []
            for hyp_text in initial_hypotheses[: self.config.max_branches]:
                hyp_id = self.world_model.add_hypothesis(
                    text=hyp_text, metadata={"source": "tree_search_initial"}
                )
                hypothesis_nodes.append({"id": hyp_id, "text": hyp_text})
        else:
            hypothesis_nodes = await self._ideation_phase(objective)

        if not hypothesis_nodes:
            print("  ⚠️  No hypotheses generated, aborting tree search")
            return {
                "branches": {},
                "generations_run": 0,
                "total_cost": self.total_cost,
                "final_report": None,
                "winning_branch_id": None,
            }

        # 2. Create branches
        for node in hypothesis_nodes[: self.config.max_branches]:
            branch = Branch(
                hypothesis_id=node["id"],
                hypothesis_text=node["text"],
            )
            self.branches[branch.branch_id] = branch
            print(f"  🌿 Branch {branch.branch_id[:8]}: {node['text'][:60]}")

        # 2b. Decompose objective into sub-questions for progress tracking
        self.objective_tracker = ObjectiveTracker(self.world_model)
        await self.objective_tracker.decompose(objective)
        print(f"  📋 Decomposed into {len(self.objective_tracker.sub_objectives)} sub-questions")

        # 3. Generation loop
        while self.generation < self.config.max_generations:
            active = [b for b in self.branches.values() if b.status == "active"]
            if not active:
                print("  ⚠️  No active branches remaining")
                break

            print(f"\n── Generation {self.generation + 1} ({len(active)} active branches) ──")

            await self._run_generation()
            await self._score_branches()
            self._prune_branches()

            if self._check_convergence():
                break

        # 4. Mark remaining active branches as completed
        for branch in self.branches.values():
            if branch.status == "active":
                branch.status = "completed"

        # 5. Final synthesis
        result = await self._synthesize_results(objective)

        # Determine winner
        all_branches = list(self.branches.values())
        completed = [b for b in all_branches if b.status == "completed"]
        winning = max(completed, key=lambda b: b.score) if completed else None

        print(f"\n🌳 Tree Search complete: {self.generation} generations, ${self.total_cost:.2f}")
        if winning:
            print(f"  🏆 Winner: {winning.branch_id[:8]} (score: {winning.score:.3f})")

        return {
            "branches": {bid: b.to_dict() for bid, b in self.branches.items()},
            "generations_run": self.generation,
            "total_cost": self.total_cost,
            "final_report": result.get("report_path"),
            "winning_branch_id": winning.branch_id if winning else None,
        }

    async def _ideation_phase(self, objective: str) -> List[Dict[str, Any]]:
        """Run an ideation cycle to generate initial hypotheses."""
        print("  💡 Ideation phase: generating hypotheses...")

        cycle = self.orchestrator.create_cycle(
            objective=objective,
            max_tasks=5,
        )
        cycle.cycle_type = CycleType.EXPLORATION
        cycle.status = TaskStatus.RUNNING
        cycle.started_at = datetime.utcnow()

        # Add search + hypothesis generation tasks
        self.orchestrator.create_task(
            cycle_id=cycle.cycle_id,
            task_type=TaskType.SEARCH_LITERATURE,
            objective=f"Search literature for: {objective}",
            context={"max_papers": 10},
        )
        self.orchestrator.create_task(
            cycle_id=cycle.cycle_id,
            task_type=TaskType.GENERATE_HYPOTHESIS,
            objective=f"Generate hypotheses for: {objective}",
            context={"objective": objective},
        )

        await self.orchestrator._execute_cycle(cycle)
        self.total_cost += cycle.budget_used

        # Collect untested hypotheses from world model
        hypothesis_nodes = []
        for node_id, data in self.world_model.graph.nodes(data=True):
            if data.get("node_type") == "hypothesis":
                meta = data.get("metadata", {})
                if meta.get("test_outcome") not in [
                    "supported",
                    "refuted",
                    "inconclusive",
                ]:
                    hypothesis_nodes.append(
                        {"id": node_id, "text": data.get("text", "")}
                    )

        print(f"  💡 Generated {len(hypothesis_nodes)} hypotheses")
        return hypothesis_nodes

    async def _run_generation(self):
        """Run one cycle per active branch in parallel."""
        active = [b for b in self.branches.values() if b.status == "active"]
        await asyncio.gather(*[self._run_branch_cycle(b) for b in active])
        self.generation += 1

    async def _run_branch_cycle(self, branch: Branch) -> None:
        """Run a single focused cycle for one branch."""
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
            context["branch_id"] = branch.branch_id
            self.orchestrator.create_task(
                cycle_id=cycle.cycle_id,
                task_type=task_type,
                objective=task_objective,
                context=context,
            )

        await self.orchestrator._execute_cycle(cycle)

        branch.total_cost += cycle.budget_used
        self.total_cost += cycle.budget_used
        branch.cycles_run += 1

        self._tag_branch_findings(branch, cycle)

    def _plan_branch_tasks(self, branch: Branch, cycle) -> List[tuple]:
        """
        Create tasks focused on this branch's hypothesis.

        Returns:
            List of (TaskType, objective, context) tuples
        """
        tasks = []
        branch_ctx = self._get_branch_context(branch.branch_id)

        # Check if hypothesis has been tested
        hyp_node = self.world_model.graph.nodes.get(branch.hypothesis_id, {})
        test_outcome = hyp_node.get("metadata", {}).get("test_outcome", "")

        if test_outcome not in ["supported", "refuted", "inconclusive"]:
            tasks.append((
                TaskType.TEST_HYPOTHESIS,
                f"Test hypothesis: {branch.hypothesis_text[:80]}",
                {"hypothesis_id": branch.hypothesis_id},
            ))

        tasks.append((
            TaskType.SEARCH_LITERATURE,
            f"Search literature related to: {branch.hypothesis_text[:80]}",
            {"max_papers": 5},
        ))

        if self.orchestrator.dataset_path:
            tasks.append((
                TaskType.ANALYZE_DATA,
                f"Analyze data for evidence regarding: {branch.hypothesis_text[:80]}",
                {"dataset_path": self.orchestrator.dataset_path},
            ))

        return tasks

    def _get_branch_context(self, branch_id: str) -> Dict[str, Any]:
        """Filter world model to show only shared + this branch's findings."""
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
                findings.append({
                    "id": node_id,
                    "text": data.get("text", ""),
                    "confidence": data.get("confidence", 0.0),
                })
            elif node_type == "hypothesis":
                hypotheses.append({
                    "id": node_id,
                    "text": data.get("text", ""),
                    "confidence": data.get("confidence", 0.0),
                })
            elif node_type == "paper":
                papers.append({
                    "id": node_id,
                    "text": data.get("text", ""),
                    "title": data.get("metadata", {}).get("title", ""),
                })

        return {"findings": findings, "hypotheses": hypotheses, "papers": papers}

    def _tag_branch_findings(self, branch: Branch, cycle) -> None:
        """Tag new nodes created during the cycle with branch_id."""
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

    def _compute_branch_score(self, branch: Branch) -> float:
        """Compute heuristic score for a branch on 4 dimensions."""
        findings = [
            self.world_model.graph.nodes[nid]
            for nid in branch.findings_ids
            if self.world_model.graph.has_node(nid)
        ]

        if not findings:
            return 0.1

        # Evidence quality: avg confidence of findings
        evidence_quality = sum(
            f.get("confidence", 0.0) for f in findings
        ) / len(findings)

        # Hypothesis support: check test outcome
        hyp_node = self.world_model.graph.nodes.get(branch.hypothesis_id, {})
        test_outcome = hyp_node.get("metadata", {}).get("test_outcome", "")
        hypothesis_support = {
            "supported": 1.0,
            "inconclusive": 0.4,
            "refuted": 0.1,
        }.get(test_outcome, 0.5)

        # Novelty: avg novelty of findings
        novelty = sum(
            f.get("metadata", {}).get("novelty", 0.5) for f in findings
        ) / len(findings)

        # Objective relevance: default for v1
        objective_relevance = 0.5

        score = (
            0.35 * evidence_quality
            + 0.25 * hypothesis_support
            + 0.20 * novelty
            + 0.20 * objective_relevance
        )
        return round(score, 3)

    async def _score_branches(self):
        """Score each active branch using heuristics + LLM comparative assessment."""
        active = [b for b in self.branches.values() if b.status == "active"]

        # Compute heuristic scores
        for branch in active:
            branch.score = self._compute_branch_score(branch)
            branch.score_history.append(branch.score)

        # LLM comparative assessment (adjusts scores)
        if len(active) > 1:
            await self._llm_compare_branches(active)

    async def _llm_compare_branches(self, active: List[Branch]) -> None:
        """Use LLM to comparatively assess active branches and adjust scores."""
        branch_summaries = []
        for branch in active:
            ctx = self._get_branch_context(branch.branch_id)
            key_findings = [
                f["text"][:100] for f in ctx["findings"][:3]
            ]
            branch_summaries.append(
                f"- Branch {branch.branch_id[:8]}:\n"
                f"  Hypothesis: {branch.hypothesis_text[:120]}\n"
                f"  Key findings: {key_findings}\n"
                f"  Current score: {branch.score:.3f}\n"
                f"  Cycles run: {branch.cycles_run}"
            )

        prompt = f"""You are evaluating parallel research branches. Each branch explores a different hypothesis.

Original objective: {self._objective}

Branch summaries:
{chr(10).join(branch_summaries)}

For each branch, assess:
1. How promising is this line of inquiry?
2. Is it making progress toward the objective?
3. Should it continue, or has it exhausted its potential?

Return JSON array:
[
  {{"branch_id": "...", "adjusted_score": 0.0-1.0, "reasoning": "...", "should_continue": true/false}}
]

Return ONLY the JSON array, no other text."""

        try:
            response = self.llm_client.create_message(
                model=os.getenv("CLAUDE_MODEL"),
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text

            # Parse JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            assessments = json.loads(response_text)

            # Build lookup by branch_id prefix
            branch_lookup = {b.branch_id: b for b in active}
            for assessment in assessments:
                bid = assessment.get("branch_id", "")
                # Match on full ID or prefix
                matched_branch = branch_lookup.get(bid)
                if not matched_branch:
                    for full_id, b in branch_lookup.items():
                        if full_id.startswith(bid):
                            matched_branch = b
                            break

                if matched_branch:
                    llm_score = float(assessment.get("adjusted_score", matched_branch.score))
                    # Average heuristic and LLM score
                    matched_branch.score = round(
                        (matched_branch.score + llm_score) / 2, 3
                    )
                    print(
                        f"  📊 Branch {matched_branch.branch_id[:8]}: "
                        f"score={matched_branch.score:.3f} "
                        f"({assessment.get('reasoning', '')[:60]})"
                    )
        except Exception as e:
            print(f"  ⚠️  LLM branch comparison failed: {e}")

    def _prune_branches(self):
        """Prune branches below min score or over max_branches."""
        active = [b for b in self.branches.values() if b.status == "active"]

        # Never prune the last branch
        if len(active) <= 1:
            return

        # Hard prune: below min score
        for branch in active:
            if branch.score < self.config.min_branch_score and len(active) > 1:
                branch.status = "pruned"
                print(
                    f"  ✂️  Pruned branch {branch.branch_id[:8]} "
                    f"(score: {branch.score:.3f})"
                )
                active = [b for b in self.branches.values() if b.status == "active"]

        # Soft prune: too many branches
        while len(active) > self.config.max_branches and len(active) > 1:
            worst = min(active, key=lambda b: b.score)
            worst.status = "pruned"
            print(
                f"  ✂️  Pruned branch {worst.branch_id[:8]} "
                f"(score: {worst.score:.3f}, over max_branches)"
            )
            active = [b for b in self.branches.values() if b.status == "active"]

    def _check_convergence(self) -> bool:
        """Check if we should stop the generation loop."""
        active = [b for b in self.branches.values() if b.status == "active"]

        if not active:
            return True  # everything pruned

        best = max(active, key=lambda b: b.score)

        # Score threshold met
        if best.score >= self.config.convergence_threshold:
            print(
                f"  🎯 Convergence: branch {best.branch_id[:8]} "
                f"score {best.score:.3f} >= {self.config.convergence_threshold}"
            )
            return True

        # Sub-objective completion check
        if self.objective_tracker is not None and self.objective_tracker.is_complete():
            print("  📋 Convergence: sub-objectives sufficiently answered")
            return True

        # Max generations reached
        if self.generation >= self.config.max_generations:
            print(f"  ⏰ Max generations ({self.config.max_generations}) reached")
            return True

        # Budget exhausted
        budget_remaining = (
            self.orchestrator.max_total_budget - self.orchestrator.total_budget_used
        )
        if budget_remaining < self.config.branch_budget:
            print(f"  💰 Budget exhausted (${budget_remaining:.2f} remaining)")
            return True

        return False

    async def _synthesize_results(self, objective: str) -> Dict[str, Any]:
        """Merge findings from all branches into a final synthesis."""
        print("\n📝 Synthesizing results across all branches...")

        # Collect findings from all branches
        all_findings = []
        for branch in self.branches.values():
            weight = 1.0 if branch.status == "completed" else 0.5
            ctx = self._get_branch_context(branch.branch_id)
            for finding in ctx["findings"]:
                all_findings.append({
                    **finding,
                    "branch_id": branch.branch_id,
                    "branch_hypothesis": branch.hypothesis_text,
                    "branch_status": branch.status,
                    "branch_score": branch.score,
                    "weight": weight,
                })

        # Build tree visualization text
        tree_lines = ["## Branch Exploration Tree\n"]
        for branch in self.branches.values():
            status_icon = {
                "completed": "✅",
                "pruned": "✂️",
                "active": "🔄",
            }.get(branch.status, "?")
            tree_lines.append(
                f"{status_icon} Branch {branch.branch_id[:8]} "
                f"(score: {branch.score:.3f}, {branch.cycles_run} cycles, "
                f"${branch.total_cost:.2f})"
            )
            tree_lines.append(f"   Hypothesis: {branch.hypothesis_text[:100]}")
            tree_lines.append(f"   Score history: {branch.score_history}")
            if branch.status == "pruned":
                tree_lines.append("   (Pruned — weaker evidence)")
            tree_lines.append("")

        tree_text = "\n".join(tree_lines)

        # LLM synthesis
        findings_text = "\n".join(
            f"- [{f['branch_status']}] (score {f['branch_score']:.2f}) {f['text'][:150]}"
            for f in all_findings[:30]
        )

        # Include sub-objective progress if available
        sub_obj_text = ""
        if self.objective_tracker and self.objective_tracker.sub_objectives:
            progress = self.objective_tracker._build_progress_result()
            sub_obj_lines = [f"\n## Sub-Objective Progress ({progress['score']:.0%} complete)\n"]
            for so in progress['sub_objectives']:
                status_icon = {"answered": "✅", "partial": "🔶", "unanswered": "❌"}.get(so['status'], "?")
                sub_obj_lines.append(f"{status_icon} {so['question']}")
                if so['answer_summary']:
                    sub_obj_lines.append(f"   Answer: {so['answer_summary'][:150]}")
            sub_obj_text = "\n".join(sub_obj_lines)

        prompt = f"""Given these findings from multiple research branches exploring the objective below, synthesize the key conclusions.

Objective: {objective}

{tree_text}
{sub_obj_text}

Findings (branches marked as pruned had weaker evidence):
{findings_text}

Provide a synthesis that:
1. Identifies the strongest conclusions supported across branches
2. Notes where branches disagreed and what that implies
3. Highlights the most promising direction for future research
4. Gives appropriate weight to findings from higher-scoring branches

Return your synthesis as a clear, structured summary."""

        synthesis_text = ""
        try:
            response = self.llm_client.create_message(
                model=os.getenv("CLAUDE_MODEL"),
                max_tokens=3000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            synthesis_text = response.content[0].text
        except Exception as e:
            synthesis_text = f"Synthesis failed: {e}"

        # Add synthesis findings to world model
        self.world_model.add_finding(
            text=synthesis_text[:2000],
            confidence=0.8,
            metadata={"source": "tree_search_synthesis"},
        )

        # Generate final report
        report_path = None
        try:
            from pathlib import Path

            report_generator = ReportGenerator(world_model=self.world_model)
            output_path = Path(self.orchestrator.output_dir) / "tree_search_report.md"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            report_result = report_generator.generate_report(output_path=output_path)
            report_path = str(report_result.get("report", output_path))
        except Exception as e:
            print(f"  ⚠️  Report generation failed: {e}")

        # Find winning branch
        completed = [b for b in self.branches.values() if b.status == "completed"]
        winning = max(completed, key=lambda b: b.score) if completed else None

        return {
            "winning_branch": winning.to_dict() if winning else None,
            "all_branches": {bid: b.to_dict() for bid, b in self.branches.items()},
            "report_path": report_path,
            "total_cost": self.total_cost,
            "synthesis": synthesis_text,
        }
