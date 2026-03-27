# Crisper Progress Metrics — Implementation Guide

## Overview
Replace the fuzzy LLM-based objective completion check with measurable sub-objectives. At discovery start, decompose the objective into 3-7 concrete sub-questions. Track which have confident answers. Progress = answered / total.

## Changes Required

### 1. New file: `src/orchestrator/sub_objectives.py`

```python
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.utils.llm_client import get_llm_client
from src.world_model.graph import NodeType, WorldModel


@dataclass
class SubObjective:
    """A concrete, measurable sub-question of the research objective."""
    sub_id: str = field(default_factory=lambda: str(uuid4()))
    question: str = ""
    status: str = "unanswered"  # "unanswered" | "partial" | "answered"
    confidence: float = 0.0
    supporting_findings: List[str] = field(default_factory=list)  # finding node IDs
    answer_summary: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    answered_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub_id": self.sub_id,
            "question": self.question,
            "status": self.status,
            "confidence": self.confidence,
            "supporting_findings": self.supporting_findings,
            "answer_summary": self.answer_summary,
            "created_at": self.created_at,
            "answered_at": self.answered_at,
        }


class ObjectiveTracker:
    """
    Tracks progress toward a research objective via measurable sub-objectives.
    
    Usage:
        tracker = ObjectiveTracker(world_model)
        await tracker.decompose("What factors drive customer churn?")
        # ... after cycles run ...
        progress = await tracker.evaluate_progress()
        # progress = {"answered": 3, "total": 5, "score": 0.6, "sub_objectives": [...]}
    """
    
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model
        self.sub_objectives: List[SubObjective] = []
        self.objective: str = ""
        self.llm_client = get_llm_client()
        self.total_cost: float = 0.0
    
    async def decompose(self, objective: str, num_questions: int = 5) -> List[SubObjective]:
        """
        Decompose a research objective into concrete sub-questions using LLM.
        
        Args:
            objective: The high-level research objective
            num_questions: Target number of sub-questions (3-7)
        
        Returns:
            List of SubObjective instances
        """
        self.objective = objective
        
        prompt = f"""You are decomposing a research objective into concrete, measurable sub-questions.

Research Objective: {objective}

Break this into {num_questions} specific sub-questions that together fully cover the objective.

Requirements for each sub-question:
- Must be answerable with data analysis or literature review
- Must be specific and falsifiable (not vague)
- Must be independent (answering one shouldn't automatically answer another)
- Together they should cover ALL aspects of the objective

Return ONLY a JSON array of strings:
["What is the overall X rate and trend?", "Which segments show highest X?", ...]

No explanations, no numbering, just the JSON array."""

        try:
            model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
            response = self.llm_client.create_message(
                model=model,
                max_tokens=1000,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}],
            )
            
            # Track cost
            from src.utils.cost_tracker import CostTracker
            self.total_cost += CostTracker.calculate_cost(
                model, response.usage.input_tokens, response.usage.output_tokens
            )
            
            # Parse response
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text
            
            # Extract JSON
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                text = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                text = text[start:end].strip()
            
            questions = json.loads(text)
            
            # Create SubObjective for each question
            self.sub_objectives = []
            for q in questions:
                if isinstance(q, str) and len(q.strip()) > 10:
                    self.sub_objectives.append(SubObjective(question=q.strip()))
            
            # Ensure we have at least 3
            if len(self.sub_objectives) < 3:
                # Fallback: create generic sub-objectives
                self.sub_objectives = [
                    SubObjective(question=f"What are the key patterns in the data related to: {objective}?"),
                    SubObjective(question=f"What does existing literature say about: {objective}?"),
                    SubObjective(question=f"What are the main factors and their relationships for: {objective}?"),
                ]
            
            return self.sub_objectives
            
        except Exception as e:
            print(f"Error decomposing objective: {e}")
            # Fallback
            self.sub_objectives = [
                SubObjective(question=f"What are the key patterns in the data related to: {objective}?"),
                SubObjective(question=f"What does existing literature say about: {objective}?"),
                SubObjective(question=f"What are the main factors and their relationships for: {objective}?"),
            ]
            return self.sub_objectives
    
    async def evaluate_progress(self) -> Dict[str, Any]:
        """
        Evaluate which sub-objectives have been answered by current findings.
        
        Uses LLM to match findings to sub-questions and assess answer quality.
        
        Returns:
            Dict with: answered (int), partial (int), unanswered (int), total (int),
            score (float 0-1), sub_objectives (list of dicts), cost (float)
        """
        # Collect all findings from world model
        findings = []
        for node_id, data in self.world_model.graph.nodes(data=True):
            if data.get("node_type") == NodeType.FINDING.value:
                confidence = data.get("confidence", 0.0)
                if confidence >= 0.4:  # only consider findings with some confidence
                    findings.append({
                        "id": node_id,
                        "text": data.get("text", ""),
                        "confidence": confidence,
                    })
        
        if not findings or not self.sub_objectives:
            return self._build_progress_result()
        
        # Format for LLM evaluation
        findings_text = "\n".join([
            f"- [{f['id'][:8]}] (confidence: {f['confidence']:.2f}) {f['text'][:200]}"
            for f in findings[:30]  # cap to avoid token bloat
        ])
        
        sub_q_text = "\n".join([
            f"{i+1}. {so.question}"
            for i, so in enumerate(self.sub_objectives)
        ])
        
        prompt = f"""Evaluate which research sub-questions have been answered by the findings below.

Sub-questions:
{sub_q_text}

Available findings:
{findings_text}

For EACH sub-question, determine:
1. status: "answered" (confident answer exists), "partial" (some evidence but incomplete), or "unanswered" (no relevant findings)
2. confidence: 0.0-1.0 how confident the answer is
3. supporting_findings: list of finding IDs (the bracketed IDs like "abc12345") that address this question
4. answer_summary: 1-2 sentence summary of the answer (empty string if unanswered)

Return JSON array (one object per sub-question, in order):
[
  {{"status": "answered", "confidence": 0.85, "supporting_findings": ["abc12345", "def67890"], "answer_summary": "The churn rate is 5.2% monthly with an upward trend..."}}
]"""

        try:
            model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
            response = self.llm_client.create_message(
                model=model,
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            
            from src.utils.cost_tracker import CostTracker
            self.total_cost += CostTracker.calculate_cost(
                model, response.usage.input_tokens, response.usage.output_tokens
            )
            
            # Parse response
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text
            
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                text = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                text = text[start:end].strip()
            
            evaluations = json.loads(text)
            
            # Update sub-objectives with evaluations
            for i, eval_data in enumerate(evaluations):
                if i >= len(self.sub_objectives):
                    break
                if not isinstance(eval_data, dict):
                    continue
                    
                so = self.sub_objectives[i]
                so.status = eval_data.get("status", "unanswered")
                so.confidence = eval_data.get("confidence", 0.0)
                so.supporting_findings = eval_data.get("supporting_findings", [])
                so.answer_summary = eval_data.get("answer_summary", "")
                
                if so.status == "answered" and not so.answered_at:
                    so.answered_at = datetime.utcnow().isoformat()
        
        except Exception as e:
            print(f"Error evaluating progress: {e}")
        
        return self._build_progress_result()
    
    def _build_progress_result(self) -> Dict[str, Any]:
        """Build the progress result dictionary."""
        answered = sum(1 for so in self.sub_objectives if so.status == "answered")
        partial = sum(1 for so in self.sub_objectives if so.status == "partial")
        unanswered = sum(1 for so in self.sub_objectives if so.status == "unanswered")
        total = len(self.sub_objectives)
        
        # Score: answered = 1.0, partial = 0.5, unanswered = 0.0
        if total > 0:
            score = (answered + 0.5 * partial) / total
        else:
            score = 0.0
        
        return {
            "answered": answered,
            "partial": partial,
            "unanswered": unanswered,
            "total": total,
            "score": round(score, 3),
            "sub_objectives": [so.to_dict() for so in self.sub_objectives],
            "cost": self.total_cost,
        }
    
    def get_unanswered_questions(self) -> List[str]:
        """Get list of unanswered and partial sub-questions for planning focus."""
        return [
            so.question for so in self.sub_objectives 
            if so.status in ("unanswered", "partial")
        ]
    
    def is_complete(self, threshold: float = 0.8) -> bool:
        """Check if enough sub-objectives are answered."""
        result = self._build_progress_result()
        return result["score"] >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize tracker state."""
        return {
            "objective": self.objective,
            "sub_objectives": [so.to_dict() for so in self.sub_objectives],
            "total_cost": self.total_cost,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], world_model: WorldModel) -> "ObjectiveTracker":
        """Deserialize tracker state."""
        tracker = cls(world_model)
        tracker.objective = data.get("objective", "")
        tracker.total_cost = data.get("total_cost", 0.0)
        for so_data in data.get("sub_objectives", []):
            so = SubObjective(
                sub_id=so_data.get("sub_id", str(uuid4())),
                question=so_data.get("question", ""),
                status=so_data.get("status", "unanswered"),
                confidence=so_data.get("confidence", 0.0),
                supporting_findings=so_data.get("supporting_findings", []),
                answer_summary=so_data.get("answer_summary", ""),
                created_at=so_data.get("created_at", ""),
                answered_at=so_data.get("answered_at"),
            )
            tracker.sub_objectives.append(so)
        return tracker
```

### 2. Integrate into `src/orchestrator/cycle_manager.py`

**Add ObjectiveTracker to Orchestrator.__init__():**
```python
self.objective_tracker: Optional[ObjectiveTracker] = None
```

**In `run_cycle()`, after setting up the first cycle but before the main loop, decompose the objective:**
```python
# Decompose objective into sub-questions
from src.orchestrator.sub_objectives import ObjectiveTracker
self.objective_tracker = ObjectiveTracker(self.world_model)
await self.objective_tracker.decompose(objective)
print(f"📋 Decomposed objective into {len(self.objective_tracker.sub_objectives)} sub-questions:")
for i, so in enumerate(self.objective_tracker.sub_objectives, 1):
    print(f"   {i}. {so.question}")
```

**Replace `_check_objective_completion()` body** — keep the method signature but change the implementation:

Instead of the current LLM-based scoring + heuristics combo, use the ObjectiveTracker:
```python
def _check_objective_completion(self) -> bool:
    if self.objective_tracker is None:
        # Fallback to old behavior if no tracker
        return self._check_objective_completion_legacy()
    
    # Evaluate progress using sub-objectives
    import asyncio
    progress = asyncio.get_event_loop().run_until_complete(
        self.objective_tracker.evaluate_progress()
    )
    
    # Track cost
    self.total_budget_used += progress.get("cost", 0.0)
    
    # Log progress
    print(f"\n{'='*60}")
    print("OBJECTIVE PROGRESS CHECK")
    print(f"{'='*60}")
    print(f"  Score: {progress['score']:.1%}")
    print(f"  Answered: {progress['answered']}/{progress['total']}")
    print(f"  Partial: {progress['partial']}/{progress['total']}")
    print(f"  Unanswered: {progress['unanswered']}/{progress['total']}")
    for so in progress['sub_objectives']:
        status_icon = {"answered": "✅", "partial": "🔶", "unanswered": "❌"}.get(so['status'], "?")
        print(f"  {status_icon} {so['question'][:80]}")
        if so['answer_summary']:
            print(f"     → {so['answer_summary'][:100]}")
    print(f"{'='*60}\n")
    
    # Store for spawn decisions
    self.last_completion_score = progress['score']
    
    # Emit event
    self._emit_event(
        "progress_update",
        f"Progress: {progress['answered']}/{progress['total']} sub-objectives answered ({progress['score']:.0%})",
        progress
    )
    
    return self.objective_tracker.is_complete()
```

Rename the existing `_check_objective_completion` to `_check_objective_completion_legacy` so it's still available as fallback.

**In `_plan_initial_tasks()` prompt, add unanswered sub-questions for focus:**
If self.objective_tracker exists and has unanswered questions, append to the planning prompt:
```
Unanswered research questions (prioritize these):
{list of unanswered sub-questions}
```

**In `_should_trigger_synthesis()`, use the tracker score** instead of the old completion check when available.

### 3. Integrate into tree search (`src/orchestrator/tree_search.py`)

In `TreeSearchOrchestrator.run()`, create an ObjectiveTracker at the start and use it for convergence:
- After ideation, decompose the objective
- In `_check_convergence()`, also check `objective_tracker.is_complete()`
- In `_synthesize_results()`, include sub-objective progress in the final report

### 4. No changes needed to world model or backend API

Sub-objectives are tracked in memory by the ObjectiveTracker, not as graph nodes. This keeps it simple. The tracker state can be serialized via `to_dict()` / `from_dict()` if persistence is needed later.

## Commit

```bash
git add -A && git commit -m 'feat: crisper progress metrics via sub-objective decomposition

- New ObjectiveTracker decomposes objectives into measurable sub-questions
- LLM matches findings to sub-questions for deterministic progress scoring
- Progress = answered/total (no fuzzy LLM completion scoring)
- Integrated into cycle_manager for completion checks
- Integrated into tree_search for convergence detection
- Unanswered sub-questions fed into task planning for focus
- Old completion check preserved as _check_objective_completion_legacy'
```
