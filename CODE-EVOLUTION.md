# Code Evolution for Analysis — Implementation Guide

## Overview
Maintain a "best analysis script" per dataset that accumulates proven code across steps. Each step builds on the previous best version instead of generating fresh code. Uses the existing iterative quality evaluation to decide keep/discard.

## New File: `src/kramer/analysis_codebase.py`

```python
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.llm_client import get_llm_client


@dataclass
class CodeVersion:
    """A version of the evolving analysis script."""
    version: int
    code: str
    score: float
    step_objective: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class AnalysisCodebase:
    """
    Evolving analysis script for a dataset.
    
    Instead of generating fresh code each step, the LLM modifies the current
    best script to address new objectives. Only improvements are kept.
    
    Usage:
        codebase = AnalysisCodebase(dataset_path="data.csv")
        # Step 1: fresh code (no existing script)
        new_code = codebase.propose_modification("Load and explore the dataset", model)
        codebase.accept(new_code, score=0.8, objective="Load and explore")
        # Step 2: modify existing script
        new_code = codebase.propose_modification("Add correlation analysis", model)
        if score > previous_score:
            codebase.accept(new_code, score=0.85, objective="Add correlation")
        else:
            codebase.reject()  # keep previous version
    """
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.current_script: str = ""
        self.version: int = 0
        self.history: List[CodeVersion] = []
        self.llm_client = get_llm_client()
    
    def propose_modification(
        self,
        step_objective: str,
        model: str,
        previous_output: str = "",
        evaluation_feedback: Optional[Dict] = None,
        is_refinement: bool = False,
    ) -> str:
        """
        Ask LLM to modify the current script to address a new objective.
        
        Args:
            step_objective: What this step should accomplish
            model: Model name for the LLM call
            previous_output: Output from the last execution (for context)
            evaluation_feedback: Feedback from quality evaluation (for refinements)
            is_refinement: If True, this is a retry of the same step
            
        Returns:
            Modified Python code string
        """
        if not self.current_script:
            # First step: generate fresh code
            return self._generate_initial_code(step_objective, model)
        
        if is_refinement and evaluation_feedback:
            return self._generate_refinement(step_objective, model, previous_output, evaluation_feedback)
        
        return self._generate_evolution(step_objective, model, previous_output)
    
    def _generate_initial_code(self, objective: str, model: str) -> str:
        """Generate the first version of the analysis script."""
        prompt = f"""Write a Python data analysis script for the following objective.

**Objective:** {objective}
**Dataset:** {self.dataset_path}

Requirements:
- Load the dataset with pandas
- Include clear print statements for all findings
- Generate visualizations where appropriate
- Use proper statistical methods
- The script should be self-contained and complete

Return ONLY the Python code in a ```python block."""

        response = self.llm_client.create_message(
            model=model,
            max_tokens=4000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        
        return self._extract_code(response)
    
    def _generate_evolution(self, objective: str, model: str, previous_output: str) -> str:
        """Modify existing script to add new analysis."""
        prompt = f"""You have an existing data analysis script. Modify it to ALSO address a new objective.

**New objective to add:** {objective}
**Dataset:** {self.dataset_path}

**Current script (version {self.version}):**
```python
{self.current_script}
```

**Output from current script:**
{previous_output[:2000] if previous_output else "Not available"}

Instructions:
- KEEP all existing analysis code that works
- ADD new code sections to address the new objective
- Organize the script logically (imports, data loading, analysis sections)
- Do NOT duplicate data loading if it's already in the script
- Add clear section comments (# === New: {objective} ===)
- Print findings for the new analysis clearly

Return ONLY the complete modified Python script in a ```python block.
Include ALL existing code plus your additions."""

        response = self.llm_client.create_message(
            model=model,
            max_tokens=8000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        
        return self._extract_code(response)
    
    def _generate_refinement(
        self,
        objective: str,
        model: str,
        previous_output: str,
        evaluation_feedback: Dict,
    ) -> str:
        """Refine the script based on quality evaluation feedback."""
        issues = evaluation_feedback.get("issues", [])
        suggestions = evaluation_feedback.get("suggestions", [])
        score = evaluation_feedback.get("score", 0.0)
        
        prompt = f"""Your analysis script needs improvement. Fix the issues identified.

**Objective:** {objective}
**Dataset:** {self.dataset_path}

**Current script:**
```python
{self.current_script}
```

**Output from execution:**
{previous_output[:2000] if previous_output else "Not available"}

**Quality score:** {score:.2f}/1.0

**Issues found:**
{chr(10).join(f'- {issue}' for issue in issues) if issues else 'None specified'}

**Suggestions:**
{chr(10).join(f'- {s}' for s in suggestions) if suggestions else 'None specified'}

Fix the identified issues and apply the suggestions. Keep everything that works.
Return ONLY the complete fixed Python script in a ```python block."""

        response = self.llm_client.create_message(
            model=model,
            max_tokens=8000,
            temperature=0.5,
            messages=[{"role": "user", "content": prompt}],
        )
        
        return self._extract_code(response)
    
    def accept(self, code: str, score: float, objective: str) -> None:
        """Accept a code modification as the new best version."""
        self.version += 1
        self.current_script = code
        self.history.append(CodeVersion(
            version=self.version,
            code=code,
            score=score,
            step_objective=objective,
        ))
    
    def reject(self) -> None:
        """Reject a modification, keeping the current version."""
        pass  # current_script stays unchanged
    
    def get_current_score(self) -> float:
        """Get the score of the current version."""
        if self.history:
            return self.history[-1].score
        return 0.0
    
    def _extract_code(self, response) -> str:
        """Extract Python code from LLM response."""
        import re
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text
        
        pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        pattern = r"```\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        return text.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "current_script": self.current_script,
            "version": self.version,
            "history": [
                {"version": v.version, "score": v.score, "step_objective": v.step_objective, "timestamp": v.timestamp}
                for v in self.history
            ],
        }
```

## Integration into `src/kramer/data_analysis_agent.py`

### Add to AgentConfig:
```python
use_code_evolution: bool = True  # Use evolving codebase instead of fresh code each step
```

### In DataAnalysisAgent.__init__():
```python
from src.kramer.analysis_codebase import AnalysisCodebase
self.codebase: Optional[AnalysisCodebase] = None
```

### Modify the `analyze()` method:

At the start, before the iteration loop, initialize the codebase if enabled:
```python
if self.config.use_code_evolution:
    from src.kramer.analysis_codebase import AnalysisCodebase
    self.codebase = AnalysisCodebase(dataset_path=dataset_path)
```

Inside the nested loop, when generating code:

**For the outer step loop (attempt == 0 and first try):**
Instead of calling `self._generate_analysis_code()`, call:
```python
if self.codebase is not None:
    code = self.codebase.propose_modification(
        step_objective=f"Step {step_num}: {objective}",
        model=self.config.model,
        previous_output=self._get_last_output(),
    )
    thinking = None  # code evolution doesn't use extended thinking for generation
else:
    code, thinking = self._generate_analysis_code(...)
```

**For refinement attempts (attempt > 0):**
```python
if self.codebase is not None:
    code = self.codebase.propose_modification(
        step_objective=f"Step {step_num}: {objective}",
        model=self.config.model,
        previous_output=prev_output,
        evaluation_feedback=evaluation,
        is_refinement=True,
    )
    thinking = None
else:
    code, thinking = self._generate_refinement_code(...)
```

**After selecting the best step (when promoting):**
```python
if self.codebase is not None and best_step is not None:
    if best_step.execution_result.success:
        self.codebase.accept(
            code=best_step.code,
            score=best_step.quality_score,
            objective=f"Step {step_num}",
        )
    else:
        self.codebase.reject()
```

### Add helper method:
```python
def _get_last_output(self) -> str:
    """Get stdout from the last successful step."""
    for step in reversed(self.current_trajectory):
        if step.execution_result.success:
            return step.execution_result.stdout[:2000]
    return ""
```

### In the return dict of analyze(), add codebase info:
```python
result = {
    ...existing fields...,
    "final_script": self.codebase.current_script if self.codebase else None,
    "script_version": self.codebase.version if self.codebase else 0,
    "script_evolution": self.codebase.to_dict() if self.codebase else None,
}
```

## Integration into `src/orchestrator/agent_coordinator.py`

In `execute_data_analysis()`, pass the config:
```python
agent_config = AgentConfig(
    ...existing...,
    use_code_evolution=task.context.get("use_code_evolution", True),
)
```

In the result metadata, include script info:
```python
metadata={
    ...existing...,
    "final_script": result.get("final_script"),
    "script_version": result.get("script_version", 0),
}
```

## Commit

```bash
git add -A && git commit -m 'feat: code evolution for analysis — evolving best script across steps

- New AnalysisCodebase maintains a best analysis script per dataset
- LLM modifies existing script to add new analysis (not fresh each step)
- Keep/discard pattern: only improvements are accepted
- Refinement mode for quality-gated retries uses evaluation feedback
- Final evolved script included in analysis results as deliverable
- Configurable via use_code_evolution in AgentConfig (default: True)'
```
