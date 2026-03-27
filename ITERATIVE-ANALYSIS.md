# Iterative Analysis Loop Implementation

## Overview
Replace the single-shot analysis in DataAnalysisAgent with a nested loop that refines code within each step. Only the best attempt per step gets promoted to findings.

## Changes Required

### 1. `src/kramer/data_analysis_agent.py` — Main changes

**AgentConfig — add new fields:**
```python
max_attempts_per_step: int = 3      # refinement attempts per analysis step
quality_threshold: float = 0.7      # minimum score to accept without retry
step_timeout: int = 120             # seconds per step (all attempts share this)
```

**AnalysisStep dataclass — add fields:**
```python
quality_score: float = 0.0
attempt_number: int = 1
evaluation_feedback: Optional[str] = None
```

**New method `_evaluate_analysis_quality()`:**
- Takes: objective, code, execution_result, parsed_results
- Uses LLM client (from src/utils/llm_client.py, NOT raw anthropic) to score the analysis 0.0-1.0
- Returns dict: {"score": float, "issues": list[str], "suggestions": list[str], "findings_quality": str, "should_retry": bool}
- Prompt should evaluate: did it error? appropriate methods? answers the objective? sufficient sample size? obvious confounds?
- Track cost from this call

**New method `_generate_refinement_code()`:**
- Like _generate_analysis_code() but takes additional context: previous attempt's code, output, and evaluation feedback
- Prompt explicitly says: "Your previous attempt scored X. Issues: Y. Suggestions: Z. Write improved code that addresses these problems."
- Returns (code, thinking) tuple same as _generate_analysis_code()

**Modify `analyze()` method — nested loop:**
```python
for iteration in range(self.config.max_iterations):
    step_start_time = time.time()
    best_step = None
    
    for attempt in range(self.config.max_attempts_per_step):
        # Check step time budget
        elapsed = time.time() - step_start_time
        if elapsed >= self.config.step_timeout and best_step is not None:
            break
        
        # Generate code (fresh for attempt 0, refinement for subsequent)
        if attempt == 0:
            code, thinking = self._generate_analysis_code(
                objective=objective,
                dataset_path=dataset_path,
                step_number=step_num,
            )
        else:
            code, thinking = self._generate_refinement_code(
                objective=objective,
                dataset_path=dataset_path,
                step_number=step_num,
                previous_code=prev_code,
                previous_output=prev_output,
                evaluation_feedback=evaluation,
            )
        
        if not code or code.strip() == "":
            break  # Agent says analysis is complete
        
        # Execute
        execution_result = self.executor.execute(code=code, ...)
        parsed_results = self.parser.parse(execution_result=execution_result, code=code)
        
        # Evaluate quality
        evaluation = self._evaluate_analysis_quality(
            objective, code, execution_result, parsed_results
        )
        
        quality_score = evaluation.get("score", 0.0)
        
        # Create step with quality info
        step = AnalysisStep(
            step_number=step_num,
            description=f"Analysis Step {step_num} (attempt {attempt + 1})",
            code=code,
            execution_result=execution_result,
            parsed_results=parsed_results,
            thinking_content=thinking,
            quality_score=quality_score,
            attempt_number=attempt + 1,
            evaluation_feedback=json.dumps(evaluation),
        )
        
        # Keep best attempt
        if best_step is None or quality_score > best_step.quality_score:
            best_step = step
        
        # Good enough? Move to next analysis step
        if quality_score >= self.config.quality_threshold:
            break
        
        # Store for refinement context
        prev_code = code
        prev_output = execution_result.stdout if execution_result.success else execution_result.error
        
        # If evaluation says don't retry, stop
        if not evaluation.get("should_retry", True):
            break
    
    # Promote best attempt
    if best_step is None:
        break  # No code generated, analysis complete
    
    self.current_trajectory.append(best_step)
    
    # Add best attempt to notebook
    self.notebook_manager.add_code_cell(
        notebook=notebook,
        code=best_step.code,
        execution_result=best_step.execution_result,
        description=best_step.description,
    )
    
    # If best attempt still failed, add error note and stop
    if not best_step.execution_result.success:
        # ... existing error handling ...
        break
```

**Evaluation prompt template:**
```
You are reviewing a data analysis step. Score it 0.0-1.0.

Research objective: {objective}
Code executed:
```python
{code}
```

Execution output:
{stdout or error}

Score criteria:
- 0.0-0.3: Errors, wrong method, or meaningless output
- 0.3-0.5: Runs but methodology is questionable  
- 0.5-0.7: Decent analysis but could be improved
- 0.7-0.9: Good analysis, appropriate methods, clear findings
- 0.9-1.0: Excellent, publication-quality analysis

Evaluate:
1. Did the code execute without errors?
2. Are the statistical methods appropriate for this data?
3. Do the findings address the research objective?
4. Are there confounding variables or methodological issues?
5. Is the output interpretable and meaningful?

Return ONLY JSON:
{"score": 0.0, "issues": ["..."], "suggestions": ["..."], "findings_quality": "none|weak|moderate|strong", "should_retry": true}
```

### 2. `src/orchestrator/agent_coordinator.py` — Config passthrough

In `execute_data_analysis()`, pass task context config to AgentConfig:
```python
agent_config = AgentConfig(
    ...existing...,
    max_attempts_per_step=task.context.get("max_attempts_per_step", 3),
    quality_threshold=task.context.get("quality_threshold", 0.7),
    step_timeout=task.context.get("step_timeout", 120),
)
```

### 3. Import requirements
- `import time` in data_analysis_agent.py  
- Use `get_llm_client()` from `src/utils/llm_client` for the evaluation LLM call (NOT raw anthropic)

## Commit
```bash
git add -A && git commit -m 'feat: iterative analysis loop with quality-gated refinement

- Add per-step retry loop with LLM quality evaluation
- Best attempt per step promoted to findings (keep/discard pattern)
- New AgentConfig options: max_attempts_per_step, quality_threshold, step_timeout
- Evaluation scores methodology, error handling, objective relevance
- Refinement prompts include previous attempt feedback
- Per-step wall-clock budget prevents runaway refinement'
```
