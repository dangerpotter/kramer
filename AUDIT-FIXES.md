# Code Audit Fixes — March 2026

Work through ALL 15 fixes below systematically. After all changes, stage and commit everything.

## 🔴 CRITICAL

### #1 - Create LLM client wrapper (src/utils/llm_client.py)
Every agent calls `anthropic.Anthropic(api_key=...)` directly. Create a wrapper that provides the same `.messages.create()` interface but supports routing through an OpenAI-compatible endpoint.

Create `src/utils/llm_client.py`:
- Class `LLMClient` with method `create_message(model, max_tokens, temperature, messages, thinking=None)` that returns an object matching Anthropic's response format (content blocks, usage)
- Check env var `LLM_BACKEND` — if "openai_compatible", use `httpx` to POST to `LLM_BASE_URL` (default `http://127.0.0.1:18789/v1/chat/completions`). Otherwise use `anthropic.Anthropic` directly.
- Add `tenacity` retry: retry on HTTP 429/500/502/503, exponential backoff, max 3 retries, starting 2s
- Factory function `get_llm_client()` that returns a singleton

Then update these files to use `get_llm_client()` instead of raw `anthropic.Anthropic()`:
- `src/kramer/data_analysis_agent.py`
- `src/kramer/hypothesis_agent.py`
- `src/kramer/hypothesis_tester_agent.py`
- `src/orchestrator/cycle_manager.py` (all Claude calls in _plan_initial_tasks, _claude_plan_synthesis_tasks, _claude_assess_synthesis_value, _claude_generate_synthesis_objective, _assess_objective_with_llm)

### #2 - Update cost_tracker.py
- Add entries for `claude-sonnet-4-6`, `claude-opus-4-6`
- Add `DEFAULT_PRICING` fallback (use Sonnet pricing)
- Make `_get_model_pricing()` return DEFAULT_PRICING instead of None for unknown models
- Log a warning when using default pricing
- `calculate_cost()` should never raise ValueError — use default pricing

### #3 - Fix Chinese character attribute
In `src/kramer/data_analysis_agent.py`, rename `思考_content` to `thinking_content` in the `AnalysisStep` dataclass. Search entire codebase for any references.

### #4 - Strengthen hypothesis ID validation
In `src/orchestrator/cycle_manager.py`:
- In `_plan_initial_tasks()` prompt, add: `"CRITICAL: hypothesis_id values must be EXACT UUID strings copied verbatim from the 'Untested Hypotheses' list above. Do NOT use integers, abbreviations, or made-up IDs."`
- In `create_task()`, if task_type is TEST_HYPOTHESIS and context has hypothesis_id, validate it exists in self.world_model.graph. Log warning and skip if not found.

## 🟡 HIGH

### #5 - Retry/backoff
Already handled by the LLM wrapper in #1 (tenacity decorator).

### #6 - Replace naive _evaluate_finding_support()
In `src/kramer/hypothesis_tester_agent.py`, replace keyword matching with an LLM call:
- Use the LLM client from #1
- Prompt: "Given hypothesis: X, and finding: Y, does this finding support (true) or refute (false) the hypothesis? Return JSON: {\"supports\": bool, \"reasoning\": \"...\"}"
- Keep old keyword method as `_evaluate_finding_support_fallback()` used if LLM fails

### #7 - Synthesis writes findings back to WorldModel
In `src/orchestrator/cycle_manager.py`, in `_execute_task()` under the `SYNTHESIZE_FINDINGS` case, after generating the report, use the LLM to extract 3-5 key insights from the report content and add them as Finding nodes to `self.world_model` with `metadata={"source": "synthesis"}`.

### #8 - Create .env.example
Create `.env.example` at project root:
```
# Required
ANTHROPIC_API_KEY=your-key-here
CLAUDE_MODEL=claude-sonnet-4-6

# Optional - Literature search
SEMANTIC_SCHOLAR_API_KEY=
CORE_API_KEY=
NCBI_API_KEY=

# Optional - LLM routing (for OpenClaw gateway)
LLM_BACKEND=anthropic
LLM_BASE_URL=http://127.0.0.1:18789/v1
```

### #9 - Wire multi-source literature agent
In `src/orchestrator/agent_coordinator.py` `execute_literature_search()`:
- Try to import `from kramer.agents.literature import LiteratureAgent as MultiSourceLiteratureAgent`
- If available and has necessary API keys, use it instead of the basic one
- Fall back to `src/kramer/literature_agent.py` if import fails

### #10 - Cost tracker graceful failure
Already handled by #2 (default pricing fallback).

## 🟢 MEDIUM

### #11 + #15 - Deprecation notice for kramer/ package
- Create `kramer/DEPRECATION_NOTICE.md` explaining this is legacy code, use `src/` instead
- In `kramer/__init__.py`, add: `import warnings; warnings.warn("The top-level 'kramer' package is deprecated. Use 'src.kramer' instead.", DeprecationWarning, stacklevel=2)`

### #12 - Cap hypothesis test scheduling
In `cycle_manager.py` `_schedule_hypothesis_tests()`, before the loop add:
```python
remaining_capacity = cycle.max_tasks - len(cycle.tasks)
```
Break out of loop when `remaining_capacity` is exhausted.

### #13 - WorldModel size limits
In `src/world_model/graph.py` `WorldModel.__init__()`:
- Add `max_nodes: int = 10000` parameter
- In `_add_node()` (or wherever nodes are added), check `self.graph.number_of_nodes() >= self.max_nodes`
- If at capacity, call `self._prune_low_value_nodes()` which removes lowest-confidence nodes older than 24h with degree 0
- Log when pruning

### #14 - Extended thinking per-task configurable
In `src/orchestrator/agent_coordinator.py` `execute_data_analysis()`:
- Pass `task.context.get("use_extended_thinking", True)` to `AgentConfig.use_extended_thinking`
- Same for `execute_hypothesis_test()`

## COMMIT

After ALL changes:
```bash
git add -A && git commit -m 'fix: address all 15 code audit findings

- Add LLM client wrapper with gateway/direct support (#1)
- Update cost tracker with current models + fallback (#2, #10)  
- Fix Chinese character attribute name (#3)
- Strengthen hypothesis ID validation (#4)
- Add retry/backoff to LLM calls (#5)
- Replace naive keyword support eval with LLM (#6)
- Synthesis writes findings back to WorldModel (#7)
- Create .env.example (#8)
- Wire multi-source literature agent (#9)
- Add deprecation notice for legacy kramer/ package (#11, #15)
- Cap hypothesis test scheduling at max_tasks (#12)
- Add WorldModel size limits with pruning (#13)
- Make extended thinking per-task configurable (#14)'
```
