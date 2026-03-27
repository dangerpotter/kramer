"""
Main data analysis agent that orchestrates code generation, execution, and result parsing.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
from src.kramer.code_executor import CodeExecutor, ExecutionResult
from src.kramer.result_parser import ResultParser, AnalysisResults
from src.kramer.notebook_manager import NotebookManager
from src.utils.cost_tracker import CostTracker
from src.utils.llm_client import get_llm_client


@dataclass
class AgentConfig:
    """Configuration for the DataAnalysisAgent."""

    api_key: Optional[str] = None
    model: str = None
    max_tokens: int = 16000
    timeout: int = 300
    max_iterations: int = 5
    use_extended_thinking: bool = True
    temperature: float = 1.0
    max_attempts_per_step: int = 3
    quality_threshold: float = 0.7
    step_timeout: int = 120
    use_code_evolution: bool = True  # Use evolving codebase instead of fresh code each step


@dataclass
class AnalysisStep:
    """A single step in the analysis trajectory."""

    step_number: int
    description: str
    code: str
    execution_result: ExecutionResult
    parsed_results: AnalysisResults
    thinking_content: Optional[str] = None  # Extended thinking content
    quality_score: float = 0.0
    attempt_number: int = 1
    evaluation_feedback: Optional[str] = None


class DataAnalysisAgent:
    """
    AI-powered data analysis agent.

    Uses Claude API with extended thinking to generate analysis code,
    executes it safely, and extracts structured findings.

    Features:
    - Autonomous code generation with extended thinking
    - Safe sandboxed execution
    - Automatic result parsing
    - Jupyter notebook creation
    - World model integration
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        notebooks_dir: Path = Path("outputs/notebooks"),
        plots_dir: Path = Path("outputs/plots"),
    ):
        """
        Initialize the data analysis agent.

        Args:
            config: Agent configuration
            notebooks_dir: Directory for saving notebooks
            plots_dir: Directory for saving plots
        """

        self.config = config or AgentConfig()

        # Get model from config or environment
        if not self.config.model:
            self.config.model = os.getenv("CLAUDE_MODEL")

        self.client = get_llm_client()

        # Initialize components
        self.executor = CodeExecutor(
            timeout=self.config.timeout,
            plots_dir=plots_dir,
        )
        self.parser = ResultParser()
        self.notebook_manager = NotebookManager(notebooks_dir=notebooks_dir)

        # Analysis state
        self.current_trajectory: List[AnalysisStep] = []
        self.world_model_context: Dict[str, Any] = {}
        self.total_cost: float = 0.0  # Track total API costs

        # Code evolution
        from src.kramer.analysis_codebase import AnalysisCodebase
        self.codebase: Optional[AnalysisCodebase] = None

    def analyze(
        self,
        objective: str,
        dataset_path: str,
        world_model_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform autonomous data analysis.

        Args:
            objective: Research objective / question to answer
            dataset_path: Path to dataset (CSV, Excel, etc.)
            world_model_context: Optional context from world model

        Returns:
            Dictionary with:
                - notebook_path: Path to generated notebook
                - findings: List of structured findings
                - steps: Analysis trajectory
                - world_model_updates: Updates to add to world model
        """

        self.world_model_context = world_model_context or {}
        self.current_trajectory = []

        # Validate dataset exists
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        # Initialize notebook
        notebook = self.notebook_manager.create_notebook(
            objective=objective,
            dataset_path=dataset_path,
            metadata={
                "model": self.config.model,
                "use_extended_thinking": self.config.use_extended_thinking,
            },
        )

        # Initialize code evolution if enabled
        if self.config.use_code_evolution:
            from src.kramer.analysis_codebase import AnalysisCodebase
            self.codebase = AnalysisCodebase(dataset_path=dataset_path)

        # Perform iterative analysis
        for iteration in range(self.config.max_iterations):
            step_num = iteration + 1
            step_start_time = time.time()
            best_step = None

            for attempt in range(self.config.max_attempts_per_step):
                # Check step time budget
                elapsed = time.time() - step_start_time
                if elapsed >= self.config.step_timeout and best_step is not None:
                    break

                # Generate code (fresh for attempt 0, refinement for subsequent)
                if attempt == 0:
                    if self.codebase is not None:
                        code = self.codebase.propose_modification(
                            step_objective=f"Step {step_num}: {objective}",
                            model=self.config.model,
                            previous_output=self._get_last_output(),
                        )
                        thinking = None  # code evolution doesn't use extended thinking for generation
                    else:
                        code, thinking = self._generate_analysis_code(
                            objective=objective,
                            dataset_path=dataset_path,
                            step_number=step_num,
                        )
                else:
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
                execution_result = self.executor.execute(
                    code=code,
                    context={"dataset_path": dataset_path},
                    capture_plots=True,
                )

                # Parse results
                parsed_results = self.parser.parse(
                    execution_result=execution_result,
                    code=code,
                )

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

            # Update code evolution codebase
            if self.codebase is not None and best_step is not None:
                if best_step.execution_result.success:
                    self.codebase.accept(
                        code=best_step.code,
                        score=best_step.quality_score,
                        objective=f"Step {step_num}",
                    )
                else:
                    self.codebase.reject()

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
                error_note = f"""
### ⚠️ Execution Error

The previous code encountered an error. This may require:
- Debugging the code
- Adjusting the approach
- Using alternative methods

Error: `{best_step.execution_result.error}`
"""
                self.notebook_manager.add_markdown_cell(notebook, error_note)
                break

        # Collect all findings
        all_findings = []
        for step in self.current_trajectory:
            all_findings.extend(step.parsed_results.findings)

        # Add findings summary to notebook
        if all_findings:
            self.notebook_manager.add_findings_summary(notebook, all_findings)

        # Save notebook
        notebook_path = self.notebook_manager.save_notebook(
            notebook,
            name=f"analysis_{objective[:30].replace(' ', '_')}.ipynb",
        )

        # Extract world model updates
        world_model_updates = []
        for step in self.current_trajectory:
            updates = self.parser.extract_world_model_updates(
                results=step.parsed_results,
                objective=objective,
            )
            world_model_updates.extend(updates)

        return {
            "notebook_path": str(notebook_path),
            "findings": [f.to_dict() for f in all_findings],
            "steps": len(self.current_trajectory),
            "world_model_updates": world_model_updates,
            "success": all(
                step.execution_result.success for step in self.current_trajectory
            ),
            "cost": self.total_cost,
            "final_script": self.codebase.current_script if self.codebase else None,
            "script_version": self.codebase.version if self.codebase else 0,
            "script_evolution": self.codebase.to_dict() if self.codebase else None,
        }

    def _generate_analysis_code(
        self,
        objective: str,
        dataset_path: str,
        step_number: int,
    ) -> tuple[str, Optional[str]]:
        """
        Generate analysis code using Claude API.

        Args:
            objective: Research objective
            dataset_path: Path to dataset
            step_number: Current step number

        Returns:
            Tuple of (generated_code, thinking_content)
        """

        # Build context from previous steps
        context = self._build_context()

        # Create prompt
        prompt = self._create_analysis_prompt(
            objective=objective,
            dataset_path=dataset_path,
            step_number=step_number,
            context=context,
        )

        # Call Claude API
        try:
            kwargs = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

            # Add extended thinking if enabled
            if self.config.use_extended_thinking:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 10000,
                }

            response = self.client.create_message(**kwargs)

            # Track API cost
            cost = CostTracker.track_call(self.config.model, response)
            self.total_cost += cost

            # Extract code and thinking
            code = ""
            thinking = None

            for block in response.content:
                if block.type == "thinking":
                    thinking = block.thinking
                elif block.type == "text":
                    # Extract code from markdown code blocks
                    code = self._extract_code_from_response(block.text)

            return code, thinking

        except Exception as e:
            print(f"Error generating code: {e}")
            return "", None

    def _evaluate_analysis_quality(
        self,
        objective: str,
        code: str,
        execution_result: ExecutionResult,
        parsed_results: AnalysisResults,
    ) -> Dict[str, Any]:
        """
        Use LLM to score an analysis step 0.0-1.0.

        Returns:
            Dict with score, issues, suggestions, findings_quality, should_retry.
        """
        output = execution_result.stdout if execution_result.success else execution_result.error
        prompt = f"""You are reviewing a data analysis step. Score it 0.0-1.0.

Research objective: {objective}
Code executed:
```python
{code}
```

Execution output:
{output}

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
{{"score": 0.0, "issues": ["..."], "suggestions": ["..."], "findings_quality": "none|weak|moderate|strong", "should_retry": true}}"""

        try:
            response = self.client.create_message(
                model=self.config.model,
                max_tokens=1024,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

            cost = CostTracker.track_call(self.config.model, response)
            self.total_cost += cost

            text = ""
            for block in response.content:
                if block.type == "text":
                    text = block.text
                    break

            return json.loads(text)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error evaluating analysis quality: {e}")
            return {
                "score": 0.5,
                "issues": ["Evaluation failed"],
                "suggestions": [],
                "findings_quality": "unknown",
                "should_retry": False,
            }

    def _generate_refinement_code(
        self,
        objective: str,
        dataset_path: str,
        step_number: int,
        previous_code: str,
        previous_output: str,
        evaluation_feedback: Dict[str, Any],
    ) -> tuple[str, Optional[str]]:
        """
        Generate improved analysis code based on feedback from a previous attempt.

        Returns:
            Tuple of (generated_code, thinking_content)
        """
        context = self._build_context()
        score = evaluation_feedback.get("score", 0.0)
        issues = evaluation_feedback.get("issues", [])
        suggestions = evaluation_feedback.get("suggestions", [])

        prompt = f"""You are a data analysis expert. Your previous attempt scored {score:.2f}. Issues: {json.dumps(issues)}. Suggestions: {json.dumps(suggestions)}. Write improved code that addresses these problems.

**Objective:** {objective}
**Dataset:** {dataset_path}
**Current Step:** {step_number} of {self.config.max_iterations}

{context}

**Previous code:**
```python
{previous_code}
```

**Previous output:**
{previous_output}

**Instructions:**
1. Fix all issues identified in the evaluation
2. Apply the suggestions for improvement
3. Use pandas for data manipulation, matplotlib/seaborn for visualization
4. Print key findings using clear print statements
5. Focus on statistical rigor and clear communication of results
6. The dataset will be available at the path: {dataset_path}

**Code Requirements:**
- Start by loading the dataset: `df = pd.read_csv('{dataset_path}')`
- Handle errors gracefully

**Output Format:**
Provide ONLY the Python code in a markdown code block. No explanations before or after.

```python
# Your improved code here
```
"""

        try:
            kwargs = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

            if self.config.use_extended_thinking:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 10000,
                }

            response = self.client.create_message(**kwargs)

            cost = CostTracker.track_call(self.config.model, response)
            self.total_cost += cost

            code = ""
            thinking = None

            for block in response.content:
                if block.type == "thinking":
                    thinking = block.thinking
                elif block.type == "text":
                    code = self._extract_code_from_response(block.text)

            return code, thinking

        except Exception as e:
            print(f"Error generating refinement code: {e}")
            return "", None

    def _create_analysis_prompt(
        self,
        objective: str,
        dataset_path: str,
        step_number: int,
        context: str,
    ) -> str:
        """Create the analysis prompt for Claude."""

        prompt = f"""You are a data analysis expert. Your task is to analyze a dataset to answer a research question.

**Objective:** {objective}

**Dataset:** {dataset_path}

**Current Step:** {step_number} of {self.config.max_iterations}

{context}

**Instructions:**
1. Write Python code to analyze the dataset and make progress toward the objective
2. Use pandas for data manipulation, matplotlib/seaborn for visualization
3. Print key findings using clear print statements (e.g., "Mean: 5.2", "P-value: 0.03")
4. Generate informative visualizations when appropriate
5. Focus on statistical rigor and clear communication of results
6. The dataset will be available at the path: {dataset_path}

**Code Requirements:**
- Start by loading the dataset: `df = pd.read_csv('{dataset_path}')`
- Include exploratory analysis (shape, dtypes, missing values) in early steps
- Print descriptive statistics and findings
- Create visualizations with clear titles and labels
- Use plt.savefig() or plt.show() to save plots
- Handle errors gracefully

**Output Format:**
Provide ONLY the Python code in a markdown code block. No explanations before or after.

If you believe the analysis is complete and no further steps are needed, respond with an empty code block.

```python
# Your code here
```
"""

        return prompt

    def _build_context(self) -> str:
        """Build context from previous analysis steps."""

        if not self.current_trajectory:
            return "**Previous Steps:** None (this is the first step)"

        context_parts = ["**Previous Steps:**\n"]

        for step in self.current_trajectory:
            context_parts.append(f"\n### Step {step.step_number}: {step.description}")

            # Add key findings
            if step.parsed_results.findings:
                context_parts.append("\nFindings:")
                for finding in step.parsed_results.findings[:3]:  # Limit to avoid token bloat
                    context_parts.append(f"- {finding.description}")

            # Add success/failure
            if not step.execution_result.success:
                context_parts.append(f"\n⚠️ Failed: {step.execution_result.error}")

        return "\n".join(context_parts)

    def _get_last_output(self) -> str:
        """Get stdout from the last successful step."""
        for step in reversed(self.current_trajectory):
            if step.execution_result.success:
                return step.execution_result.stdout[:2000]
        return ""

    def _extract_code_from_response(self, text: str) -> str:
        """Extract Python code from markdown code blocks."""

        import re

        # Look for ```python code blocks
        pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Try generic code blocks
        pattern = r"```\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Return as-is if no code blocks found
        return text.strip()

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Get the current analysis trajectory."""

        return [
            {
                "step_number": step.step_number,
                "description": step.description,
                "code": step.code,
                "success": step.execution_result.success,
                "findings": [f.to_dict() for f in step.parsed_results.findings],
                "execution_time": step.execution_result.execution_time,
            }
            for step in self.current_trajectory
        ]

    def save_trajectory(self, path: Path) -> None:
        """Save the analysis trajectory to JSON."""

        trajectory_data = {
            "trajectory": self.get_trajectory(),
            "config": {
                "model": self.config.model,
                "max_iterations": self.config.max_iterations,
                "use_extended_thinking": self.config.use_extended_thinking,
            },
        }

        with open(path, "w") as f:
            json.dump(trajectory_data, f, indent=2)
