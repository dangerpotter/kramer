"""
Main data analysis agent that orchestrates code generation, execution, and result parsing.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import anthropic

from kramer.code_executor import CodeExecutor, ExecutionResult
from kramer.result_parser import ResultParser, AnalysisResults
from kramer.notebook_manager import NotebookManager
from src.utils.cost_tracker import CostTracker


@dataclass
class AgentConfig:
    """Configuration for the DataAnalysisAgent."""

    api_key: Optional[str] = None
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 16000
    timeout: int = 300
    max_iterations: int = 5
    use_extended_thinking: bool = True
    temperature: float = 1.0


@dataclass
class AnalysisStep:
    """A single step in the analysis trajectory."""

    step_number: int
    description: str
    code: str
    execution_result: ExecutionResult
    parsed_results: AnalysisResults
    思考_content: Optional[str] = None  # Extended thinking content


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

        # Get API key from config or environment
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY must be provided in config or environment"
            )

        self.client = anthropic.Anthropic(api_key=api_key)

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

        # Perform iterative analysis
        for iteration in range(self.config.max_iterations):
            step_num = iteration + 1

            # Generate analysis code
            code, thinking = self._generate_analysis_code(
                objective=objective,
                dataset_path=dataset_path,
                step_number=step_num,
            )

            if not code or code.strip() == "":
                # Agent decided analysis is complete
                break

            # Execute code
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

            # Create analysis step
            step = AnalysisStep(
                step_number=step_num,
                description=f"Analysis Step {step_num}",
                code=code,
                execution_result=execution_result,
                parsed_results=parsed_results,
                思考_content=thinking,
            )
            self.current_trajectory.append(step)

            # Add to notebook
            self.notebook_manager.add_code_cell(
                notebook=notebook,
                code=code,
                execution_result=execution_result,
                description=step.description,
            )

            # Check if we should continue
            if not execution_result.success:
                # Add error analysis
                error_note = f"""
### ⚠️ Execution Error

The previous code encountered an error. This may require:
- Debugging the code
- Adjusting the approach
- Using alternative methods

Error: `{execution_result.error}`
"""
                self.notebook_manager.add_markdown_cell(notebook, error_note)
                # For now, stop on error. Could enhance to retry with error feedback.
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

            response = self.client.messages.create(**kwargs)

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
