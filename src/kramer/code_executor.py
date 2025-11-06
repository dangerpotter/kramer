"""
Safe code execution module for running Python code with sandboxing and timeout.
"""

import subprocess
import tempfile
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import sys


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    stdout: str
    stderr: str
    return_value: Any = None
    execution_time: float = 0.0
    plots: List[Path] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.plots is None:
            self.plots = []


class CodeExecutor:
    """
    Executes Python code safely with timeout and sandboxing.

    Features:
    - Subprocess isolation
    - Configurable timeout
    - Automatic plot capture
    - Result serialization
    - Error handling
    """

    def __init__(
        self,
        timeout: int = 300,
        plots_dir: Path = Path("outputs/plots"),
        max_retries: int = 3,
    ):
        """
        Initialize the code executor.

        Args:
            timeout: Maximum execution time in seconds
            plots_dir: Directory to save plots
            max_retries: Maximum number of retries on transient failures
        """
        self.timeout = timeout
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries

    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        capture_plots: bool = True,
    ) -> ExecutionResult:
        """
        Execute Python code safely.

        Args:
            code: Python code to execute
            context: Optional context variables to pass to the code
            capture_plots: Whether to automatically capture matplotlib plots

        Returns:
            ExecutionResult with output, errors, and captured plots
        """
        # Create temporary directory for this execution
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create execution script
            script_path = temp_path / "execute.py"
            result_path = temp_path / "result.json"
            plots_path = temp_path / "plots"
            plots_path.mkdir(exist_ok=True)

            # Prepare the execution script
            exec_script = self._prepare_execution_script(
                code=code,
                context=context or {},
                result_path=result_path,
                plots_path=plots_path,
                capture_plots=capture_plots,
            )

            script_path.write_text(exec_script)

            # Execute with retries
            for attempt in range(self.max_retries):
                try:
                    result = self._run_subprocess(script_path, temp_path)

                    # Load result metadata if available
                    if result_path.exists():
                        metadata = json.loads(result_path.read_text())
                        result.return_value = metadata.get("return_value")
                        result.execution_time = metadata.get("execution_time", 0.0)

                    # Copy plots to persistent storage
                    saved_plots = self._save_plots(plots_path)
                    result.plots = saved_plots

                    return result

                except subprocess.TimeoutExpired:
                    error_msg = f"Execution timed out after {self.timeout} seconds"
                    return ExecutionResult(
                        success=False,
                        stdout="",
                        stderr=error_msg,
                        error=error_msg,
                    )

                except Exception as e:
                    if attempt == self.max_retries - 1:
                        error_msg = f"Execution failed: {str(e)}"
                        return ExecutionResult(
                            success=False,
                            stdout="",
                            stderr=error_msg,
                            error=error_msg,
                        )
                    time.sleep(2 ** attempt)  # Exponential backoff

            # Should not reach here, but return error just in case
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="Max retries exceeded",
                error="Max retries exceeded",
            )

    def _prepare_execution_script(
        self,
        code: str,
        context: Dict[str, Any],
        result_path: Path,
        plots_path: Path,
        capture_plots: bool,
    ) -> str:
        """Prepare the execution script with context and result capture."""

        # Serialize context
        context_json = json.dumps(context, default=str)

        # Escape code for embedding in script
        escaped_code = code.replace("'", "\\'")

        script = f"""
import sys
import json
import time
import traceback
from pathlib import Path

# Setup
result_path = Path("{result_path}")
plots_path = Path("{plots_path}")
context = json.loads('''{context_json}''')

# Inject context into namespace
globals().update(context)

# Setup plot capture
if {capture_plots}:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Override savefig to save to our plots directory
    _original_savefig = plt.savefig
    _plot_counter = [0]

    def _custom_savefig(*args, **kwargs):
        if args and not str(args[0]).startswith(str(plots_path)):
            plot_file = plots_path / f"plot_{{_plot_counter[0]:03d}}.png"
            _plot_counter[0] += 1
            _original_savefig(plot_file, *args[1:], **kwargs)
        else:
            _original_savefig(*args, **kwargs)

    plt.savefig = _custom_savefig

    # Auto-save on show
    _original_show = plt.show
    def _custom_show(*args, **kwargs):
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            plot_file = plots_path / f"plot_{{_plot_counter[0]:03d}}.png"
            _plot_counter[0] += 1
            fig.savefig(plot_file, bbox_inches='tight', dpi=300)
        plt.close('all')

    plt.show = _custom_show

# Execute user code
start_time = time.time()
return_value = None
try:
    # Execute the code
    exec_globals = {{}}
    exec_globals.update(globals())
    exec('''{escaped_code}''', exec_globals)

    # Try to capture any return value from last expression
    # This is a simple heuristic and may not work for all cases

    execution_time = time.time() - start_time

    # Save result metadata
    result_data = {{
        "success": True,
        "execution_time": execution_time,
        "return_value": None,  # Could enhance this to capture final values
    }}
    result_path.write_text(json.dumps(result_data))

    # Save any remaining plots
    if {capture_plots}:
        try:
            if plt.get_fignums():
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    plot_file = plots_path / f"plot_{{_plot_counter[0]:03d}}.png"
                    _plot_counter[0] += 1
                    fig.savefig(plot_file, bbox_inches='tight', dpi=300)
                plt.close('all')
        except:
            pass

except Exception as e:
    execution_time = time.time() - start_time
    print(f"ERROR: {{str(e)}}", file=sys.stderr)
    traceback.print_exc()

    result_data = {{
        "success": False,
        "execution_time": execution_time,
        "error": str(e),
        "traceback": traceback.format_exc(),
    }}
    result_path.write_text(json.dumps(result_data))
    sys.exit(1)
"""
        return script

    def _run_subprocess(self, script_path: Path, cwd: Path) -> ExecutionResult:
        """Run the script in a subprocess."""

        start_time = time.time()

        process = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=str(cwd),
        )

        execution_time = time.time() - start_time

        success = process.returncode == 0

        return ExecutionResult(
            success=success,
            stdout=process.stdout,
            stderr=process.stderr,
            execution_time=execution_time,
            error=None if success else process.stderr,
        )

    def _save_plots(self, plots_path: Path) -> List[Path]:
        """Copy plots from temporary directory to persistent storage."""

        saved_plots = []

        if not plots_path.exists():
            return saved_plots

        # Find all plot files
        plot_files = sorted(plots_path.glob("plot_*.png"))

        # Generate timestamp for this execution
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Copy to persistent storage
        for i, plot_file in enumerate(plot_files):
            target_file = self.plots_dir / f"{timestamp}_{i:03d}.png"
            target_file.write_bytes(plot_file.read_bytes())
            saved_plots.append(target_file)

        return saved_plots

    def execute_cell(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute a single notebook cell.

        This is a convenience method that wraps execute() with
        settings optimized for notebook cell execution.

        Args:
            code: Cell code to execute
            context: Optional context from previous cells

        Returns:
            ExecutionResult
        """
        return self.execute(
            code=code,
            context=context,
            capture_plots=True,
        )
