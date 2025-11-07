"""
Kramer - AI-powered data analysis agent with autonomous research capabilities.
"""

# Note: Imports use different paths depending on context
# The actual modules are in src/kramer/ but may be imported as kramer/ from old code
try:
    from kramer.data_analysis_agent import DataAnalysisAgent
    from kramer.code_executor import CodeExecutor
    from kramer.result_parser import ResultParser
    from kramer.notebook_manager import NotebookManager
except ModuleNotFoundError:
    # Fallback for when running from src/ directory
    pass

__version__ = "0.1.0"

__all__ = [
    "DataAnalysisAgent",
    "CodeExecutor",
    "ResultParser",
    "NotebookManager",
]
