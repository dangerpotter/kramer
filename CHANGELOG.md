# Changelog

All notable changes to the Kramer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-06

### Added - Phase 2: Data Analysis Agent

#### Core Components
- **DataAnalysisAgent**: Main agent class that orchestrates autonomous data analysis
  - Integration with Claude API using extended thinking
  - Multi-step iterative analysis
  - World model context integration
  - Trajectory tracking and saving

- **CodeExecutor**: Safe code execution system
  - Subprocess-based sandboxing
  - Configurable timeout (default 300s)
  - Automatic plot capture using matplotlib
  - Error handling with retries
  - Context variable injection

- **ResultParser**: Intelligent result extraction
  - Statistical value extraction (mean, median, std, p-values, correlations)
  - Insight and conclusion detection
  - Plot tracking with descriptions
  - Code provenance for all findings
  - World model update generation

- **NotebookManager**: Jupyter notebook creation and management
  - Automated notebook generation
  - Cell creation with execution results
  - Embedded visualizations
  - Metadata tracking
  - Trajectory-to-notebook conversion

#### Testing
- Comprehensive test suite (23 unit tests)
  - CodeExecutor tests (7 tests)
  - ResultParser tests (7 tests)
  - NotebookManager tests (9 tests)
  - Integration tests (with API key requirement)
- Sample dataset for testing
- Test fixtures and utilities

#### Documentation
- Comprehensive README with:
  - Quick start guide
  - Architecture overview
  - Configuration options
  - Usage examples
  - Testing instructions
- Example scripts:
  - Basic usage example
  - Advanced usage with world model context
- Code documentation and docstrings

#### Project Structure
- Modern Python package structure with pyproject.toml
- Development environment setup
- Git ignore configuration
- Requirements specification

### Features Delivered (Phase 2 Acceptance Criteria)
- ✅ Can analyze CSV files
- ✅ Can generate plots automatically
- ✅ Can extract statistical findings
- ✅ All results have code provenance
- ✅ Safe sandboxed execution
- ✅ Structured findings with metadata
- ✅ World model integration
- ✅ Jupyter notebook output

### Technical Details
- Python 3.9+ support
- Claude Sonnet 4 integration
- Extended thinking capability
- Automatic plot capture (matplotlib/seaborn)
- JSON-based trajectory persistence
- Comprehensive error handling

## [Unreleased]

### Planned for Future Releases

#### Phase 3: Multi-Trajectory Planning
- Parallel trajectory exploration
- Trajectory comparison and selection
- Confidence scoring

#### Phase 4: World Model Persistence
- Database integration
- Knowledge graph construction
- Semantic search over findings

#### Phase 5: Multi-Agent Collaboration
- Agent coordination
- Specialized sub-agents
- Consensus mechanisms

### Future Enhancements
- Support for Excel, Parquet, SQL databases
- Interactive refinement loop
- Automated report generation
- MLflow/W&B integration
- Web UI for results visualization
- Real-time progress streaming
