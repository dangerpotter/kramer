# Kramer - AI-Powered Data Analysis Agent

Kramer is an autonomous data analysis agent that uses Claude AI with extended thinking to generate, execute, and analyze Python code for data science tasks. It creates complete Jupyter notebooks with code, visualizations, and structured findings - all with full provenance tracking.

## 🌟 Features

- **Autonomous Analysis**: Generates multi-step analysis plans and executes them independently
- **Extended Thinking**: Uses Claude's extended thinking capability for deeper analytical reasoning
- **Safe Execution**: Sandboxed code execution with timeout and error handling
- **Structured Findings**: Automatically extracts statistics, insights, and visualizations
- **Jupyter Notebooks**: Creates publication-ready notebooks with all analysis steps
- **World Model Integration**: Tracks findings with full code provenance for knowledge building
- **Robust Error Handling**: Gracefully handles errors with detailed logging

## 📋 Phase 2 Requirements - ✅ Complete

This implementation fulfills all Phase 2 requirements:

### ✅ DataAnalysisAgent Class
- Uses Claude API with extended thinking
- Input: objective + dataset + world_model context
- Output: Jupyter notebook + structured findings

### ✅ Safe Code Execution
- Subprocess-based sandboxing
- Configurable timeout handling
- Automatic result capture (plots, statistics, dataframes)

### ✅ Result Parser
- Extracts quantitative findings (means, p-values, correlations)
- Extracts visualizations with descriptions
- Links all results to generating code (provenance)
- Adds findings to world model with metadata

### ✅ Acceptance Criteria
- ✅ Can analyze CSV files
- ✅ Can generate plots automatically
- ✅ Can extract statistical findings
- ✅ All results have code provenance

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd kramer

# Install dependencies
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"

# Set up your API key
export ANTHROPIC_API_KEY='your-api-key-here'
```

### Basic Usage

```python
from kramer import DataAnalysisAgent, AgentConfig

# Configure the agent
config = AgentConfig(
    model="claude-sonnet-4-20250514",
    max_iterations=5,
    use_extended_thinking=True,
)

# Initialize
agent = DataAnalysisAgent(config=config)

# Run analysis
result = agent.analyze(
    objective="Analyze customer satisfaction drivers",
    dataset_path="data/sample_data.csv",
)

# Access results
print(f"Notebook: {result['notebook_path']}")
print(f"Findings: {len(result['findings'])}")
print(f"Success: {result['success']}")
```

### Command Line

```bash
# Run the basic example
python examples/basic_usage.py

# Run the advanced example with world model context
python examples/advanced_usage.py
```

## 📚 Architecture

### Core Components

1. **DataAnalysisAgent** (`data_analysis_agent.py`)
   - Main orchestrator
   - Manages analysis trajectory
   - Integrates Claude API
   - Handles world model context

2. **CodeExecutor** (`code_executor.py`)
   - Safe subprocess-based execution
   - Timeout handling
   - Automatic plot capture
   - Result serialization

3. **ResultParser** (`result_parser.py`)
   - Extracts statistics from output
   - Identifies insights and patterns
   - Links findings to code provenance
   - Formats for world model

4. **NotebookManager** (`notebook_manager.py`)
   - Creates Jupyter notebooks
   - Adds cells with execution results
   - Embeds visualizations
   - Tracks metadata

### Data Flow

```
User Objective → DataAnalysisAgent
    ↓
Claude API (with extended thinking)
    ↓
Generated Python Code
    ↓
CodeExecutor (sandboxed subprocess)
    ↓
Execution Results + Plots
    ↓
ResultParser (extract findings)
    ↓
NotebookManager (create notebook)
    ↓
Structured Output + World Model Updates
```

## 🔧 Configuration

### AgentConfig Options

```python
AgentConfig(
    api_key: str = None,              # Claude API key (or use env var)
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 16000,          # Max tokens per API call
    timeout: int = 300,               # Max execution time per step (seconds)
    max_iterations: int = 5,          # Max analysis steps
    use_extended_thinking: bool = True,  # Enable extended thinking
    temperature: float = 1.0,         # Sampling temperature
)
```

### Directory Structure

```
kramer/
├── src/kramer/           # Source code
│   ├── data_analysis_agent.py
│   ├── code_executor.py
│   ├── result_parser.py
│   └── notebook_manager.py
├── tests/                # Test suite
│   ├── test_code_executor.py
│   ├── test_result_parser.py
│   ├── test_notebook_manager.py
│   └── test_integration.py
├── examples/             # Usage examples
│   ├── basic_usage.py
│   └── advanced_usage.py
├── data/                 # Sample datasets
│   └── sample_data.csv
└── outputs/              # Generated outputs
    ├── notebooks/
    └── plots/
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_code_executor.py

# Run with coverage
pytest --cov=kramer tests/

# Run only unit tests (skip integration tests that need API key)
pytest tests/ -k "not integration"
```

### Integration Tests

Integration tests require an `ANTHROPIC_API_KEY` environment variable:

```bash
export ANTHROPIC_API_KEY='your-key-here'
pytest tests/test_integration.py
```

## 📊 Example Output

### Generated Findings

```json
{
  "type": "statistic",
  "description": "mean: 52.3",
  "value": 52.3,
  "code_provenance": "df['income'].mean()",
  "metadata": {
    "stat_name": "mean",
    "context_line": "Mean income: 52.3"
  }
}
```

### World Model Updates

```json
{
  "type": "statistic",
  "content": "correlation: 0.85",
  "value": 0.85,
  "objective": "Analyze satisfaction drivers",
  "provenance": {
    "code": "correlation = df[['income', 'satisfaction']].corr()",
    "execution_time": 1.23
  }
}
```

## 🎯 Use Cases

### 1. Exploratory Data Analysis
```python
result = agent.analyze(
    objective="Perform comprehensive EDA on this dataset",
    dataset_path="data.csv",
)
```

### 2. Hypothesis Testing
```python
result = agent.analyze(
    objective="Test if income significantly affects satisfaction (p<0.05)",
    dataset_path="data.csv",
)
```

### 3. Correlation Analysis
```python
result = agent.analyze(
    objective="Find all significant correlations and create a correlation matrix heatmap",
    dataset_path="data.csv",
)
```

### 4. Segmentation Analysis
```python
result = agent.analyze(
    objective="Segment customers by purchase behavior and profile each segment",
    dataset_path="data.csv",
)
```

## 🔐 Security

- Code execution is isolated in subprocess
- Timeout limits prevent infinite loops
- No network access from executed code (by default)
- All code is logged with provenance
- Errors are caught and reported safely

## 🤝 Contributing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/

# Lint
ruff check src/ tests/
```

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🔮 Roadmap

- [ ] **Phase 3**: Multi-trajectory planning and execution
- [ ] **Phase 4**: World model persistence and querying
- [ ] **Phase 5**: Multi-agent collaboration
- [ ] Support for more data formats (Excel, Parquet, SQL)
- [ ] Interactive refinement based on user feedback
- [ ] Automated report generation
- [ ] Integration with experiment tracking (MLflow, Weights & Biases)

## 📚 Citation

If you use Kramer in your research, please cite:

```bibtex
@software{kramer2025,
  title={Kramer: AI-Powered Data Analysis Agent},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/kramer}
}
```

## 🙋 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Documentation**: See the `docs/` directory for detailed guides

## 🎓 Examples

See the `examples/` directory for:
- Basic usage patterns
- Advanced configurations
- Custom result processing
- World model integration
- Error handling strategies

---

**Built with ❤️ using Claude AI**
