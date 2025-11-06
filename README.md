# Kramer

A pragmatic implementation of autonomous discovery cycles for scientific research with AI-powered data analysis.

## Overview

Kramer is an autonomous research system that uses Claude AI to run discovery cycles. It combines:
- **Data Analysis**: Autonomous generation, execution, and analysis of Python code for data science tasks
- **Literature Review**: Semantic Scholar integration for research paper discovery
- **Hypothesis Generation**: Automated hypothesis generation from findings
- **Knowledge Management**: Graph-based world model with full provenance tracking
- **Report Generation**: Markdown reports with citations

## 🌟 Features

- **Extended Thinking**: Uses Claude's extended thinking capability for deeper analytical reasoning
- **Safe Code Execution**: Sandboxed code execution with timeout and error handling
- **Structured Findings**: Automatically extracts statistics, insights, and visualizations
- **Jupyter Notebooks**: Creates publication-ready notebooks with all analysis steps
- **World Model**: NetworkX graph + SQLite for persistent knowledge storage
- **Literature Integration**: Semantic Scholar API integration for research papers
- **Full Provenance**: All findings linked to source code and execution metadata
- **Robust Error Handling**: Graceful error handling with detailed logging

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd kramer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Set up your API key
export ANTHROPIC_API_KEY='your-api-key-here'
```

### Basic Usage

```python
from kramer import DataAnalysisAgent, AgentConfig
from src.world_model.graph import WorldModel

# Configure the agent
config = AgentConfig(
    model="claude-sonnet-4-20250514",
    max_iterations=5,
    use_extended_thinking=True,
)

# Initialize
agent = DataAnalysisAgent(config=config)
world_model = WorldModel()

# Run analysis
result = agent.analyze(
    objective="Analyze customer satisfaction drivers",
    dataset_path="data/sample_data.csv",
)

# Access results
print(f"Notebook: {result['notebook_path']}")
print(f"Findings: {len(result['findings'])}")

# Add findings to world model
for finding in result['findings']:
    world_model.add_finding(
        text=finding['description'],
        code_link=finding.get('code_provenance', ''),
        confidence=finding.get('value', 0.5),
    )
```

### Command Line

```bash
# Run basic example
python examples/basic_usage.py

# Run advanced example with world model
python examples/advanced_usage.py

# Run literature search
python examples/literature_search.py
```

## 📚 Architecture

### Core Components

1. **World Model** (`src/world_model/graph.py`)
   - NetworkX graph for knowledge representation
   - SQLite persistence
   - Node types: findings, hypotheses, questions
   - Edge types: supports, contradicts, related_to

2. **Orchestrator** (`src/orchestrator/cycle_manager.py`)
   - Async task queue for cycle management
   - Task spawning and prioritization
   - Budget and resource management

3. **Data Analysis Agent** (`src/kramer/data_analysis_agent.py`)
   - Claude API integration with extended thinking
   - Multi-step analysis planning
   - Jupyter notebook generation

4. **Code Executor** (`src/kramer/code_executor.py`)
   - Safe subprocess-based execution
   - Timeout handling
   - Automatic plot capture

5. **Result Parser** (`src/kramer/result_parser.py`)
   - Statistics extraction
   - Insight identification
   - Provenance linking

6. **Literature Agent** (`kramer/agents/literature.py`)
   - Semantic Scholar integration
   - Paper search and retrieval
   - Citation management

### Directory Structure

```
kramer/
├── src/
│   ├── world_model/        # Graph-based knowledge store
│   ├── orchestrator/       # Task spawning and cycle management
│   ├── agents/             # Agent implementations
│   ├── reporting/          # Report generation
│   └── kramer/             # Data analysis components
│       ├── data_analysis_agent.py
│       ├── code_executor.py
│       ├── result_parser.py
│       └── notebook_manager.py
├── kramer/                 # Additional agent implementations
│   ├── agents/
│   │   └── literature.py
│   ├── api_clients/
│   │   └── semantic_scholar.py
│   └── world_model.py
├── tests/                  # Comprehensive test suite
├── examples/               # Usage examples
├── data/                   # Sample datasets
└── outputs/                # Generated reports and notebooks
```

## 🎯 Development Phases

### ✅ Phase 1: Core Infrastructure (Complete)
- [x] Project structure
- [x] WorldModel with NetworkX + SQLite
- [x] Basic orchestrator with async task queue
- [x] Comprehensive test suite

### ✅ Phase 2: Data Analysis Agent (Complete)
- [x] DataAnalysisAgent with extended thinking
- [x] Safe code execution with sandboxing
- [x] Result parser with provenance tracking
- [x] Jupyter notebook generation
- [x] Statistical finding extraction

### ✅ Phase 3: Literature Agent (Complete)
- [x] Semantic Scholar API integration
- [x] Paper search and retrieval
- [x] Citation management
- [x] Literature findings integration with world model

### ✅ Phase 4: Discovery Loop (Complete)
- [x] CycleManager for discovery orchestration
- [x] Priority queue for task management
- [x] Integration of data and literature agents
- [x] End-to-end discovery cycles

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_code_executor.py

# Run with coverage
pytest --cov=kramer --cov=src tests/

# Run only unit tests (skip integration tests that need API key)
pytest tests/ -k "not integration"
```

### Integration Tests

Integration tests require an `ANTHROPIC_API_KEY` environment variable:

```bash
export ANTHROPIC_API_KEY='your-key-here'
pytest tests/test_integration.py
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

## 📊 Use Cases

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

### 3. Literature Review
```python
from kramer.agents.literature import LiteratureAgent

lit_agent = LiteratureAgent()
papers = lit_agent.search("machine learning for climate prediction")
```

### 4. Full Discovery Cycle
```python
from src.orchestrator.cycle_manager import Orchestrator

orchestrator = Orchestrator(world_model)
await orchestrator.spawn_cycle(
    objective="Analyze climate change trends and find supporting literature",
    max_tasks=10
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
black src/ kramer/ tests/

# Lint
ruff check src/ kramer/ tests/
```

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🔮 Roadmap

- [ ] Multi-trajectory planning and execution
- [ ] Real-time world model querying and updates
- [ ] Multi-agent collaboration
- [ ] Support for more data formats (Excel, Parquet, SQL)
- [ ] Interactive refinement based on user feedback
- [ ] Automated report generation with citations
- [ ] Integration with experiment tracking (MLflow, Weights & Biases)
- [ ] Web interface for exploration

## 📚 Documentation

See the following files for detailed documentation:
- `PHASE3_DOCUMENTATION.md` - Literature agent implementation details
- `TESTING.md` - Testing guidelines
- `CHANGELOG.md` - Version history
- `CONTRIBUTING.md` - Contribution guidelines

## 🙋 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions

---

**Built with ❤️ using Claude AI**
