# Kramer

A comprehensive AI-powered research discovery platform that autonomously runs discovery cycles, generates and tests hypotheses, integrates scientific literature, and maintains a knowledge graph with full provenance tracking.

## Overview

Kramer is an autonomous research system that uses Claude AI to conduct iterative discovery cycles. It combines:
- **Data Analysis**: Autonomous generation, execution, and analysis of Python code for data science tasks
- **Literature Integration**: Multi-source academic paper discovery (Semantic Scholar, arXiv, OpenAlex, PubMed, CORE)
- **Hypothesis Generation & Testing**: Automated hypothesis generation, novelty detection, and validation
- **Knowledge Graph**: Graph-based world model with full provenance tracking and relationship mapping
- **Real-time Web Interface**: Interactive dashboard, graph visualization, and reporting
- **Report Generation**: Multiple report types (summary, detailed, executive) with citations

## Features

### Discovery System
- **Autonomous Discovery Cycles**: Multi-cycle discovery framework with exploration, synthesis, and pivot cycles
- **Budget Management**: Total budget tracking and per-cycle budget allocation
- **Multiple Sessions**: Create and manage multiple independent discovery sessions
- **Real-time Monitoring**: WebSocket-powered live updates on discovery progress
- **Checkpointing**: Automatic checkpoint system to save progress at configurable intervals
- **Parallel Execution**: Run multiple tasks concurrently within each cycle

### AI & Analysis
- **Extended Thinking**: Uses Claude's extended thinking capability for deeper analytical reasoning
- **Safe Code Execution**: Sandboxed subprocess execution with timeout and error handling
- **Multi-Model Support**: Choose from available Claude models (Opus, Sonnet)
- **Cost Tracking**: Comprehensive cost tracking per cycle and cumulative budget consumption

### Knowledge Management
- **World Model Graph**: NetworkX graph + SQLite for persistent knowledge storage
- **Node Types**: Findings, hypotheses, questions, datasets, and papers
- **Edge Types**: Supports, refutes, derives_from, relates_to relationships
- **Confidence Scoring**: Confidence levels on all nodes for reliability assessment
- **Full Provenance**: All findings linked to source code, papers, and execution metadata

### Literature Integration
- **Multi-Source Search**: Parallel search across 5 academic databases:
  - **Semantic Scholar**: 200M+ papers with citation metrics
  - **arXiv**: Preprint server for physics, math, CS, and more
  - **OpenAlex**: 240M+ works, open catalog replacing Microsoft Academic
  - **PubMed**: 36M+ biomedical and life sciences papers
  - **CORE**: 200M+ open access papers with full-text availability
- **Smart Deduplication**: DOI-based deduplication across sources
- **RAG Engine**: ChromaDB-based vector embeddings for paper chunks
- **Full-Text Processing**: PDF download, extraction, and semantic indexing
- **Claim Extraction**: AI-powered extraction of claims from papers
- **Citation Management**: Automatic bibliography generation with citations

### Hypothesis System
- **Automatic Generation**: AI-powered hypothesis generation from findings
- **Novelty Detection**: Embedding-based novelty detection for new hypotheses
- **Automated Testing**: Hypothesis validation through data analysis
- **Status Tracking**: Track hypothesis states (untested, testing, supported, refuted)

## Web Interface

Kramer includes a full React-based web interface with the following pages:

### Configure Page (`/configure`)
Create new discovery sessions with:
- Objective specification
- Model selection (dynamically loaded from Anthropic API)
- Dataset path configuration
- Cycle count and budget parameters
- Parallel task settings
- Checkpoint intervals

### Discovery History (`/history`)
- View all past and current discoveries
- Search by objective
- Filter by status (running, completed, failed, stopped)
- Quick access to any discovery session

### Dashboard (`/dashboard/:discoveryId`)
Real-time discovery monitoring:
- Cost and budget visualization
- Cycle timeline with performance metrics
- Task breakdown and distribution
- WebSocket connection status
- Stop discovery controls

### Explorer (`/explorer/:discoveryId`)
Browse discovery results:
- All findings with confidence scores
- Hypotheses with test status
- Papers discovered
- Filter by confidence level
- Source badges (literature, data analysis, hypothesis test)
- Expandable details with full metadata

### World Model View (`/world-model/:discoveryId`)
Interactive knowledge graph:
- Cytoscape-based graph visualization
- Node detail panel with relationships
- Legend panel for node/edge types
- Zoom, fit, and layout controls
- Progress overview statistics
- Relationship analysis (supports, refutes counts)
- Confidence distribution charts

### Reports (`/reports/:discoveryId`)
Report generation and viewing:
- Generate summary, detailed, or executive reports
- Per-cycle reports for LLM context
- Markdown rendering with citations
- Download and delete reports
- Configurable confidence thresholds

## Quick Start

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

### Starting the Application

```bash
# Start the backend API server
cd backend
uvicorn main:app --reload --port 8000

# In another terminal, start the frontend
cd frontend
npm install
npm run dev
```

Access the web interface at `http://localhost:5173`

### Programmatic Usage

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

## REST API

### Discovery Endpoints (`/api/v1/discovery`)
- `POST /start` - Create and start new discovery
- `GET /{discovery_id}/status` - Get discovery status
- `POST /{discovery_id}/stop` - Stop running discovery
- `GET /{discovery_id}/cycles` - Get all cycles for discovery
- `GET /{discovery_id}/metrics` - Get real-time metrics
- `GET /` - List all discoveries

### World Model Endpoints (`/api/v1/world-model`)
- `GET /{discovery_id}/graph` - Get complete graph data with filtering
- `GET /{discovery_id}/nodes/{node_id}` - Get detailed node information
- `GET /{discovery_id}/findings` - Get findings with confidence filtering
- `GET /{discovery_id}/hypotheses` - Get hypotheses
- `GET /{discovery_id}/papers` - Get all papers discovered

### Dataset Endpoints (`/api/v1/datasets`)
- `POST /upload` - Upload dataset file
- `GET /` - List uploaded datasets
- `DELETE /{filename}` - Delete dataset

### Report Endpoints (`/api/v1/reports`)
- `GET /{discovery_id}` - List reports for discovery
- `POST /{discovery_id}/generate` - Generate new report
- `GET /{discovery_id}/{report_id}` - Get report content
- `DELETE /{discovery_id}/{report_id}` - Delete report
- `GET /{discovery_id}/cycle-reports` - List cycle reports

### WebSocket (`/api/v1/ws`)
- `GET /ws/{discovery_id}` - Real-time discovery updates

## Architecture

### Core Components

1. **World Model** (`src/world_model/graph.py`)
   - NetworkX graph for knowledge representation
   - SQLite persistence
   - Node types: findings, hypotheses, questions, datasets, papers
   - Edge types: supports, refutes, derives_from, relates_to

2. **Orchestrator** (`src/orchestrator/`)
   - CycleManager for discovery orchestration
   - AgentCoordinator for multi-agent task management
   - Budget enforcement and cycle management
   - Exploration, synthesis, and pivot cycles

3. **Agents** (`src/agents/`)
   - **DataAnalysisAgent**: Code generation and execution
   - **LiteratureAgent**: Paper search and claim extraction
   - **HypothesisAgent**: Hypothesis generation with novelty detection
   - **HypothesisTesterAgent**: Automated hypothesis validation

4. **Code Executor** (`src/kramer/code_executor.py`)
   - Safe subprocess-based execution
   - Timeout handling (configurable, default 300s)
   - Automatic plot capture

5. **RAG Engine** (`src/rag/`)
   - ChromaDB vector store for paper embeddings
   - Semantic search across paper chunks
   - Embedding-based novelty detection

6. **Backend API** (`backend/`)
   - FastAPI web framework
   - WebSocket manager for real-time updates
   - Discovery service for session management
   - Persistence service for SQLite storage

7. **Frontend** (`frontend/`)
   - React + TypeScript
   - Cytoscape for graph visualization
   - Recharts for metrics visualization
   - React Query for server state management

### Directory Structure

```
kramer/
├── backend/
│   ├── main.py                 # FastAPI application entry
│   ├── api/                    # REST API routes
│   │   └── v1/
│   │       ├── discovery.py
│   │       ├── world_model.py
│   │       ├── datasets.py
│   │       └── reports.py
│   ├── services/               # Business logic
│   │   ├── discovery_service.py
│   │   ├── world_model_service.py
│   │   └── persistence_service.py
│   └── bridge.py               # Orchestrator integration
├── frontend/
│   ├── src/
│   │   ├── pages/              # Route components
│   │   ├── components/         # Reusable UI components
│   │   ├── hooks/              # Custom React hooks
│   │   └── api/                # API client functions
│   └── package.json
├── src/
│   ├── world_model/            # Graph-based knowledge store
│   ├── orchestrator/           # Discovery cycle management
│   ├── agents/                 # AI agent implementations
│   ├── rag/                    # RAG engine with ChromaDB
│   ├── reporting/              # Report generation
│   └── kramer/                 # Data analysis components
│       ├── data_analysis_agent.py
│       ├── code_executor.py
│       ├── result_parser.py
│       └── notebook_manager.py
├── kramer/                     # Additional agent implementations
│   ├── agents/
│   │   └── literature.py       # Multi-source literature agent
│   └── api_clients/
│       ├── semantic_scholar.py # Semantic Scholar API
│       ├── openalex.py         # OpenAlex API
│       ├── pubmed.py           # PubMed E-utilities API
│       └── core.py             # CORE API
├── tests/                      # Comprehensive test suite
├── examples/                   # Usage examples
├── data/                       # Sample datasets
└── outputs/                    # Generated reports and notebooks
```

## Development

### Testing

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

Integration tests require an `ANTHROPIC_API_KEY` environment variable:

```bash
export ANTHROPIC_API_KEY='your-key-here'
pytest tests/test_integration.py
```

### Code Quality

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
black src/ kramer/ tests/ backend/

# Lint
ruff check src/ kramer/ tests/ backend/
```

## Configuration

### Environment Variables

Create a `.env` file in the project root (see `.env.example`):

```bash
# Required
ANTHROPIC_API_KEY=your_anthropic_key

# Literature Sources (optional but recommended)
SEMANTIC_SCHOLAR_API_KEY=your_s2_key      # Higher rate limits
CORE_API_KEY=your_core_key                # Required for CORE (free at https://core.ac.uk/services/api)
NCBI_API_KEY=your_ncbi_key                # Optional, for higher PubMed rate limits

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname
```

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

### Discovery Configuration

When creating a discovery via the API or UI:
- **objective**: Research question or goal
- **model**: Claude model to use
- **dataset_path**: Path to data file for analysis
- **num_cycles**: Number of discovery cycles to run
- **total_budget**: Maximum budget in dollars
- **max_parallel_tasks**: Tasks to run concurrently per cycle
- **checkpoint_interval**: How often to save progress (cycles)

## Use Cases

### 1. Exploratory Data Analysis
```python
result = agent.analyze(
    objective="Perform comprehensive EDA on this dataset",
    dataset_path="data.csv",
)
```

### 2. Hypothesis-Driven Research
```python
# Start a discovery that generates and tests hypotheses
from src.orchestrator.cycle_manager import Orchestrator

orchestrator = Orchestrator(world_model)
await orchestrator.spawn_cycle(
    objective="Identify factors affecting customer churn",
    max_tasks=10
)
```

### 3. Literature Review
```python
import asyncio
from kramer.agents.literature import LiteratureAgent
from kramer.world_model import WorldModel

async def search_literature():
    world_model = WorldModel()

    async with LiteratureAgent(
        world_model=world_model,
        anthropic_api_key="your-key",
        core_api_key="your-core-key",  # Enables CORE source
        sources=['semantic_scholar', 'openalex', 'pubmed', 'core']
    ) as agent:
        results = await agent.search_and_extract(
            query="machine learning for climate prediction",
            max_papers=20
        )
        print(f"Found {len(results['papers'])} papers from {results['sources_searched']}")

asyncio.run(search_literature())
```

### 4. Full Discovery Pipeline
Use the web interface to:
1. Configure a new discovery with your objective and dataset
2. Monitor progress in real-time on the dashboard
3. Explore findings, hypotheses, and papers in the Explorer
4. Visualize the knowledge graph in World Model view
5. Generate reports with citations

## Security

- Code execution is isolated in subprocess
- Timeout limits prevent infinite loops
- No network access from executed code (by default)
- All code is logged with provenance
- Errors are caught and reported safely
- Budget limits prevent runaway costs

## Troubleshooting

### Docker Disk Space Issues (Windows/WSL2)

On Windows with Docker Desktop using WSL2, the virtual disk file (`docker_data.vhdx`) can grow very large and **never automatically shrinks**, even when you delete data inside Docker. This can consume hundreds of gigabytes of disk space.

**Symptoms:**
- C: drive running out of space
- `C:\Users\<username>\AppData\Local\Docker\wsl\disk\docker_data.vhdx` is very large
- `docker system df` shows much less usage than the VHDX file size

**Prevention - Add to `~/.wslconfig`:**
```ini
[wsl2]
memory=8GB
swap=0
processors=4

[experimental]
sparseVhd=true
autoMemoryReclaim=gradual
```

**Recovery - Reclaim disk space:**
```powershell
# 1. Clean up Docker (with Docker running)
docker system prune -a --volumes -f
docker builder prune -a -f

# 2. Trim free space inside the VM
wsl -d docker-desktop -e fstrim /mnt/docker-desktop-disk

# 3. Quit Docker Desktop completely (right-click tray icon -> Quit)

# 4. Compact the VHDX (run in elevated PowerShell)
Optimize-VHD -Path "C:\Users\<username>\AppData\Local\Docker\wsl\disk\docker_data.vhdx" -Mode Full
```

**Note:** The build cache alone can consume 8+ GB. Running `docker builder prune -a -f` periodically helps prevent bloat.

## Roadmap

- [ ] Multi-trajectory planning and execution
- [ ] Support for more data formats (Excel, Parquet, SQL)
- [ ] Interactive refinement based on user feedback
- [ ] Integration with experiment tracking (MLflow, Weights & Biases)
- [ ] Export to publication formats (LaTeX, PDF)
- [ ] Collaborative multi-user sessions

## License

See [LICENSE](LICENSE) file for details.

## Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
