# Phase 3: Literature Agent - Complete Documentation

## Overview

The Literature Agent is a complete system for searching scientific literature and extracting key claims using AI. It integrates:

- **Semantic Scholar API** for paper search and metadata retrieval
- **Claude (Anthropic)** for intelligent claim extraction from abstracts
- **World Model** knowledge graph for storing papers, claims, and their relationships

## Architecture

```
┌─────────────────────┐
│  LiteratureAgent    │
│                     │
│  ┌───────────────┐  │
│  │ Semantic      │  │  Search papers
│  │ Scholar API   │◄─┼──────────────
│  └───────────────┘  │
│                     │
│  ┌───────────────┐  │  Extract claims
│  │ Claude API    │◄─┼──────────────
│  └───────────────┘  │
│                     │
│  ┌───────────────┐  │  Store results
│  │ World Model   │◄─┼──────────────
│  └───────────────┘  │
└─────────────────────┘
```

## Components

### 1. World Model (`kramer/world_model.py`)

The knowledge graph that stores:

- **Nodes**: Papers, Claims, Hypotheses, Concepts
- **Edges**: Relationships (supports, contradicts, cites, etc.)
- **Findings**: Extracted claims with source provenance

```python
from kramer.world_model import WorldModel, NodeType

world = WorldModel()

# Papers stored as nodes
paper_node = world.create_paper_node(
    title="Example Paper",
    authors=["Alice", "Bob"],
    year=2023,
    doi="10.1234/example"
)

# Claims stored as findings + claim nodes
finding = Finding(
    claim_text="Key finding from paper",
    source_node_id=paper_node.id,
    confidence=0.9
)
world.add_finding(finding)

# Automatically creates:
# - Claim node
# - Edge from claim to paper (DERIVED_FROM)
```

### 2. Semantic Scholar Client (`kramer/api_clients/semantic_scholar.py`)

Handles all interactions with Semantic Scholar API:

- Paper search with query strings
- Metadata retrieval (title, authors, year, DOI, abstract)
- Citation information
- Rate limiting and retry logic

```python
from kramer.api_clients.semantic_scholar import SemanticScholarClient

async with SemanticScholarClient() as client:
    papers = await client.search_papers(
        query="DNA repair mechanisms",
        limit=10
    )

    for paper in papers:
        print(f"{paper.title} ({paper.year})")
        print(f"Authors: {paper.authors}")
        print(f"Citations: {paper.citation_count}")
```

**Features:**
- Automatic retry on network errors (exponential backoff)
- Rate limiting (100ms delay between requests)
- Handles missing fields gracefully
- No API key required (optional for higher limits)

### 3. Literature Agent (`kramer/agents/literature.py`)

The main agent that orchestrates the entire workflow:

```python
from kramer.agents.literature import LiteratureAgent
from kramer.world_model import WorldModel

world = WorldModel()

async with LiteratureAgent(
    world_model=world,
    anthropic_api_key="your-key-here"
) as agent:

    claims = await agent.search_and_extract(
        query="nucleotide salvage pathways",
        max_papers=10
    )

    for claim in claims:
        print(f"Claim: {claim.claim_text}")
        print(f"Confidence: {claim.confidence}")
        print(f"Citation: {agent.format_citation(claim)}")
```

**Workflow:**

1. Search Semantic Scholar for relevant papers
2. Filter papers with substantive abstracts
3. Add papers to world model as nodes
4. Use Claude to extract 3-5 key claims from each abstract
5. Store claims as findings with edges to source papers
6. Return structured claims with full citations

## ExtractedClaim Structure

Each claim includes:

```python
@dataclass
class ExtractedClaim:
    claim_text: str          # The factual claim
    paper_title: str         # Source paper title
    authors: List[str]       # Paper authors
    year: Optional[int]      # Publication year
    doi: Optional[str]       # Digital Object Identifier
    confidence: float        # How central to paper (0-1)
    paper_id: str           # Semantic Scholar ID
```

## Claim Extraction Process

Claude is prompted to:

1. Read the paper abstract
2. Identify 3-5 key factual claims
3. Focus on concrete findings (not background)
4. Assign confidence scores (0.0-1.0) based on prominence
5. Format claims as clear, self-contained statements

**Prompt Engineering:**
- Temperature = 0 (deterministic)
- Structured output format
- Context-aware (considers current hypotheses if provided)
- Filters out speculation and background information

## Installation & Setup

### Using pip:

```bash
pip install -r requirements.txt
```

### Using Poetry (recommended):

```bash
poetry install
```

### Environment Setup:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Running Tests

All components have comprehensive test coverage (32 tests):

```bash
pytest tests/ -v

# With coverage
pytest tests/ --cov=kramer --cov-report=html
```

**Test Coverage:**
- World model: nodes, edges, findings, graph operations
- Semantic Scholar: API parsing, error handling, retries
- Literature Agent: claim extraction, citation formatting, integration

## Usage Example

### Basic Usage:

```python
import asyncio
from kramer.world_model import WorldModel
from kramer.agents.literature import LiteratureAgent

async def search_literature():
    world = WorldModel()

    async with LiteratureAgent(
        world_model=world,
        anthropic_api_key="your-key"
    ) as agent:

        claims = await agent.search_and_extract(
            query="nucleotide salvage pathways in hypothermia",
            max_papers=10
        )

        print(f"Extracted {len(claims)} claims")
        print(f"World model: {world.summary()}")

asyncio.run(search_literature())
```

### Running the Example:

```bash
export ANTHROPIC_API_KEY="your-key"
python examples/literature_search.py
```

Expected output:
```
Initializing Literature Agent...
================================================================================
Query: nucleotide salvage pathways in hypothermia
================================================================================

Searching for papers on: nucleotide salvage pathways in hypothermia
Found 10 papers

Processing paper 1/10: [Paper Title]
  Extracted 4 claims

...

================================================================================
EXTRACTED CLAIMS
================================================================================

1. Nucleotide salvage pathways are critical during hypothermic conditions
   Confidence: 0.90
   Source: Smith, Jones, Brown (2023). Metabolic adaptations...

2. [Additional claims...]

================================================================================
WORLD MODEL SUMMARY
================================================================================
  total_nodes: 24
  total_edges: 20
  total_findings: 20
  papers: 10
  claims: 20
```

## Rate Limiting & Error Handling

### Semantic Scholar API:
- 100ms delay between requests
- Automatic retry on timeout/network errors (3 attempts)
- Exponential backoff (2s, 4s, 8s)
- Graceful handling of 404s and missing data

### Claude API:
- Standard Anthropic rate limits apply
- 500ms delay between papers to avoid rapid succession
- Errors logged but don't stop the entire batch

### Best Practices:
- Process papers in batches
- Use async/await for concurrent operations
- Monitor API usage in production
- Consider caching results for repeated queries

## Acceptance Criteria ✓

- [x] Can search for papers on a topic
- [x] Can extract 3-5 key claims from abstract
- [x] Citations are formatted correctly
- [x] Claims added to world model with paper provenance
- [x] Rate limiting implemented
- [x] Error handling for API failures
- [x] Comprehensive test coverage
- [x] Works with real queries

## Future Enhancements

Potential improvements for future phases:

1. **Full-text processing**: Extract claims from PDFs, not just abstracts
2. **Claim linking**: Identify when claims support/contradict each other
3. **Citation network**: Explore papers that cite the found papers
4. **Hypothesis refinement**: Use extracted claims to generate new hypotheses
5. **Multi-agent collaboration**: Integrate with other agents (reasoning, experimentation)
6. **Semantic search**: Use embeddings for better paper relevance
7. **Claim verification**: Cross-reference claims across multiple papers
8. **Interactive exploration**: Web UI for browsing the knowledge graph

## API Documentation

### Semantic Scholar API
- Documentation: https://api.semanticscholar.org/api-docs/
- No authentication required for basic usage
- Rate limits: Reasonable for academic use
- Coverage: 200M+ papers across all fields

### Anthropic Claude API
- Documentation: https://docs.anthropic.com/
- API key required
- Model used: `claude-3-5-sonnet-20241022`
- Best for: Structured extraction from scientific text

## License

See LICENSE file in repository root.
