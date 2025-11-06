# Testing the Literature Agent

## Unit Tests

Run the comprehensive test suite (32 tests):

```bash
# Install dependencies first
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=kramer --cov-report=html
```

All tests should pass:
- 8 tests for Literature Agent
- 6 tests for Semantic Scholar client
- 18 tests for World Model

## Integration Test with Real Query

To test with the real Semantic Scholar API and Claude:

### 1. Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 2. Run the example script:

```bash
python examples/literature_search.py
```

This will:
1. Search Semantic Scholar for "nucleotide salvage pathways in hypothermia"
2. Find up to 10 relevant papers
3. Extract 3-5 claims from each paper using Claude
4. Display all claims with citations
5. Show world model statistics

### 3. Expected Output:

```
Initializing Literature Agent...
================================================================================
Query: nucleotide salvage pathways in hypothermia
================================================================================

Searching for papers on: nucleotide salvage pathways in hypothermia
Found 10 papers

Processing paper 1/10: [Title of first paper]
  Extracted 4 claims

Processing paper 2/10: [Title of second paper]
  Extracted 5 claims

... (continues for all papers)

✓ Total claims extracted: 42

================================================================================
EXTRACTED CLAIMS
================================================================================

1. Hypothermia alters nucleotide metabolism by reducing enzymatic activity
   Confidence: 0.90
   Source: Smith, Jones et al. (2023). Metabolic changes during hypothermia. DOI: 10.1234/example

2. Salvage pathways become more important than de novo synthesis in cold conditions
   Confidence: 0.85
   Source: Brown, Wilson (2022). Nucleotide recycling mechanisms. DOI: 10.5678/example

... (continues for all claims)

================================================================================
WORLD MODEL SUMMARY
================================================================================
  total_nodes: 52
  total_edges: 42
  total_findings: 42
  papers: 10
  claims: 42
  hypotheses: 0

================================================================================
PAPERS IN WORLD MODEL
================================================================================

1. Metabolic adaptations to hypothermia in mammalian cells
   Smith, Jones et al. (2023)
   Citations: 45
   Claims extracted: 4

2. Nucleotide salvage pathways under thermal stress
   Brown, Wilson (2022)
   Citations: 32
   Claims extracted: 5

... (continues for all papers)
```

## Custom Query Test

You can also test with your own query:

```python
import asyncio
import os
from kramer.world_model import WorldModel
from kramer.agents.literature import LiteratureAgent

async def test_custom_query():
    world = WorldModel()

    async with LiteratureAgent(
        world_model=world,
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"]
    ) as agent:

        # Your custom query
        claims = await agent.search_and_extract(
            query="your research question here",
            max_papers=5
        )

        for claim in claims:
            print(f"{claim.claim_text}")
            print(f"  → {agent.format_citation(claim)}")
            print()

asyncio.run(test_custom_query())
```

## What Gets Tested

### Unit Tests (Mocked):
- ✓ World model graph operations
- ✓ Node and edge creation
- ✓ Finding storage and retrieval
- ✓ Paper metadata parsing
- ✓ Claim extraction parsing
- ✓ Citation formatting
- ✓ Error handling

### Integration Test (Real APIs):
- ✓ Semantic Scholar paper search
- ✓ Abstract retrieval
- ✓ Claude claim extraction
- ✓ World model integration
- ✓ End-to-end workflow
- ✓ Rate limiting behavior
- ✓ Real scientific query handling

## Troubleshooting

### "No module named anthropic"
```bash
pip install anthropic
```

### "ANTHROPIC_API_KEY environment variable not set"
```bash
export ANTHROPIC_API_KEY="your-key"
```

### Rate limit errors
- Add delays between requests
- Use the built-in rate limiting (already implemented)
- For Semantic Scholar: 100ms between requests (implemented)
- For Claude: 500ms between papers (implemented)

### No papers found
- Check your query string
- Try broader search terms
- Verify Semantic Scholar API is accessible
- Check network connectivity

## Performance Notes

For the test query "nucleotide salvage pathways in hypothermia":

- **Search time**: ~1-2 seconds
- **Processing time**: ~30-60 seconds for 10 papers
  - Depends on Claude API latency (~3-5s per paper)
- **Total claims**: Typically 30-50 claims from 10 papers
- **World model size**: ~50 nodes, ~40 edges

## API Costs

Using Claude Sonnet 4:
- Cost per paper: ~$0.001-0.002 (abstract extraction)
- Cost for 10 papers: ~$0.01-0.02
- Very affordable for research purposes

Semantic Scholar API:
- **Free** with no authentication required
- Reasonable rate limits for academic use
