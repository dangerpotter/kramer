# Kramer

Multi-agent system for scientific hypothesis exploration and literature analysis.

## Features

### Phase 3: Literature Agent

- Search scientific papers via Semantic Scholar API
- Extract key claims from paper abstracts using Claude
- Maintain citation provenance in world model
- Track papers as nodes and claims as findings

## Installation

```bash
poetry install
```

## Usage

```python
from kramer.agents.literature import LiteratureAgent
from kramer.world_model import WorldModel

# Initialize
world = WorldModel()
agent = LiteratureAgent(world, anthropic_api_key="your-key")

# Search and extract claims
claims = await agent.search_and_extract(
    query="nucleotide salvage pathways in hypothermia",
    max_papers=10
)
```

## Testing

```bash
poetry run pytest
```

## License

See LICENSE file.
