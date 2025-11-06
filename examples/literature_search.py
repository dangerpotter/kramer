"""Example: Search literature and extract claims."""

import asyncio
import os
from kramer.world_model import WorldModel
from kramer.agents.literature import LiteratureAgent


async def main():
    """Run a literature search and extract claims."""

    # Get API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return

    # Initialize world model and agent
    print("Initializing Literature Agent...")
    world = WorldModel()

    async with LiteratureAgent(
        world_model=world,
        anthropic_api_key=api_key
    ) as agent:

        # Run the search
        query = "nucleotide salvage pathways in hypothermia"
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}\n")

        claims = await agent.search_and_extract(
            query=query,
            max_papers=10
        )

        # Display results
        print(f"\n{'='*80}")
        print("EXTRACTED CLAIMS")
        print(f"{'='*80}\n")

        for i, claim in enumerate(claims, 1):
            print(f"\n{i}. {claim.claim_text}")
            print(f"   Confidence: {claim.confidence:.2f}")
            print(f"   Source: {agent.format_citation(claim)}")

        # Display world model summary
        print(f"\n{'='*80}")
        print("WORLD MODEL SUMMARY")
        print(f"{'='*80}\n")

        summary = world.summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")

        # Show paper titles
        print(f"\n{'='*80}")
        print("PAPERS IN WORLD MODEL")
        print(f"{'='*80}\n")

        from kramer.world_model import NodeType
        papers = world.get_nodes_by_type(NodeType.PAPER)
        for i, paper in enumerate(papers, 1):
            metadata = paper.metadata
            authors_str = ", ".join(metadata.get("authors", [])[:2])
            if len(metadata.get("authors", [])) > 2:
                authors_str += " et al."

            print(f"{i}. {metadata.get('title', 'Unknown')}")
            print(f"   {authors_str} ({metadata.get('year', 'N/A')})")
            print(f"   Citations: {metadata.get('citation_count', 0)}")
            print(f"   Claims extracted: {len(world.get_findings_for_paper(paper.id))}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
