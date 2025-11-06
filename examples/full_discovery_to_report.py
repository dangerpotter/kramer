"""
Full Discovery to Report Pipeline Demo

This example demonstrates the complete workflow:
1. Create world model with discoveries
2. Run discovery cycles (simulated)
3. Generate publication-quality report
4. Show how to convert to PDF
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reporting.report_generator import ReportGenerator
from src.world_model.graph import EdgeType, NodeType, WorldModel


def simulate_discovery_cycle(wm: WorldModel) -> None:
    """
    Simulate a discovery cycle that populates the world model.

    In a real scenario, this would be the CycleManager running
    data analysis and literature agents.
    """
    print("\n" + "=" * 70)
    print("SIMULATING DISCOVERY CYCLE")
    print("=" * 70)

    # Cycle 1: Initial data exploration
    print("\nCycle 1: Initial Data Exploration")
    print("-" * 70)

    # Initial question
    q1 = wm.add_question(
        "What patterns exist in the multivariate dataset?",
        metadata={"cycle": 1, "priority": "high"},
    )
    print(f"  Added question: {q1[:8]}...")

    # Dataset
    ds1 = wm.add_dataset(
        "Multivariate dataset with 10,000 samples and 50 features",
        path="/data/multivariate_data.csv",
        metadata={"samples": 10000, "features": 50, "cycle": 1},
    )
    print(f"  Added dataset: {ds1[:8]}...")

    # Initial findings from data analysis
    f1 = wm.add_finding(
        "Features A and B show strong positive correlation (r=0.87, p<0.001)",
        code_link="notebooks/exploratory_analysis.ipynb:cell_12",
        confidence=0.93,
        metadata={"cycle": 1, "analysis_type": "correlation"},
    )
    print(f"  Found correlation: {f1[:8]}...")

    f2 = wm.add_finding(
        "Three distinct clusters identified in PCA space (silhouette=0.72)",
        code_link="notebooks/clustering_analysis.ipynb:cell_25",
        confidence=0.89,
        metadata={"cycle": 1, "analysis_type": "clustering", "n_clusters": 3},
    )
    print(f"  Found clusters: {f2[:8]}...")

    # Link findings
    wm.add_edge(ds1, f1, EdgeType.RELATES_TO)
    wm.add_edge(ds1, f2, EdgeType.RELATES_TO)
    wm.add_edge(q1, f1, EdgeType.RELATES_TO)

    # Cycle 2: Literature review
    print("\nCycle 2: Literature Review")
    print("-" * 70)

    # Search literature for context
    p1 = wm.add_paper(
        "Analysis of correlation patterns in high-dimensional data",
        title="Correlation Analysis in High-Dimensional Datasets",
        authors=["Zhang, Y.", "Chen, L.", "Wang, X."],
        year=2022,
        doi="10.1007/s10618-022-00123-4",
    )
    print(f"  Found paper: {p1[:8]}...")

    p2 = wm.add_paper(
        "Clustering methods achieve best results with silhouette > 0.6",
        title="Cluster Validation Techniques: A Survey",
        authors=["Rodriguez, M.", "Garcia, P."],
        year=2021,
        doi="10.1109/TKDE.2021.1234567",
    )
    print(f"  Found paper: {p2[:8]}...")

    # Link papers to findings
    wm.add_edge(p1, f1, EdgeType.SUPPORTS)
    wm.add_edge(p2, f2, EdgeType.SUPPORTS)

    # Cycle 3: Hypothesis generation and testing
    print("\nCycle 3: Hypothesis Testing")
    print("-" * 70)

    # Generate hypothesis
    h1 = wm.add_hypothesis(
        "The three clusters represent distinct underlying data generation processes",
        parent_finding=f2,
        confidence=0.78,
        metadata={"cycle": 3, "basis": "clustering_results"},
    )
    print(f"  Generated hypothesis: {h1[:8]}...")

    # Test hypothesis with new analysis
    f3 = wm.add_finding(
        "Cluster 1 shows higher variance in features C-E compared to other clusters (F=12.4, p<0.001)",
        code_link="notebooks/cluster_characterization.ipynb:cell_8",
        confidence=0.91,
        metadata={"cycle": 3, "analysis_type": "anova"},
    )
    print(f"  Found cluster differences: {f3[:8]}...")

    f4 = wm.add_finding(
        "Cluster 2 has significantly lower mean values across features F-J (t=-8.2, p<0.001)",
        code_link="notebooks/cluster_characterization.ipynb:cell_15",
        confidence=0.88,
        metadata={"cycle": 3, "analysis_type": "t_test"},
    )
    print(f"  Found cluster characteristics: {f4[:8]}...")

    # Link supporting evidence
    wm.add_edge(f3, h1, EdgeType.SUPPORTS)
    wm.add_edge(f4, h1, EdgeType.SUPPORTS)
    wm.add_edge(f2, f3, EdgeType.RELATES_TO)
    wm.add_edge(f2, f4, EdgeType.RELATES_TO)

    # Cycle 4: Novel discovery
    print("\nCycle 4: Novel Discovery")
    print("-" * 70)

    # New question emerges
    q2 = wm.add_question(
        "Do the clusters exhibit temporal patterns?",
        metadata={"cycle": 4, "priority": "high", "derived_from": h1},
    )
    print(f"  New question: {q2[:8]}...")

    # Novel finding
    f5 = wm.add_finding(
        "Cluster membership shows strong temporal autocorrelation (lag-1 r=0.73)",
        code_link="notebooks/temporal_analysis.ipynb:cell_22",
        confidence=0.85,
        metadata={"cycle": 4, "analysis_type": "time_series", "novel": True},
    )
    print(f"  Novel finding: {f5[:8]}...")

    # This finding answers the question and is novel
    wm.add_edge(q2, f5, EdgeType.RELATES_TO)
    wm.add_edge(f2, f5, EdgeType.SUPPORTS)

    # Check literature - no papers found on this specific aspect
    # (high novelty score due to no supporting literature)

    # New hypothesis from novel finding
    h2 = wm.add_hypothesis(
        "Cluster transitions follow a Markov process with state-dependent probabilities",
        parent_finding=f5,
        confidence=0.72,
        metadata={"cycle": 4, "novel": True},
    )
    print(f"  Novel hypothesis: {h2[:8]}...")

    # Supporting analysis
    f6 = wm.add_finding(
        "Markov model achieves 82% accuracy in predicting next cluster state",
        code_link="notebooks/markov_modeling.ipynb:cell_34",
        confidence=0.87,
        metadata={"cycle": 4, "model_type": "markov", "accuracy": 0.82},
    )
    print(f"  Model validation: {f6[:8]}...")

    wm.add_edge(f6, h2, EdgeType.SUPPORTS)

    print(f"\nDiscovery cycle complete!")
    print(f"World model now contains: {wm}")


def generate_comprehensive_report(wm: WorldModel, output_dir: Path) -> dict:
    """Generate a comprehensive research report."""
    print("\n" + "=" * 70)
    print("GENERATING RESEARCH REPORT")
    print("=" * 70)

    # Create report generator
    rg = ReportGenerator(
        world_model=wm,
        min_confidence=0.7,  # Include high-confidence findings
        max_discoveries=5,
    )

    # Extract and analyze findings
    print("\nAnalyzing world model...")
    findings = rg.extract_high_confidence_findings()
    print(f"  Extracted {len(findings)} high-confidence findings")

    # Show novelty distribution
    novelty_scores = [f["novelty_score"] for f in findings]
    avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0
    print(f"  Average novelty score: {avg_novelty:.2f}")

    high_novelty = [f for f in findings if f["novelty_score"] > 0.5]
    if high_novelty:
        print(f"  High novelty findings: {len(high_novelty)}")
        for f in high_novelty:
            print(f"    - {f['text'][:60]}... (novelty: {f['novelty_score']:.2f})")

    # Group into discoveries
    print("\nGrouping into discoveries...")
    discoveries = rg.group_findings_into_discoveries(findings)
    print(f"  Created {len(discoveries)} discoveries")

    for i, disc in enumerate(discoveries, 1):
        print(f"  Discovery {i}: {disc.title[:60]}...")
        print(f"    - Findings: {len(disc.findings)}")
        print(f"    - Papers: {len(disc.papers)}")
        print(f"    - Hypotheses: {len(disc.hypotheses)}")
        print(f"    - Confidence: {disc.confidence:.2f}")
        print(f"    - Novelty: {disc.novelty_score:.2f}")

    # Generate report
    print("\nGenerating markdown report...")
    output_path = output_dir / "comprehensive_report.md"

    result = rg.generate_report(
        output_path=output_path,
        include_appendix=True,
        generate_narratives=False,  # Set True if you have API key
    )

    print(f"  Main report: {result['report']}")
    print(f"  Appendix: {result.get('appendix', 'N/A')}")

    # Show report stats
    report_text = result['report'].read_text()
    num_lines = len(report_text.split('\n'))
    num_citations = report_text.count('[Analysis')
    num_papers = report_text.count('[1]') + report_text.count('[2]') + report_text.count('[3]')

    print(f"\nReport Statistics:")
    print(f"  Lines: {num_lines}")
    print(f"  Code citations: {num_citations}")
    print(f"  Paper citations: {num_papers}")

    return result


def main():
    """Run the full pipeline."""
    print("=" * 70)
    print("FULL DISCOVERY-TO-REPORT PIPELINE")
    print("=" * 70)

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Initialize world model
    print("\nStep 1: Initialize World Model")
    print("-" * 70)
    wm = WorldModel()
    print(f"Created: {wm}")

    # Step 2: Simulate discovery cycles
    simulate_discovery_cycle(wm)

    # Step 3: Generate report
    result = generate_comprehensive_report(wm, output_dir)

    # Step 4: Save world model
    print("\n" + "=" * 70)
    print("SAVING ARTIFACTS")
    print("=" * 70)

    wm_path = output_dir / "discovery_world_model.db"
    wm.save(wm_path)
    print(f"\nWorld model saved to: {wm_path}")
    print(f"File size: {wm_path.stat().st_size} bytes")

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print("\nGenerated Files:")
    print(f"  📄 Main Report: {result['report']}")
    print(f"  📋 Appendix: {result.get('appendix', 'N/A')}")
    print(f"  💾 World Model: {wm_path}")

    print("\nNext Steps:")
    print(f"  1. Review the report: cat {result['report']}")
    print(f"  2. Convert to PDF: scripts/convert_report_to_pdf.sh -i {result['report']}")
    print(f"  3. Load world model: WorldModel.load('{wm_path}')")

    print("\nTo enable AI-powered narratives:")
    print("  export ANTHROPIC_API_KEY=your-key")
    print("  # Then set generate_narratives=True in the code")

    print()


if __name__ == "__main__":
    main()
