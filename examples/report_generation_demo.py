"""
Report Generation Demo - Shows how to generate publication-quality reports.

This example creates a realistic world model with discoveries and generates
a comprehensive research report.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reporting.report_generator import ReportGenerator
from src.world_model.graph import EdgeType, NodeType, WorldModel


def create_sample_world_model() -> WorldModel:
    """Create a sample world model with discoveries."""
    print("Creating sample world model...")
    wm = WorldModel()

    # ========================================
    # Discovery 1: Temperature-CO2 Relationship
    # ========================================
    print("  Adding Discovery 1: Temperature-CO2 Relationship")

    # Dataset
    ds1 = wm.add_dataset(
        text="Global climate data 1980-2023",
        path="/data/climate.csv",
        metadata={"rows": 50000, "features": 15},
    )

    # Findings
    f1 = wm.add_finding(
        text="Global average temperature has increased by 1.1°C since pre-industrial times",
        code_link="analysis/temperature_trends.py:142",
        confidence=0.98,
        metadata={"source": "climate_data_2023.csv", "method": "linear_regression"},
    )

    f2 = wm.add_finding(
        text="CO2 concentration has risen from 280ppm to 420ppm over the same period",
        code_link="analysis/co2_analysis.py:89",
        confidence=0.97,
        metadata={"source": "atmospheric_data.csv", "correlation_with_temp": 0.94},
    )

    f3 = wm.add_finding(
        text="Strong positive correlation (r=0.94, p<0.001) between CO2 and temperature",
        code_link="analysis/correlation_analysis.py:234",
        confidence=0.96,
        metadata={"statistical_test": "pearson", "sample_size": 50000},
    )

    # Connect findings
    wm.add_edge(f1, f3, EdgeType.SUPPORTS)
    wm.add_edge(f2, f3, EdgeType.SUPPORTS)
    wm.add_edge(ds1, f1, EdgeType.RELATES_TO)
    wm.add_edge(ds1, f2, EdgeType.RELATES_TO)

    # Hypothesis
    h1 = wm.add_hypothesis(
        text="Rising CO2 is the primary driver of observed temperature increase",
        parent_finding=f3,
        confidence=0.85,
        metadata={"mechanism": "greenhouse_effect"},
    )

    # Supporting papers
    p1 = wm.add_paper(
        text="Comprehensive analysis of greenhouse gas effects on global temperature",
        title="Greenhouse Gases and Global Temperature: A Meta-Analysis",
        authors=["Smith, J.", "Johnson, M.", "Williams, K."],
        year=2022,
        doi="10.1038/nature12345",
    )
    wm.add_edge(p1, h1, EdgeType.SUPPORTS)

    p2 = wm.add_paper(
        text="Historical CO2 and temperature data show strong correlation",
        title="Historical Climate Data Analysis: CO2 and Temperature Trends",
        authors=["Chen, L.", "Wang, X."],
        year=2023,
        doi="10.1126/science.abcd1234",
    )
    wm.add_edge(p2, f3, EdgeType.SUPPORTS)

    # ========================================
    # Discovery 2: Ice Melt Acceleration
    # ========================================
    print("  Adding Discovery 2: Ice Melt Acceleration")

    ds2 = wm.add_dataset(
        text="Arctic ice extent satellite measurements 1979-2023",
        path="/data/ice_extent.nc",
        metadata={"source": "NSIDC", "resolution": "25km"},
    )

    f4 = wm.add_finding(
        text="Arctic sea ice extent has declined by 13% per decade since 1979",
        code_link="analysis/ice_trends.py:56",
        confidence=0.95,
        metadata={"source": "satellite_data.nc", "trend": "accelerating"},
    )

    f5 = wm.add_finding(
        text="Rate of ice loss has accelerated significantly in the past 10 years",
        code_link="analysis/ice_acceleration.py:78",
        confidence=0.92,
        metadata={"recent_rate": "15%_per_decade", "historical_rate": "11%_per_decade"},
    )

    # Connect to temperature findings
    wm.add_edge(f1, f4, EdgeType.SUPPORTS)
    wm.add_edge(f4, f5, EdgeType.SUPPORTS)
    wm.add_edge(ds2, f4, EdgeType.RELATES_TO)

    h2 = wm.add_hypothesis(
        text="Temperature increase causes accelerated polar ice melt through albedo feedback",
        parent_finding=f5,
        confidence=0.80,
        metadata={"feedback_mechanism": "ice_albedo"},
    )

    # Novel finding - contradicts some literature
    p3 = wm.add_paper(
        text="Previous models predicted slower ice melt rates",
        title="Projections of Arctic Ice Loss Under Climate Change",
        authors=["Anderson, R.", "Brown, T."],
        year=2015,
        doi="10.1029/jgr12345",
    )
    wm.add_edge(p3, f5, EdgeType.REFUTES)  # Our finding contradicts old projections

    # ========================================
    # Discovery 3: Ocean Acidification
    # ========================================
    print("  Adding Discovery 3: Ocean Acidification")

    q1 = wm.add_question(
        text="How does rising CO2 affect ocean chemistry?",
        metadata={"priority": "high", "domain": "oceanography"},
    )

    f6 = wm.add_finding(
        text="Ocean pH has decreased by 0.1 units since pre-industrial times (30% increase in acidity)",
        code_link="analysis/ocean_chemistry.py:145",
        confidence=0.94,
        metadata={"ph_change": -0.1, "baseline": 8.2},
    )

    f7 = wm.add_finding(
        text="Ocean CO2 absorption rate strongly correlates with atmospheric CO2 levels",
        code_link="analysis/ocean_co2.py:203",
        confidence=0.91,
        metadata={"absorption_rate": "2.5Gt_per_year"},
    )

    # This finding answers the question
    wm.add_edge(q1, f6, EdgeType.RELATES_TO)
    wm.add_edge(f2, f6, EdgeType.SUPPORTS)  # CO2 rise supports acidification
    wm.add_edge(f6, f7, EdgeType.SUPPORTS)

    p4 = wm.add_paper(
        text="Ocean acidification impacts on marine ecosystems",
        title="The Other CO2 Problem: Ocean Acidification",
        authors=["Doney, S.", "Fabry, V.", "Feely, R."],
        year=2020,
        doi="10.1146/annurev.marine.010908.163834",
    )
    wm.add_edge(p4, f6, EdgeType.SUPPORTS)

    # ========================================
    # Some lower confidence findings (should be filtered)
    # ========================================
    f8 = wm.add_finding(
        text="Preliminary analysis suggests correlation with solar cycles",
        code_link="analysis/exploratory.py:45",
        confidence=0.45,  # Low confidence - will be filtered
    )

    print(f"  Created world model: {wm}")
    return wm


def main():
    """Main demonstration."""
    print("=" * 70)
    print("Report Generation Demo")
    print("=" * 70)
    print()

    # Create sample world model
    wm = create_sample_world_model()

    # Show world model stats
    stats = wm.get_stats()
    print(f"\nWorld Model Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Findings: {stats['node_types'].get('finding', 0)}")
    print(f"  Papers: {stats['node_types'].get('paper', 0)}")
    print(f"  Hypotheses: {stats['node_types'].get('hypothesis', 0)}")
    print(f"  Questions: {stats['node_types'].get('question', 0)}")
    print(f"  Total edges: {stats['total_edges']}")
    print()

    # Create report generator
    print("Creating report generator...")
    rg = ReportGenerator(
        world_model=wm,
        min_confidence=0.7,  # Filter to high-confidence findings
        max_discoveries=5,
    )
    print()

    # Extract high-confidence findings
    print("Extracting high-confidence findings...")
    findings = rg.extract_high_confidence_findings()
    print(f"  Found {len(findings)} high-confidence findings (>0.7)")
    print()

    # Show top findings
    print("Top 3 findings:")
    for i, finding in enumerate(findings[:3], 1):
        print(f"  {i}. {finding['text'][:70]}...")
        print(f"     Confidence: {finding['confidence']:.2f}, Novelty: {finding['novelty_score']:.2f}")
    print()

    # Group into discoveries
    print("Grouping findings into discoveries...")
    discoveries = rg.group_findings_into_discoveries(findings)
    print(f"  Created {len(discoveries)} discoveries")
    print()

    # Show discoveries
    print("Discoveries:")
    for i, disc in enumerate(discoveries, 1):
        print(f"  {i}. {disc.title}")
        print(f"     Findings: {len(disc.findings)}, Papers: {len(disc.papers)}, "
              f"Confidence: {disc.confidence:.2f}")
    print()

    # Generate report
    print("Generating report...")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "discovery_report.md"

    result = rg.generate_report(
        output_path=output_path,
        include_appendix=True,
        generate_narratives=False,  # Set to True to use Claude API
    )

    print(f"  Main report: {result['report']}")
    print(f"  Appendix: {result.get('appendix', 'N/A')}")
    print()

    # Show report preview
    print("Report preview (first 30 lines):")
    print("-" * 70)
    with open(result['report'], 'r') as f:
        for i, line in enumerate(f):
            if i >= 30:
                print("...")
                break
            print(line.rstrip())
    print("-" * 70)
    print()

    # Save world model for later use
    wm_path = output_dir / "sample_world_model.db"
    wm.save(wm_path)
    print(f"Saved world model to: {wm_path}")
    print()

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print(f"  - View the full report: {result['report']}")
    print(f"  - View the appendix: {result.get('appendix', 'N/A')}")
    print(f"  - Load world model: WorldModel.load('{wm_path}')")
    print()
    print("To use Claude API for narrative generation:")
    print("  1. Set ANTHROPIC_API_KEY environment variable")
    print("  2. Pass api_key to ReportGenerator")
    print("  3. Set generate_narratives=True in generate_report()")
    print()


if __name__ == "__main__":
    main()
