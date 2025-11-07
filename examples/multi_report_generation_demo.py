"""
Multi-Report Generation Demo - Shows how to generate multiple reports with different focuses.

This example demonstrates the multi-report generation feature that creates
three or four different report variants from the same world model data:
1. Comprehensive - All findings above baseline confidence
2. High Confidence - Only the most reliable findings
3. Novel Discoveries - Findings with high novelty scores
4. Validated Hypotheses - Findings connected to tested hypotheses
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reporting.report_generator import ReportGenerator, ReportConfig
from src.world_model.graph import EdgeType, NodeType, WorldModel


def create_sample_world_model() -> WorldModel:
    """Create a sample world model with diverse findings."""
    print("Creating sample world model with diverse findings...")
    wm = WorldModel()

    # ========================================
    # Discovery 1: High confidence, well-supported
    # ========================================
    print("  Adding Discovery 1: Temperature-CO2 Relationship (high confidence)")

    ds1 = wm.add_dataset(
        text="Global climate data 1980-2023",
        path="/data/climate.csv",
        metadata={"rows": 50000, "features": 15},
    )

    f1 = wm.add_finding(
        text="Global average temperature has increased by 1.1°C since pre-industrial times",
        code_link="analysis/temperature_trends.py:142",
        confidence=0.98,
        metadata={"source": "climate_data_2023.csv"},
    )

    f2 = wm.add_finding(
        text="CO2 concentration has risen from 280ppm to 420ppm",
        code_link="analysis/co2_analysis.py:89",
        confidence=0.97,
        metadata={"source": "atmospheric_data.csv"},
    )

    wm.add_edge(ds1, f1, EdgeType.RELATES_TO)
    wm.add_edge(ds1, f2, EdgeType.RELATES_TO)

    # Tested hypothesis
    h1 = wm.add_hypothesis(
        text="Rising CO2 is the primary driver of temperature increase",
        parent_finding=f1,
        confidence=0.85,  # Tested with confidence score
        metadata={"status": "tested", "result": "supported"},
    )
    wm.add_edge(h1, f1, EdgeType.SUPPORTS)

    # Supporting papers
    p1 = wm.add_paper(
        text="Meta-analysis of greenhouse gas effects",
        title="Greenhouse Gases and Global Temperature: A Meta-Analysis",
        authors=["Smith, J.", "Johnson, M."],
        year=2022,
        doi="10.1038/nature12345",
    )
    wm.add_edge(p1, h1, EdgeType.SUPPORTS)

    # ========================================
    # Discovery 2: Novel finding (contradicts literature)
    # ========================================
    print("  Adding Discovery 2: Ice Melt Acceleration (novel)")

    f3 = wm.add_finding(
        text="Arctic sea ice loss has accelerated by 50% beyond model predictions",
        code_link="analysis/ice_acceleration.py:78",
        confidence=0.88,
        metadata={"novelty": "contradicts_models"},
    )

    # Novel finding contradicts old projections
    p2 = wm.add_paper(
        text="Previous models predicted slower ice melt",
        title="Projections of Arctic Ice Loss",
        authors=["Anderson, R."],
        year=2015,
        doi="10.1029/jgr12345",
    )
    wm.add_edge(p2, f3, EdgeType.REFUTES)  # Novel: contradicts literature

    # ========================================
    # Discovery 3: Medium confidence, hypothesis-driven
    # ========================================
    print("  Adding Discovery 3: Ocean Acidification (hypothesis-tested)")

    q1 = wm.add_question(
        text="How does rising CO2 affect ocean chemistry?",
        metadata={"priority": "high"},
    )

    f4 = wm.add_finding(
        text="Ocean pH has decreased by 0.1 units since pre-industrial times",
        code_link="analysis/ocean_chemistry.py:145",
        confidence=0.82,
        metadata={"ph_change": -0.1},
    )
    wm.add_edge(q1, f4, EdgeType.RELATES_TO)

    # Another tested hypothesis
    h2 = wm.add_hypothesis(
        text="Ocean acidification increases with atmospheric CO2",
        parent_finding=f4,
        confidence=0.78,  # Tested with confidence score
        metadata={"status": "tested"},
    )
    wm.add_edge(h2, f4, EdgeType.SUPPORTS)

    # ========================================
    # Discovery 4: Novel but lower confidence
    # ========================================
    print("  Adding Discovery 4: Methane Release (novel, medium confidence)")

    f5 = wm.add_finding(
        text="Permafrost methane release rates exceed previous estimates by 2x",
        code_link="analysis/methane.py:234",
        confidence=0.76,
        metadata={"novelty": "new_discovery"},
    )

    # Novel: few supporting papers
    p3 = wm.add_paper(
        text="Limited research on methane release mechanisms",
        title="Methane Emissions from Thawing Permafrost",
        authors=["Wilson, K."],
        year=2023,
        doi="10.1126/science.xyz123",
    )
    wm.add_edge(p3, f5, EdgeType.SUPPORTS)

    # ========================================
    # Discovery 5: Medium confidence, not hypothesis-tested
    # ========================================
    print("  Adding Discovery 5: Precipitation Patterns (medium confidence)")

    f6 = wm.add_finding(
        text="Extreme precipitation events have increased by 30% in past decade",
        code_link="analysis/precipitation.py:156",
        confidence=0.81,
        metadata={"trend": "increasing"},
    )

    # Untested hypothesis (no confidence score)
    h3 = wm.add_hypothesis(
        text="Warmer air holds more moisture, leading to extreme events",
        parent_finding=f6,
        # No confidence - not tested yet
        metadata={"status": "proposed"},
    )

    # ========================================
    # Low confidence findings (should be filtered in high_confidence report)
    # ========================================
    f7 = wm.add_finding(
        text="Preliminary correlation detected between solar cycles and temperature",
        code_link="analysis/exploratory.py:45",
        confidence=0.65,  # Below high-confidence threshold
    )

    f8 = wm.add_finding(
        text="Early-stage analysis of cloud feedback mechanisms",
        code_link="analysis/clouds.py:89",
        confidence=0.58,  # Low confidence
    )

    print(f"  Created world model: {wm}")
    return wm


def main():
    """Main demonstration."""
    print("=" * 80)
    print("Multi-Report Generation Demo")
    print("=" * 80)
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

    # ========================================
    # Option 1: Use default report configs
    # ========================================
    print("=" * 80)
    print("Option 1: Generate Multiple Reports with Default Configurations")
    print("=" * 80)
    print()

    # Create report generator
    print("Creating report generator...")
    rg = ReportGenerator(
        world_model=wm,
        min_confidence=0.7,  # Base threshold (configs can override)
        max_discoveries=5,
    )
    print()

    # Generate multiple reports
    print("Generating multiple reports with default configurations...")
    output_dir = Path("outputs/multi_reports")

    result = rg.generate_multiple_reports(
        output_dir=output_dir,
        report_configs=None,  # Use defaults
        generate_narratives=False,  # Set to True to use Claude API
        include_appendix=True,
    )

    print(f"\nGenerated {len(result['reports'])} reports:")
    print()
    for report in result["reports"]:
        print(f"  {report['name']}:")
        print(f"    Path: {report['report']}")
        print(f"    Discoveries: {report['discoveries_count']}")
        print(f"    Total Findings: {report['total_findings']}")
        print(f"    High Confidence: {report['high_confidence_findings']}")
        if 'appendix' in report:
            print(f"    Appendix: {report['appendix']}")
        print()

    # ========================================
    # Option 2: Use custom report configs
    # ========================================
    print("=" * 80)
    print("Option 2: Generate Reports with Custom Configurations")
    print("=" * 80)
    print()

    # Define custom report configurations
    custom_configs = [
        ReportConfig(
            name="executive_summary",
            min_confidence=0.95,
            sort_by="confidence",
            max_discoveries=3,
            filter_type="all",
            include_all=False,
        ),
        ReportConfig(
            name="research_highlights",
            min_confidence=0.75,
            sort_by="novelty",
            top_n=5,
            max_discoveries=5,
            filter_type="novel_only",
        ),
    ]

    print("Generating reports with custom configurations...")
    custom_output_dir = Path("outputs/custom_reports")

    custom_result = rg.generate_multiple_reports(
        output_dir=custom_output_dir,
        report_configs=custom_configs,
        generate_narratives=False,
        include_appendix=False,  # No appendix for custom reports
    )

    print(f"\nGenerated {len(custom_result['reports'])} custom reports:")
    print()
    for report in custom_result["reports"]:
        print(f"  {report['name']}:")
        print(f"    Path: {report['report']}")
        print(f"    Discoveries: {report['discoveries_count']}")
        print(f"    Total Findings: {report['total_findings']}")
        print()

    # ========================================
    # Preview one report
    # ========================================
    if result["reports"]:
        print("=" * 80)
        print("Preview: Comprehensive Report (first 30 lines)")
        print("=" * 80)
        comprehensive_report = result["reports"][0]["report"]
        with open(comprehensive_report, 'r') as f:
            for i, line in enumerate(f):
                if i >= 30:
                    print("...")
                    break
                print(line.rstrip())
        print()

    # ========================================
    # Summary
    # ========================================
    print("=" * 80)
    print("Summary of Generated Reports")
    print("=" * 80)
    print()
    print("Default Reports (outputs/multi_reports/):")
    print("  1. comprehensive_report.md - All findings >= 0.7 confidence")
    print("  2. high_confidence_report.md - Only findings >= 0.9 confidence")
    print("  3. novel_discoveries_report.md - Top 10 findings by novelty score")
    print("  4. validated_hypotheses_report.md - Only hypothesis-tested findings")
    print()
    print("Custom Reports (outputs/custom_reports/):")
    print("  1. executive_summary.md - Top 3 discoveries, >= 0.95 confidence")
    print("  2. research_highlights.md - Top 5 novel findings")
    print()
    print("Next steps:")
    print(f"  - View reports in: {output_dir}")
    print(f"  - View custom reports in: {custom_output_dir}")
    print()
    print("To enable AI-generated narratives:")
    print("  1. Set ANTHROPIC_API_KEY environment variable")
    print("  2. Pass api_key to ReportGenerator constructor")
    print("  3. Set generate_narratives=True in generate_multiple_reports()")
    print()
    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
