#!/usr/bin/env python3
"""
Example: Expert Evaluation System

This script demonstrates how to use the expert evaluation system
to extract claims, collect evaluations, and track accuracy.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    ClaimExtractor,
    EvaluationInterface,
    MetricsTracker,
    Verdict,
)


def create_sample_report(output_path: Path) -> None:
    """Create a sample report for testing."""
    sample_report = """# Discovery Report

*Generated on 2025-11-07 10:00:00*

## Executive Summary

This report presents 2 key discoveries extracted from the research world model. All findings have confidence scores above 0.70 and are supported by data analysis and literature review.

**World Model Statistics:**
- Total nodes: 45
- Findings: 23
- Papers: 8
- Hypotheses: 5

## Discovery 1: Temperature increase correlates with species decline

**Confidence:** 0.85 | **Novelty Score:** 0.60

Analysis of climate data from 2010-2020 reveals a significant correlation between rising temperatures and declining species populations in the study region. This finding contradicts some earlier optimistic projections and suggests more urgent intervention may be needed. The mean temperature increased by 2.3°C over the study period, while species diversity decreased by 18%.

### Supporting Evidence

**Data Analysis:**

- Mean temperature increased by 2.3°C from 2010 to 2020 [Analysis r1] (confidence: 0.92)
- Species diversity index decreased by 18% over the same period [Analysis r2] (confidence: 0.87)
- Strong negative correlation (r=-0.78, p<0.001) between temperature and diversity [Analysis r3] (confidence: 0.85)

**Literature Support:**

This discovery is supported by 3 papers from the literature [1], [2], [3].

**Related Hypotheses:**

- Rising temperatures may exceed species adaptation rates (confidence: 0.75)
- Conservation interventions could mitigate decline trends (confidence: 0.65)

---

## Discovery 2: Habitat fragmentation amplifies temperature effects

**Confidence:** 0.78 | **Novelty Score:** 0.45

Fragmented habitats show significantly greater species loss compared to continuous habitats under similar temperature increases. This interaction effect suggests that habitat connectivity should be a priority for conservation efforts. The effect is most pronounced in areas with less than 30% habitat connectivity.

### Supporting Evidence

**Data Analysis:**

- Fragmented habitats showed 2.5x greater species loss than continuous habitats [Analysis r4] (confidence: 0.81)
- Interaction effect between temperature and fragmentation is significant (F=12.4, p<0.01) [Analysis r5] (confidence: 0.79)
- Habitat connectivity below 30% threshold correlates with accelerated decline [Analysis r6] (confidence: 0.76)

**Literature Support:**

This discovery is supported by 2 papers from the literature [4], [5].

---

## References

### Literature

[1] Smith, J. et al. (2019). Climate change impacts on biodiversity. Nature 123:456-789.

[2] Jones, M. and Brown, R. (2020). Temperature thresholds for species survival. Science 456:123-456.

[3] Davis, K. et al. (2021). Rapid climate change and ecosystem collapse. Ecology Letters 78:234-567.

[4] Wilson, P. (2018). Habitat fragmentation and climate resilience. Conservation Biology 45:678-901.

[5] Taylor, S. and Anderson, L. (2020). Landscape connectivity under climate change. Landscape Ecology 34:123-456.

### Code References

[Analysis r1] Code: notebook_001.ipynb (Finding 7a8b9c0d)

[Analysis r2] Code: notebook_002.ipynb (Finding 4e5f6a7b)

[Analysis r3] Code: notebook_003.ipynb (Finding 1c2d3e4f)

[Analysis r4] Code: notebook_004.ipynb (Finding 9f8e7d6c)

[Analysis r5] Code: notebook_005.ipynb (Finding 5b4a3c2d)

[Analysis r6] Code: notebook_006.ipynb (Finding 1e2f3a4b)
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(sample_report)

    print(f"Created sample report: {output_path}")


def example_claim_extraction():
    """Example 1: Extract claims from a report."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Claim Extraction")
    print("="*80 + "\n")

    # Create sample report
    report_path = Path("outputs/example_report.md")
    create_sample_report(report_path)

    # Extract claims
    extractor = ClaimExtractor()
    claims = extractor.extract_claims(report_path)

    print(f"Extracted {len(claims)} claims from report\n")

    # Show first few claims
    print("Sample claims:")
    for i, claim in enumerate(claims[:5], 1):
        print(f"\n{i}. [{claim.claim_type.value}] {claim.text}")
        if claim.confidence:
            print(f"   Confidence: {claim.confidence:.2f}")

    # Save claims
    claims_path = Path("outputs/evaluation_example/claims.json")
    extractor.save_claims(claims, claims_path)
    print(f"\nSaved claims to: {claims_path}")

    return claims, claims_path


def example_programmatic_evaluation(claims):
    """Example 2: Programmatic evaluation (non-interactive)."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Programmatic Evaluation")
    print("="*80 + "\n")

    # Initialize interface
    db_path = Path("outputs/evaluation_example/evaluations.db")
    interface = EvaluationInterface(db_path)

    # Store claims
    interface.store_claims(claims)
    print(f"Stored {len(claims)} claims in database")

    # Create some example evaluations programmatically
    from src.evaluation.evaluation_interface import Evaluation

    evaluations = []

    # Evaluate first 3 claims
    for i, claim in enumerate(claims[:3]):
        # Simulate expert evaluation
        # In real use, this would come from interactive session
        verdict = Verdict.SUPPORTED if claim.confidence and claim.confidence > 0.8 else Verdict.UNCLEAR

        evaluation = Evaluation(
            claim_id=claim.claim_id,
            verdict=verdict,
            evaluator_id="example_evaluator",
            notes=f"Example evaluation for claim {i+1}",
            confidence_in_verdict=0.9,
        )
        evaluations.append(evaluation)

    # Save evaluations
    interface.save_evaluations(evaluations)
    print(f"Saved {len(evaluations)} example evaluations")

    return db_path


def example_metrics_tracking(db_path):
    """Example 3: Track accuracy metrics."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Metrics Tracking")
    print("="*80 + "\n")

    # Initialize tracker
    tracker = MetricsTracker(db_path)

    # Compute accuracy
    metrics = tracker.compute_accuracy()

    print(f"Overall Metrics:")
    print(f"  Total claims: {metrics.total_claims}")
    print(f"  Evaluated claims: {metrics.evaluated_claims}")
    print(f"  Accuracy: {metrics.accuracy:.1%}")
    print(f"  Support rate: {metrics.support_rate:.1%}")
    print(f"  Refute rate: {metrics.refute_rate:.1%}")
    print(f"  Unclear rate: {metrics.unclear_rate:.1%}")

    # By claim type
    if metrics.by_type:
        print(f"\nAccuracy by claim type:")
        for claim_type, type_metrics in metrics.by_type.items():
            print(f"  {claim_type}: {type_metrics['accuracy']:.1%} ({type_metrics['total']} claims)")

    # Generate report
    report_path = Path("outputs/evaluation_example/accuracy_report.md")
    tracker.generate_report(report_path)
    print(f"\nGenerated accuracy report: {report_path}")


def example_integration():
    """Example 4: Full integration pipeline."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Full Integration Pipeline")
    print("="*80 + "\n")

    from src.evaluation.integrate_with_report import integrate_with_orchestrator

    # Create sample report
    report_path = Path("outputs/example_report_2.md")
    create_sample_report(report_path)

    # Run full pipeline (non-interactive)
    results = integrate_with_orchestrator(
        report_path=report_path,
        evaluation_dir=Path("outputs/evaluation_example_2"),
        run_interactive=False,
    )

    print(f"Pipeline results:")
    print(f"  Claims extracted: {results['claims_extracted']}")
    print(f"  Claims path: {results['claims_path']}")
    print(f"  Database path: {results['database_path']}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("EXPERT EVALUATION SYSTEM - EXAMPLES")
    print("="*80)

    # Example 1: Claim extraction
    claims, claims_path = example_claim_extraction()

    # Example 2: Programmatic evaluation
    db_path = example_programmatic_evaluation(claims)

    # Example 3: Metrics tracking
    example_metrics_tracking(db_path)

    # Example 4: Full integration
    example_integration()

    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
    print("\nTo run interactive evaluation:")
    print(f"  python -m src.evaluation.run_evaluation evaluate {claims_path}")
    print("\nTo view the accuracy report:")
    print("  cat outputs/evaluation_example/accuracy_report.md")
    print()


if __name__ == "__main__":
    main()
