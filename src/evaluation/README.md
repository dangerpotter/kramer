# Expert Evaluation System

Post-hoc expert evaluation system for validating research claims, as described in the Kosmos paper.

## Overview

This system provides tools for:

1. **Extracting testable claims** from generated research reports
2. **Collecting expert evaluations** through an interactive interface
3. **Tracking accuracy metrics** over time and by claim type
4. **Generating evaluation reports** with calibration analysis

## Components

### 1. ClaimExtractor (`claim_extractor.py`)

Parses markdown reports and extracts testable claims, categorizing them into:

- **Data Analysis**: Statistical findings, correlations, measurements
- **Literature**: Claims about existing research
- **Interpretation**: Explanatory claims and implications

```python
from src.evaluation import ClaimExtractor

extractor = ClaimExtractor()
claims = extractor.extract_claims(Path("outputs/report.md"))
extractor.save_claims(claims, Path("outputs/evaluation/claims.json"))
```

### 2. EvaluationInterface (`evaluation_interface.py`)

Interactive interface for expert review with SQLite database storage:

```python
from src.evaluation import EvaluationInterface

interface = EvaluationInterface(Path("outputs/evaluation/evaluations.db"))

# Store claims
interface.store_claims(claims)

# Run interactive session
interface.interactive_evaluation_session(
    claims,
    evaluator_id="expert_1",
    auto_save=True
)
```

### 3. MetricsTracker (`metrics_tracker.py`)

Computes accuracy metrics and generates reports:

```python
from src.evaluation import MetricsTracker

tracker = MetricsTracker(Path("outputs/evaluation/evaluations.db"))

# Compute overall accuracy
metrics = tracker.compute_accuracy()
print(f"Overall accuracy: {metrics.accuracy:.1%}")

# Generate full report
tracker.generate_report(Path("outputs/evaluation/accuracy_report.md"))
```

## Quick Start

### Command-Line Interface

The easiest way to use the system is through the CLI:

```bash
# Full pipeline (extract + evaluate + report)
python -m src.evaluation.run_evaluation pipeline outputs/report.md

# Or run steps individually:

# 1. Extract claims
python -m src.evaluation.run_evaluation extract outputs/report.md

# 2. Evaluate claims
python -m src.evaluation.run_evaluation evaluate outputs/evaluation/claims.json

# 3. Generate accuracy report
python -m src.evaluation.run_evaluation report
```

### Programmatic Integration

Integrate with the orchestrator or report generation:

```python
from pathlib import Path
from src.evaluation.integrate_with_report import integrate_with_orchestrator

# After generating a report
report_path = Path("outputs/report.md")

# Run evaluation pipeline (non-interactive)
results = integrate_with_orchestrator(
    report_path=report_path,
    run_interactive=False  # Set True for interactive session
)

print(f"Extracted {results['claims_extracted']} claims")
```

### Integration with ReportGenerator

Add to your orchestrator after report generation:

```python
from src.reporting.report_generator import ReportGenerator
from src.evaluation.integrate_with_report import post_report_hook

# Generate report
generator = ReportGenerator(world_model, anthropic_api_key)
report_result = generator.generate_report(output_path)

# Run evaluation pipeline automatically
post_report_hook(report_result)
```

## Evaluation Workflow

### Step 1: Extract Claims

Claims are automatically extracted from markdown reports:

- Parses discovery sections
- Extracts individual findings and narratives
- Categorizes by claim type
- Includes confidence scores and context

### Step 2: Expert Evaluation

Interactive session presents each claim:

```
================================================================================
CLAIM claim_0001
================================================================================

Discovery: Mean temperature increased significantly over study period
Type: data_analysis
Original Confidence: 0.85

Claim:
The mean temperature showed a significant increase of 2.3°C over the study period.

--------------------------------------------------------------------------------

Verdicts:
  1. Supported - Claim is accurate/supported by evidence
  2. Refuted - Claim is inaccurate/contradicted by evidence
  3. Unclear - Insufficient evidence or ambiguous
  4. Partially Supported - Claim is partially accurate

Enter verdict (1-4) or 's' to skip:
```

### Step 3: Accuracy Tracking

The system computes:

- **Overall accuracy**: Percentage of supported/partially supported claims
- **By category**: Accuracy for data analysis, literature, interpretation
- **Trends**: Accuracy over time windows
- **Calibration**: System confidence vs. actual accuracy

### Step 4: Reports

Generated reports include:

- Verdict distribution
- Accuracy by claim type
- Temporal trends
- Confidence calibration analysis
- Actionable recommendations

## Database Schema

SQLite database with two main tables:

### `claims` table
```sql
CREATE TABLE claims (
    claim_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    claim_type TEXT NOT NULL,  -- data_analysis | literature | interpretation
    discovery_title TEXT,
    confidence REAL,
    context TEXT,
    source_section TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL
);
```

### `evaluations` table
```sql
CREATE TABLE evaluations (
    evaluation_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    verdict TEXT,  -- supported | refuted | unclear | partially_supported
    evaluator_id TEXT NOT NULL,
    notes TEXT,
    confidence_in_verdict REAL,
    timestamp TEXT NOT NULL,
    metadata TEXT,
    UNIQUE(claim_id, evaluator_id)
);
```

## Output Files

Running the full pipeline generates:

```
outputs/evaluation/
├── claims.json              # Extracted claims with metadata
├── evaluations.db           # SQLite database with evaluations
└── accuracy_report.md       # Comprehensive accuracy report
```

## Example Accuracy Report

```markdown
# Expert Evaluation Accuracy Report

*Generated on 2025-11-07 14:32:15*

## Executive Summary

- **Total Claims**: 47
- **Evaluated Claims**: 47
- **Overall Accuracy**: 78.7%
- **Support Rate**: 63.8%
- **Refute Rate**: 14.9%
- **Unclear Rate**: 6.4%

## Accuracy by Claim Type

| Claim Type | Total | Accuracy | Support Rate | Refute Rate | Unclear Rate |
|------------|-------|----------|--------------|-------------|--------------|
| data_analysis | 28 | 85.7% | 75.0% | 10.7% | 3.6% |
| interpretation | 12 | 66.7% | 50.0% | 25.0% | 8.3% |
| literature | 7 | 71.4% | 57.1% | 14.3% | 14.3% |

## Confidence Calibration

| Confidence Range | Count | Actual Accuracy | Expected Confidence |
|------------------|-------|-----------------|---------------------|
| 0.7-0.8 | 12 | 66.7% | 75.0% |
| 0.8-0.9 | 23 | 78.3% | 85.0% |
| 0.9-1.0 | 12 | 91.7% | 95.0% |

## Recommendations

- **Over-confidence detected**: System confidence scores are higher than actual accuracy. Consider recalibrating confidence thresholds.
```

## Advanced Usage

### Custom Evaluator IDs

Track evaluations from multiple experts:

```python
# Expert 1
interface.interactive_evaluation_session(claims, evaluator_id="expert_1")

# Expert 2
interface.interactive_evaluation_session(claims, evaluator_id="expert_2")

# Compare evaluations
expert1_evals = interface.get_evaluations(evaluator_id="expert_1")
expert2_evals = interface.get_evaluations(evaluator_id="expert_2")
```

### Filtering and Analysis

```python
# Get all refuted claims
refuted = interface.get_evaluations(verdict=Verdict.REFUTED)

# Analyze specific claim
claim_evals = interface.get_evaluations(claim_id="claim_0001")

# Compute metrics for specific time period
from datetime import datetime, timedelta

metrics = tracker.compute_accuracy(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now()
)
```

### Export Data

```python
# Export evaluations to JSON
interface.export_evaluations(Path("outputs/evaluation/evaluations.json"))

# Export metrics to JSON
tracker.export_metrics(Path("outputs/evaluation/metrics.json"))
```

## Integration Points

### 1. Orchestrator Integration

Add to `src/orchestrator/agent_coordinator.py`:

```python
from src.evaluation.integrate_with_report import post_report_hook

# After report generation
if report_result:
    post_report_hook(report_result)
```

### 2. Automated Pipeline

Run as part of CI/CD or scheduled evaluations:

```bash
#!/bin/bash
# evaluate.sh

# Generate report
python main.py --generate-report

# Extract and store claims (non-interactive)
python -m src.evaluation.run_evaluation pipeline \
    outputs/report.md \
    --non-interactive

# Manual evaluation can be done later
echo "Claims extracted. Run interactive evaluation with:"
echo "python -m src.evaluation.run_evaluation evaluate outputs/evaluation/claims.json"
```

### 3. Batch Processing

Process multiple reports:

```python
from pathlib import Path
from src.evaluation import ClaimExtractor, EvaluationInterface

extractor = ClaimExtractor()
interface = EvaluationInterface(Path("outputs/evaluation/evaluations.db"))

# Process all reports
for report_path in Path("outputs/reports").glob("*.md"):
    claims = extractor.extract_claims(report_path)
    interface.store_claims(claims)

# Run single evaluation session for all claims
all_claims = interface.get_unevaluated_claims()
interface.interactive_evaluation_session(all_claims)
```

## Future Enhancements

Potential improvements:

1. **Multi-evaluator consensus**: Compute inter-rater reliability
2. **Active learning**: Prioritize uncertain claims for evaluation
3. **LLM-assisted evaluation**: Use Claude to pre-evaluate claims
4. **Claim versioning**: Track changes in claims over report versions
5. **Evidence linking**: Connect claims to specific data/plots
6. **Export to other formats**: CSV, Excel, etc.

## References

This implementation is based on the expert evaluation methodology described in:

**Kosmos: Automating Research Discovery with AI** - Post-hoc expert evaluation section

The system follows the Kosmos approach of:
- Extracting factual claims from generated reports
- Categorizing by evidence type
- Collecting binary or multi-class expert verdicts
- Tracking accuracy over time for continuous improvement
