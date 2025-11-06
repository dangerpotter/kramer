# Report Generator

Publication-quality report generation for Kramer's discovery loop.

## Overview

The `ReportGenerator` analyzes the world model to extract high-confidence findings, groups them into coherent discoveries, and generates narrative reports with proper citations.

## Features

- **Finding Extraction**: Filters findings by confidence threshold (default >0.7)
- **Novelty Assessment**: Scores findings based on:
  - Contradiction with existing literature
  - Answers to research questions
  - Limited or no literature coverage
- **Discovery Grouping**: Uses graph connectivity to cluster related findings
- **Narrative Generation**:
  - Template-based (always available)
  - AI-powered with Claude API (optional)
- **Citation System**:
  - Papers: `[1, 2, 3]` with full bibliography
  - Code: `[Analysis r5]` with provenance links
- **Multi-format Output**:
  - Main report (markdown)
  - Appendix with detailed methods and all findings

## Quick Start

```python
from pathlib import Path
from src.reporting.report_generator import ReportGenerator
from src.world_model.graph import WorldModel

# Load your world model
wm = WorldModel.load(Path("world_model.db"))

# Create report generator
rg = ReportGenerator(
    world_model=wm,
    min_confidence=0.7,
    max_discoveries=5,
)

# Generate report
result = rg.generate_report(
    output_path=Path("outputs/report.md"),
    include_appendix=True,
    generate_narratives=False,  # Set True to use Claude API
)

print(f"Report: {result['report']}")
print(f"Appendix: {result['appendix']}")
```

## Using Claude API for Narratives

To enable AI-powered narrative generation:

1. Install the anthropic package:
   ```bash
   pip install anthropic
   ```

2. Set your API key:
   ```bash
   export ANTHROPIC_API_KEY=your-key-here
   ```

3. Pass API key to ReportGenerator:
   ```python
   import os

   rg = ReportGenerator(
       world_model=wm,
       anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
   )

   result = rg.generate_report(
       output_path=Path("outputs/report.md"),
       generate_narratives=True,  # Enable AI narratives
   )
   ```

## Report Structure

### Main Report

```markdown
# Discovery Report

## Executive Summary
- Overview of findings
- World model statistics

## Discovery 1: [Title]
**Confidence:** X.XX | **Novelty Score:** X.XX

[Narrative description]

### Supporting Evidence
**Data Analysis:**
- Finding 1 [Analysis r1] (confidence: 0.95)
- Finding 2 [Analysis r2] (confidence: 0.92)

**Literature Support:**
Supported by papers [1], [2], [3].

**Related Hypotheses:**
- Hypothesis A (confidence: 0.85)

## References
### Literature
[1] Author et al. (2023). Title. DOI

### Code References
[Analysis r1] Code: path/to/file.py:123
```

### Appendix

- Methods and parameters
- All high-confidence findings (detailed)
- World model statistics
- Discovery groupings

## Configuration

### ReportGenerator Parameters

- `world_model`: WorldModel instance
- `anthropic_api_key`: Optional API key for Claude
- `min_confidence`: Minimum confidence threshold (default: 0.7)
- `max_discoveries`: Maximum discoveries in report (default: 5)

### generate_report Parameters

- `output_path`: Path for main report
- `include_appendix`: Generate appendix (default: True)
- `generate_narratives`: Use AI narratives (default: True if API key provided)

## Examples

See `examples/report_generation_demo.py` for a complete example.

## Converting to PDF

To convert the markdown report to PDF:

### Using pandoc (recommended)

```bash
# Install pandoc
sudo apt-get install pandoc texlive-latex-base

# Convert to PDF
pandoc outputs/report.md -o outputs/report.pdf \
  --pdf-engine=pdflatex \
  -V geometry:margin=1in
```

### Using grip + browser

```bash
# Install grip
pip install grip

# Preview in browser
grip outputs/report.md

# Then print to PDF from browser
```

### Using markdown-pdf

```bash
# Install markdown-pdf
npm install -g markdown-pdf

# Convert
markdown-pdf outputs/report.md
```

## Architecture

The report generator follows this pipeline:

```
World Model
    ↓
Extract High-Confidence Findings (>0.7)
    ↓
Assess Novelty (literature contradictions, questions answered)
    ↓
Group into Discoveries (graph connectivity)
    ↓
Generate Narratives (Claude API or templates)
    ↓
Add Citations (papers + code)
    ↓
Format as Markdown
    ↓
Write Report + Appendix
```

## Testing

Run tests with:

```bash
python -m pytest tests/test_report_generator.py -v
```

## Future Enhancements

- [ ] Support for LaTeX output
- [ ] Interactive HTML reports with graphs
- [ ] Citation style options (APA, MLA, etc.)
- [ ] Figure generation from analysis code
- [ ] Multi-language support
- [ ] Report templates for different domains
