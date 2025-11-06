#!/bin/bash
# Convert markdown report to PDF using pandoc

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "Error: pandoc is not installed"
    echo "Install with: sudo apt-get install pandoc texlive-latex-base"
    exit 1
fi

# Default values
INPUT_FILE="outputs/discovery_report.md"
OUTPUT_FILE="outputs/discovery_report.pdf"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-i input.md] [-o output.pdf]"
            echo ""
            echo "Options:"
            echo "  -i, --input   Input markdown file (default: outputs/discovery_report.md)"
            echo "  -o, --output  Output PDF file (default: outputs/discovery_report.pdf)"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h for help"
            exit 1
            ;;
    esac
done

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Convert to PDF
echo "Converting $INPUT_FILE to $OUTPUT_FILE..."

pandoc "$INPUT_FILE" -o "$OUTPUT_FILE" \
    --pdf-engine=pdflatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    -V documentclass=article \
    --highlight-style=tango

if [ $? -eq 0 ]; then
    echo "Success! PDF created: $OUTPUT_FILE"
    echo "File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
else
    echo "Error: PDF conversion failed"
    exit 1
fi
