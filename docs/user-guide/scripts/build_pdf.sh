#!/bin/bash
#
# Build pipeline for Jaato User Guide PDF
#
# This script:
# 1. Extracts content from codebase (commands, env vars, keybindings)
# 2. Generates screenshots (optional)
# 3. Converts markdown to LaTeX
# 4. Builds PDF with pdflatex
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$DOCS_DIR")")"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"

# Output directories
BUILD_DIR="${DOCS_DIR}/build"
GENERATED_DIR="${DOCS_DIR}/generated"
ASSETS_DIR="${DOCS_DIR}/assets"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Options
SKIP_SCREENSHOTS=false
SKIP_EXTRACTION=false
SKIP_AUTODOC=false
FORCE_AUTODOC=false
OUTPUT_HTML=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-screenshots)
            SKIP_SCREENSHOTS=true
            shift
            ;;
        --skip-extraction)
            SKIP_EXTRACTION=true
            shift
            ;;
        --skip-autodoc)
            SKIP_AUTODOC=true
            shift
            ;;
        --force-autodoc)
            FORCE_AUTODOC=true
            shift
            ;;
        --html)
            OUTPUT_HTML=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Jaato User Guide Build Pipeline${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""

# Create directories
mkdir -p "$BUILD_DIR" "$GENERATED_DIR" "$ASSETS_DIR/screenshots"

# Step 0: Auto-generate/update documentation using AI
if [ "$SKIP_AUTODOC" = false ]; then
    echo -e "${YELLOW}[0/6]${NC} Auto-generating documentation with AI..."

    AUTODOC_ARGS=""
    if [ "$FORCE_AUTODOC" = true ]; then
        AUTODOC_ARGS="--force"
    fi

    "$VENV_PYTHON" "$SCRIPT_DIR/auto_document.py" $AUTODOC_ARGS

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Documentation auto-generated"
    else
        echo -e "${YELLOW}⚠${NC} Auto-generation had issues, continuing with existing content"
    fi
else
    echo -e "${YELLOW}[0/6]${NC} Skipping auto-documentation"
fi

# Step 1: Extract content from codebase
if [ "$SKIP_EXTRACTION" = false ]; then
    echo -e "${YELLOW}[1/6]${NC} Extracting content from codebase..."
    "$VENV_PYTHON" "$SCRIPT_DIR/generate_docs.py"
    echo -e "${GREEN}✓${NC} Content extracted"
else
    echo -e "${YELLOW}[1/6]${NC} Skipping content extraction"
fi

# Step 2: Generate screenshots
if [ "$SKIP_SCREENSHOTS" = false ]; then
    echo -e "${YELLOW}[2/6]${NC} Generating screenshots..."
    if command -v pexpect &> /dev/null; then
        "$VENV_PYTHON" "$SCRIPT_DIR/generate_screenshots.py" \
            --output-dir "$ASSETS_DIR/screenshots" \
            --venv-python "$VENV_PYTHON"
        echo -e "${GREEN}✓${NC} Screenshots generated"
    else
        echo -e "${YELLOW}⚠${NC} pexpect not available, skipping screenshots"
    fi
else
    echo -e "${YELLOW}[2/6]${NC} Skipping screenshot generation"
fi

# Step 3: Combine markdown files
echo -e "${YELLOW}[3/6]${NC} Combining markdown files..."

# Concatenate all markdown in order
cat > "$BUILD_DIR/user-guide-combined.md" << 'EOF'
---
title: "Jaato Rich Client User Guide"
author: "Jaato Development Team"
date: \today
documentclass: book
geometry: margin=1in
fontsize: 11pt
colorlinks: true
linkcolor: blue
urlcolor: blue
toccolor: black
toc: true
toc-depth: 3
numbersections: true
---

EOF

# Add each section
for file in "$DOCS_DIR"/chapters/*.md; do
    if [ -f "$file" ]; then
        echo "Adding: $(basename "$file")"
        cat "$file" >> "$BUILD_DIR/user-guide-combined.md"
        echo -e "\n\\newpage\n" >> "$BUILD_DIR/user-guide-combined.md"
    fi
done

# Add generated content
echo "Adding generated content..."
for file in "$GENERATED_DIR"/*.md; do
    if [ -f "$file" ]; then
        echo "  - $(basename "$file")"
        cat "$file" >> "$BUILD_DIR/user-guide-combined.md"
        echo -e "\n\\newpage\n" >> "$BUILD_DIR/user-guide-combined.md"
    fi
done

echo -e "${GREEN}✓${NC} Markdown combined"

# Step 4: Convert to LaTeX
echo -e "${YELLOW}[4/6]${NC} Converting to LaTeX..."

pandoc "$BUILD_DIR/user-guide-combined.md" \
    --from markdown \
    --to latex \
    --output "$BUILD_DIR/user-guide.tex" \
    --template="$DOCS_DIR/latex/template.tex" \
    --listings \
    --number-sections \
    --toc \
    --variable=geometry:margin=1in \
    --variable=fontsize:11pt \
    --variable=documentclass:book \
    --variable=papersize:letter \
    --variable=classoption:openany \
    --highlight-style=tango \
    --pdf-engine=pdflatex

echo -e "${GREEN}✓${NC} LaTeX generated"

# Step 5: Build PDF
echo -e "${YELLOW}[5/6]${NC} Building PDF..."

cd "$BUILD_DIR"

# Run pdflatex multiple times for references
echo "  Running pdflatex (1/3)..."
pdflatex -interaction=nonstopmode user-guide.tex > pdflatex.log 2>&1 || {
    echo -e "${RED}✗${NC} First pass failed, check build/pdflatex.log"
    tail -n 50 pdflatex.log
    exit 1
}

echo "  Running pdflatex (2/3)..."
pdflatex -interaction=nonstopmode user-guide.tex > pdflatex.log 2>&1

echo "  Running pdflatex (3/3)..."
pdflatex -interaction=nonstopmode user-guide.tex > pdflatex.log 2>&1

if [ -f "user-guide.pdf" ]; then
    echo -e "${GREEN}✓${NC} PDF built successfully"
    echo ""
    echo -e "${GREEN}Output:${NC} $BUILD_DIR/user-guide.pdf"

    # Copy to docs root for easy access
    cp user-guide.pdf "$DOCS_DIR/user-guide.pdf"
    echo -e "${GREEN}Copied:${NC} $DOCS_DIR/user-guide.pdf"

    # Get file size
    SIZE=$(du -h user-guide.pdf | cut -f1)
    PAGES=$(pdfinfo user-guide.pdf 2>/dev/null | grep Pages | awk '{print $2}')
    echo -e "${GREEN}Size:${NC} $SIZE ($PAGES pages)"
else
    echo -e "${RED}✗${NC} PDF generation failed"
    exit 1
fi

# Step 6: Optional: Generate HTML
if [ "$OUTPUT_HTML" = true ]; then
    echo ""
    echo -e "${YELLOW}[6/6]${NC} Generating HTML version..."

    pandoc "$BUILD_DIR/user-guide-combined.md" \
        --from markdown \
        --to html5 \
        --output "$BUILD_DIR/user-guide.html" \
        --standalone \
        --toc \
        --toc-depth=3 \
        --css="$DOCS_DIR/latex/style.css" \
        --highlight-style=tango \
        --metadata title="Jaato Rich Client User Guide"

    if [ -f "$BUILD_DIR/user-guide.html" ]; then
        echo -e "${GREEN}✓${NC} HTML generated: $BUILD_DIR/user-guide.html"
        cp "$BUILD_DIR/user-guide.html" "$DOCS_DIR/user-guide.html"
    fi
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Build complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
