#!/bin/bash
#
# Build pipeline for Jaato User Guide PDF
#
# This script:
# 1. Extracts content from codebase (commands, env vars, keybindings)
# 2. Generates screenshots (optional)
# 3. Converts each chapter markdown to LaTeX individually
# 4. Assembles master document and builds PDF with xelatex
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
    echo -e "${YELLOW}[0/5]${NC} Auto-generating documentation with AI..."

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
    echo -e "${YELLOW}[0/5]${NC} Skipping auto-documentation"
fi

# Step 1: Extract content from codebase
if [ "$SKIP_EXTRACTION" = false ]; then
    echo -e "${YELLOW}[1/5]${NC} Extracting content from codebase..."
    "$VENV_PYTHON" "$SCRIPT_DIR/generate_docs.py"
    echo -e "${GREEN}✓${NC} Content extracted"
else
    echo -e "${YELLOW}[1/5]${NC} Skipping content extraction"
fi

# Step 2: Generate screenshots
if [ "$SKIP_SCREENSHOTS" = false ]; then
    echo -e "${YELLOW}[2/5]${NC} Generating screenshots..."
    if command -v pexpect &> /dev/null; then
        "$VENV_PYTHON" "$SCRIPT_DIR/generate_screenshots.py" \
            --output-dir "$ASSETS_DIR/screenshots" \
            --venv-python "$VENV_PYTHON"
        echo -e "${GREEN}✓${NC} Screenshots generated"
    else
        echo -e "${YELLOW}⚠${NC} pexpect not available, skipping screenshots"
    fi
else
    echo -e "${YELLOW}[2/5]${NC} Skipping screenshot generation"
fi

# Step 3: Convert each chapter to LaTeX individually
echo -e "${YELLOW}[3/5]${NC} Converting chapters to LaTeX..."

# Generate pandoc highlighting macros (needed by master.tex)
TMPL=$(mktemp --suffix=.latex)
echo '$highlighting-macros$' > "$TMPL"
echo '```python
x=1
```' | pandoc --from markdown --to latex --highlight-style=tango \
    --template "$TMPL" \
    > "$BUILD_DIR/highlighting-macros.tex" 2>/dev/null
rm -f "$TMPL"

echo "  Generated highlighting-macros.tex"

# Convert each chapter markdown to a .tex fragment (body only, no standalone)
CHAPTER_INPUTS=""
for file in "$DOCS_DIR"/chapters/*.md; do
    if [ -f "$file" ]; then
        basename=$(basename "$file" .md)
        texfile="$BUILD_DIR/${basename}.tex"
        echo "  Converting: $(basename "$file") → ${basename}.tex"
        pandoc "$file" \
            --from markdown \
            --to latex \
            --number-sections \
            --highlight-style=tango \
            --output "$texfile"
        CHAPTER_INPUTS="${CHAPTER_INPUTS}\\input{${basename}}\n"
    fi
done

# Convert generated content
for file in "$GENERATED_DIR"/*.md; do
    if [ -f "$file" ]; then
        basename=$(basename "$file" .md)
        texfile="$BUILD_DIR/${basename}.tex"
        echo "  Converting: $(basename "$file") → ${basename}.tex"
        pandoc "$file" \
            --from markdown \
            --to latex \
            --number-sections \
            --highlight-style=tango \
            --output "$texfile"
        CHAPTER_INPUTS="${CHAPTER_INPUTS}\\input{${basename}}\n"
    fi
done

# Write chapters.tex with all \input lines
echo -e "$CHAPTER_INPUTS" > "$BUILD_DIR/chapters.tex"
echo "  Generated chapters.tex"

echo -e "${GREEN}✓${NC} Chapters converted"

# Step 4: Build PDF
echo -e "${YELLOW}[4/5]${NC} Building PDF..."

# Copy master.tex to build directory
cp "$DOCS_DIR/latex/master.tex" "$BUILD_DIR/user-guide.tex"

cd "$BUILD_DIR"

# Run xelatex multiple times for TOC and cross-references
# xelatex returns non-zero for warnings, so check PDF output instead of exit code
echo "  Running xelatex (1/3)..."
xelatex -interaction=nonstopmode user-guide.tex > xelatex.log 2>&1 || true
if [ ! -f "user-guide.pdf" ]; then
    echo -e "${RED}✗${NC} First pass produced no PDF, check build/xelatex.log"
    tail -n 50 xelatex.log
    exit 1
fi

echo "  Running xelatex (2/3)..."
xelatex -interaction=nonstopmode user-guide.tex > xelatex.log 2>&1 || true

echo "  Running xelatex (3/3)..."
xelatex -interaction=nonstopmode user-guide.tex > xelatex.log 2>&1 || true

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

# Step 5: Optional: Generate HTML
if [ "$OUTPUT_HTML" = true ]; then
    echo ""
    echo -e "${YELLOW}[5/5]${NC} Generating HTML version..."

    # Combine markdown for HTML (HTML doesn't need per-chapter splitting)
    COMBINED_MD="$BUILD_DIR/user-guide-combined.md"
    > "$COMBINED_MD"
    for file in "$DOCS_DIR"/chapters/*.md; do
        if [ -f "$file" ]; then
            cat "$file" >> "$COMBINED_MD"
            echo "" >> "$COMBINED_MD"
        fi
    done
    for file in "$GENERATED_DIR"/*.md; do
        if [ -f "$file" ]; then
            cat "$file" >> "$COMBINED_MD"
            echo "" >> "$COMBINED_MD"
        fi
    done

    pandoc "$COMBINED_MD" \
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
