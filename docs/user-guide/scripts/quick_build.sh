#!/bin/bash
#
# Quick build script for common documentation workflows
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_help() {
    cat << EOF
Jaato Documentation Quick Build

Usage: $0 <command> [options]

Commands:
    full            Full build (auto-doc + extract + PDF)
    fast            Fast build (skip auto-doc and screenshots)
    autodoc         Only run auto-documentation
    chapter <name>  Generate/update specific chapter
    pdf             Only build PDF from existing markdown
    html            Build both PDF and HTML
    clean           Remove build artifacts

Examples:
    $0 full                          # Complete rebuild
    $0 fast                          # Quick iteration
    $0 chapter 06-basic-commands.md  # Update one chapter
    $0 autodoc --force               # Regenerate all chapters
    $0 pdf                           # Just compile PDF

Environment Variables:
    JAATO_DOC_PROVIDER   AI provider (default: google_genai)
    JAATO_DOC_MODEL      Model name (default: gemini-2.5-flash)

EOF
}

case "$1" in
    full)
        echo -e "${BLUE}══════════════════════════════════════${NC}"
        echo -e "${BLUE}  Full Documentation Build${NC}"
        echo -e "${BLUE}══════════════════════════════════════${NC}"
        "$SCRIPT_DIR/build_pdf.sh"
        ;;

    fast)
        echo -e "${BLUE}══════════════════════════════════════${NC}"
        echo -e "${BLUE}  Fast Documentation Build${NC}"
        echo -e "${BLUE}══════════════════════════════════════${NC}"
        "$SCRIPT_DIR/build_pdf.sh" --skip-autodoc --skip-screenshots
        ;;

    autodoc)
        shift
        echo -e "${BLUE}══════════════════════════════════════${NC}"
        echo -e "${BLUE}  Auto-Documentation${NC}"
        echo -e "${BLUE}══════════════════════════════════════${NC}"

        PROVIDER="${JAATO_DOC_PROVIDER:-google_genai}"
        MODEL="${JAATO_DOC_MODEL:-gemini-2.5-flash}"

        "$SCRIPT_DIR/auto_document.py" \
            --provider "$PROVIDER" \
            --model "$MODEL" \
            "$@"
        ;;

    chapter)
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Error: chapter name required${NC}"
            echo "Usage: $0 chapter <filename>"
            exit 1
        fi

        echo -e "${BLUE}══════════════════════════════════════${NC}"
        echo -e "${BLUE}  Generate Chapter: $2${NC}"
        echo -e "${BLUE}══════════════════════════════════════${NC}"

        PROVIDER="${JAATO_DOC_PROVIDER:-google_genai}"
        MODEL="${JAATO_DOC_MODEL:-gemini-2.5-flash}"

        "$SCRIPT_DIR/auto_document.py" \
            --provider "$PROVIDER" \
            --model "$MODEL" \
            --chapter "$2"
        ;;

    pdf)
        echo -e "${BLUE}══════════════════════════════════════${NC}"
        echo -e "${BLUE}  PDF Build Only${NC}"
        echo -e "${BLUE}══════════════════════════════════════${NC}"
        "$SCRIPT_DIR/build_pdf.sh" --skip-autodoc --skip-extraction --skip-screenshots
        ;;

    html)
        echo -e "${BLUE}══════════════════════════════════════${NC}"
        echo -e "${BLUE}  PDF + HTML Build${NC}"
        echo -e "${BLUE}══════════════════════════════════════${NC}"
        "$SCRIPT_DIR/build_pdf.sh" --html
        ;;

    clean)
        echo -e "${BLUE}══════════════════════════════════════${NC}"
        echo -e "${BLUE}  Clean Build Artifacts${NC}"
        echo -e "${BLUE}══════════════════════════════════════${NC}"

        BUILD_DIR="$(dirname "$SCRIPT_DIR")/build"
        GENERATED_DIR="$(dirname "$SCRIPT_DIR")/generated"

        rm -rf "$BUILD_DIR"
        echo -e "${GREEN}✓${NC} Removed $BUILD_DIR"

        # Optionally clean generated content (be careful!)
        read -p "Also remove generated/ directory? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$GENERATED_DIR"
            echo -e "${GREEN}✓${NC} Removed $GENERATED_DIR"
        fi
        ;;

    help|--help|-h)
        show_help
        ;;

    *)
        echo -e "${YELLOW}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
