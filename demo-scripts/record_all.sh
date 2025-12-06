#!/bin/bash
# Record all plugin demos with termtosvg
#
# Prerequisites:
#   pip install pexpect termtosvg
#   Valid .env file with API credentials
#
# Usage:
#   ./record_all.sh              # Record all demos
#   ./record_all.sh cli          # Record specific demo
#   ./record_all.sh cli file_edit # Record multiple demos

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$SCRIPT_DIR/recordings"

cd "$PROJECT_ROOT"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Demo configurations: name, width, height
declare -A DEMO_CONFIGS=(
    ["cli"]="100x40"
    ["file_edit"]="100x45"
    ["web_search"]="100x40"
    ["todo"]="100x45"
    ["references"]="100x42"
    ["subagent"]="100x45"
)

record_demo() {
    local name=$1
    local geometry=${DEMO_CONFIGS[$name]}

    echo "=============================================="
    echo "Recording: $name ($geometry)"
    echo "=============================================="

    termtosvg \
        -c "python $SCRIPT_DIR/run_demo.py $name" \
        -g "$geometry" \
        "$OUTPUT_DIR/${name}_demo.svg"

    echo "Saved: $OUTPUT_DIR/${name}_demo.svg"
    echo ""
}

# Determine which demos to record
if [ $# -eq 0 ]; then
    # Record all demos
    DEMOS=("cli" "file_edit" "web_search" "todo" "references" "subagent")
else
    DEMOS=("$@")
fi

# Check prerequisites
if ! command -v termtosvg &> /dev/null; then
    echo "Error: termtosvg not found. Install with: pip install termtosvg"
    exit 1
fi

python -c "import pexpect" 2>/dev/null || {
    echo "Error: pexpect not found. Install with: pip install pexpect"
    exit 1
}

if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "Warning: .env file not found. Demos require valid API credentials."
fi

# Record requested demos
for demo in "${DEMOS[@]}"; do
    if [[ -v "DEMO_CONFIGS[$demo]" ]]; then
        record_demo "$demo"
    else
        echo "Unknown demo: $demo"
        echo "Available demos: ${!DEMO_CONFIGS[*]}"
        exit 1
    fi
done

echo "=============================================="
echo "Recording complete!"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="

# Optionally copy to plugin directories
read -p "Copy recordings to plugin demo directories? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    for demo in "${DEMOS[@]}"; do
        src="$OUTPUT_DIR/${demo}_demo.svg"
        if [ -f "$src" ]; then
            dest="$PROJECT_ROOT/shared/plugins/$demo/demo/demo.svg"
            mkdir -p "$(dirname "$dest")"
            cp "$src" "$dest"
            echo "Copied: $dest"
        fi
    done
fi
