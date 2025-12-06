#!/bin/bash
# Record all plugin demos with termtosvg
#
# Prerequisites:
#   pip install pexpect pyyaml termtosvg
#   Valid .env file with API credentials
#
# Usage:
#   ./record_all.sh                           # Record all demos
#   ./record_all.sh shared/plugins/cli        # Record specific plugin demo

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check prerequisites
if ! command -v termtosvg &> /dev/null; then
    echo "Error: termtosvg not found. Install with: pip install termtosvg"
    exit 1
fi

python -c "import pexpect" 2>/dev/null || {
    echo "Error: pexpect not found. Install with: pip install pexpect"
    exit 1
}

python -c "import yaml" 2>/dev/null || {
    echo "Error: pyyaml not found. Install with: pip install pyyaml"
    exit 1
}

if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "Warning: .env file not found. Demos require valid API credentials."
fi

record_demo() {
    local yaml_file=$1
    local plugin_dir=$(dirname "$yaml_file")
    local output_file="$plugin_dir/demo.svg"

    echo "=============================================="
    echo "Recording: $yaml_file"
    echo "Output: $output_file"
    echo "=============================================="

    termtosvg \
        -c "python $SCRIPT_DIR/run_demo.py $yaml_file" \
        -g "100x40" \
        "$output_file"

    # Make animation play only once (not loop indefinitely)
    if [ -f "$output_file" ]; then
        sed -i 's/repeatCount="indefinite"/repeatCount="1"/g' "$output_file"
        echo "Saved: $output_file (single play)"
    fi
    echo ""
}

# Find demo scripts to record
if [ $# -eq 0 ]; then
    # Find all demo.yaml files in plugin directories + root demo
    DEMO_FILES=$(find shared/plugins -name "demo.yaml" -type f 2>/dev/null)

    # Also include the main README demo if it exists
    if [ -f "demo.yaml" ]; then
        DEMO_FILES="demo.yaml $DEMO_FILES"
    fi

    if [ -z "$DEMO_FILES" ]; then
        echo "No demo.yaml files found"
        exit 1
    fi
else
    # Use provided paths
    DEMO_FILES=""
    for arg in "$@"; do
        if [ -f "$arg/demo.yaml" ]; then
            DEMO_FILES="$DEMO_FILES $arg/demo.yaml"
        elif [ -f "$arg" ]; then
            DEMO_FILES="$DEMO_FILES $arg"
        else
            echo "Error: Not found: $arg or $arg/demo.yaml"
            exit 1
        fi
    done
fi

# Record each demo
for yaml_file in $DEMO_FILES; do
    record_demo "$yaml_file"
done

echo "=============================================="
echo "Recording complete!"
echo "=============================================="
