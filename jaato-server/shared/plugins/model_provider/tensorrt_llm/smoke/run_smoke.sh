#!/usr/bin/env bash
# Bootstrap a workspace from the .jaato.example/ templates, fill in the
# REMOTE_HOST + MODEL_ID placeholders, and run the smoke harness.
#
# Idempotent — re-running with different --host / --model overwrites
# the workspace's profile copies with fresh templates before sed.
#
# Example:
#   ./run_smoke.sh --host http://192.168.1.50:8000 \
#                  --model meta-llama/Llama-3.1-8B-Instruct
#
#   ./run_smoke.sh --host http://192.168.1.50:8000 \
#                  --model Qwen/Qwen2.5-7B-Instruct \
#                  --scenario tools
#
# Pre-reqs:
#   - jaato daemon already running at /tmp/jaato.sock
#   - python executable available (defaults to <repo>/.venv/bin/python)

set -euo pipefail

SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# smoke/ → tensorrt_llm/ → model_provider/ → plugins/ → shared/ → jaato-server/ → repo root
JAATO_REPO="$(cd "$SMOKE_DIR/../../../../../.." && pwd)"

# Defaults
WORKSPACE="/tmp/jaato-tensorrt-smoke"
PYTHON="${PYTHON:-$JAATO_REPO/.venv/bin/python}"
SCENARIO="chat"

HOST=""
MODEL=""

usage() {
    cat <<USAGE
Usage: $0 --host <url> --model <id> [options]

Required:
  --host <url>        trtllm-serve or Triton OpenAI URL (e.g. http://192.168.1.10:8000)
  --model <id>        Model id returned by GET /v1/models on the remote endpoint

Options:
  --workspace <path>  Workspace path (default: $WORKSPACE)
  --scenario <name>   "chat" or "tools" (default: $SCENARIO)
  --python <path>     Python executable to run the harness with
                      (default: $JAATO_REPO/.venv/bin/python; also honors \$PYTHON env)
  -h, --help          Show this help

Behavior:
  1. mkdir the workspace + .jaato/{profiles,agents}/
  2. Copy templates from $SMOKE_DIR/.jaato.example/
  3. sed the host + model placeholders
  4. cd into the workspace and run smoke.py (or smoke_tools.py)
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)      HOST="$2";      shift 2 ;;
        --model)     MODEL="$2";     shift 2 ;;
        --workspace) WORKSPACE="$2"; shift 2 ;;
        --scenario)  SCENARIO="$2";  shift 2 ;;
        --python)    PYTHON="$2";    shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

[[ -z "$HOST"  ]] && { echo "ERROR: --host is required"  >&2; usage >&2; exit 1; }
[[ -z "$MODEL" ]] && { echo "ERROR: --model is required" >&2; usage >&2; exit 1; }

case "$SCENARIO" in
    chat)  HARNESS="smoke.py" ;;
    tools) HARNESS="smoke_tools.py" ;;
    *) echo "ERROR: --scenario must be 'chat' or 'tools', got: $SCENARIO" >&2; exit 1 ;;
esac

if ! [[ -x "$PYTHON" ]]; then
    echo "ERROR: python executable not found or not executable: $PYTHON" >&2
    echo "Override with --python <path> or PYTHON=<path>" >&2
    exit 1
fi

echo "==> Bootstrapping workspace at $WORKSPACE"
mkdir -p "$WORKSPACE/.jaato/profiles" "$WORKSPACE/.jaato/agents"
cp -f "$SMOKE_DIR/.jaato.example/profiles/"*.json "$WORKSPACE/.jaato/profiles/"
cp -f "$SMOKE_DIR/.jaato.example/agents/"*.md "$WORKSPACE/.jaato/agents/"

echo "==> Filling placeholders (host=$HOST, model=$MODEL)"
# Use | as sed delimiter so URLs containing / don't break the substitution
sed -i "s|http://REMOTE_HOST:8000|$HOST|g" "$WORKSPACE/.jaato/profiles/"*.json
sed -i "s|REPLACE_WITH_MODEL_ID_FROM_v1_models|$MODEL|g" "$WORKSPACE/.jaato/profiles/"*.json

echo "==> Running $HARNESS from $WORKSPACE"
cd "$WORKSPACE"
exec "$PYTHON" "$SMOKE_DIR/$HARNESS"
