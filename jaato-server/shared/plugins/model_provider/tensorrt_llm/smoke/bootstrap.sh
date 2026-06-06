#!/usr/bin/env bash
# Bootstrap a self-contained smoke install at the target workspace.
#
# Primary purpose: copies the smoke harness scripts + profile/agent
# templates from the repo into the workspace, then sed-replaces the
# REMOTE_HOST + MODEL_ID placeholders.
#
# Optional: with --run chat | --run tools, also invokes the harness
# from the workspace dir after bootstrap (cwd-fallback resolves
# workspace_path).  Without --run, the script stops after bootstrap
# and the user runs the smoke themselves.
#
# Idempotent: re-running with different --host / --model overwrites
# the workspace's copies with fresh templates before sed.
#
# Examples:
#   # Bootstrap only — user runs the smoke themselves.
#   ./bootstrap.sh --host http://192.168.1.50:8000 \
#                  --model Qwen/Qwen2.5-7B-Instruct
#
#   # Bootstrap + immediately run the chat smoke.
#   ./bootstrap.sh --host http://192.168.1.50:8000 \
#                  --model Qwen/Qwen2.5-7B-Instruct \
#                  --run chat
#
#   # Bootstrap + run the tools smoke.
#   ./bootstrap.sh --host ... --model ... --run tools

set -euo pipefail

SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# smoke/ → tensorrt_llm/ → model_provider/ → plugins/ → shared/ → jaato-server/ → repo root
JAATO_REPO="$(cd "$SMOKE_DIR/../../../../../.." && pwd)"

# Defaults
WORKSPACE="/tmp/jaato-tensorrt-smoke"
PYTHON="${PYTHON:-$JAATO_REPO/.venv/bin/python}"
RUN=""

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
  --run <scenario>    After bootstrap, run the smoke. Scenario is "chat" or "tools".
                      Omit to bootstrap only.
  --python <path>     Python executable used when --run is set
                      (default: $JAATO_REPO/.venv/bin/python; also honors \$PYTHON env)
  -h, --help          Show this help

Behavior (always):
  1. mkdir the workspace + .jaato/{profiles,agents}/
  2. Copy smoke.py + smoke_tools.py to the workspace root
  3. Copy profile + agent templates to .jaato/{profiles,agents}/
  4. sed the host + model placeholders in the profile copies

Behavior (only with --run):
  5. cd into the workspace and exec the chosen harness

The daemon must be listening on /tmp/jaato.sock before running the smoke
(either before invoking with --run, or before the manual run step).
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)      HOST="$2";      shift 2 ;;
        --model)     MODEL="$2";     shift 2 ;;
        --workspace) WORKSPACE="$2"; shift 2 ;;
        --run)       RUN="$2";       shift 2 ;;
        --python)    PYTHON="$2";    shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

[[ -z "$HOST"  ]] && { echo "ERROR: --host is required"  >&2; usage >&2; exit 1; }
[[ -z "$MODEL" ]] && { echo "ERROR: --model is required" >&2; usage >&2; exit 1; }

HARNESS=""
case "$RUN" in
    "")    HARNESS="" ;;
    chat)  HARNESS="smoke.py" ;;
    tools) HARNESS="smoke_tools.py" ;;
    *) echo "ERROR: --run must be 'chat' or 'tools', got: $RUN" >&2; exit 1 ;;
esac

if [[ -n "$HARNESS" ]] && ! [[ -x "$PYTHON" ]]; then
    echo "ERROR: python executable not found or not executable: $PYTHON" >&2
    echo "Override with --python <path> or PYTHON=<path>" >&2
    exit 1
fi

echo "==> Bootstrapping smoke install at $WORKSPACE"
mkdir -p "$WORKSPACE/.jaato/profiles" "$WORKSPACE/.jaato/agents"

echo "==> Copying harness scripts"
cp -f "$SMOKE_DIR/smoke.py" "$SMOKE_DIR/smoke_tools.py" "$WORKSPACE/"

echo "==> Copying profile + agent templates"
cp -f "$SMOKE_DIR/.jaato.example/profiles/"*.json "$WORKSPACE/.jaato/profiles/"
cp -f "$SMOKE_DIR/.jaato.example/agents/"*.md "$WORKSPACE/.jaato/agents/"

echo "==> Filling placeholders (host=$HOST, model=$MODEL)"
# Use | as sed delimiter so URLs containing / don't break the substitution
sed -i "s|http://REMOTE_HOST:8000|$HOST|g" "$WORKSPACE/.jaato/profiles/"*.json
sed -i "s|REPLACE_WITH_MODEL_ID_FROM_v1_models|$MODEL|g" "$WORKSPACE/.jaato/profiles/"*.json

if [[ -n "$HARNESS" ]]; then
    echo "==> Running $HARNESS from $WORKSPACE"
    cd "$WORKSPACE"
    exec "$PYTHON" "$WORKSPACE/$HARNESS"
fi

cat <<NEXT

Bootstrap complete. To run the smoke:

  cd $WORKSPACE
  $PYTHON smoke.py         # chat smoke
  $PYTHON smoke_tools.py   # tools smoke

(Or re-invoke this script with --run chat / --run tools.)
(Daemon must already be listening on /tmp/jaato.sock.)
NEXT
