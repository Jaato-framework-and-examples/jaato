#!/usr/bin/env bash
# Bootstrap a self-contained smoke install at the target workspace.
#
# Copies the smoke harness scripts + profile/agent templates from the
# repo into the workspace, then sed-replaces the REMOTE_HOST + MODEL_ID
# placeholders.  Does NOT run anything — the user runs the smoke
# themselves from the workspace (cwd-fallback handles workspace_path).
#
# Idempotent: re-running with different --host / --model overwrites
# the workspace's copies with fresh templates before sed.
#
# Example:
#   ./bootstrap.sh --host http://192.168.1.50:8000 \
#                  --model Qwen/Qwen2.5-7B-Instruct
#
# After bootstrap:
#   cd /tmp/jaato-tensorrt-smoke
#   <repo>/.venv/bin/python smoke.py        # chat smoke
#   <repo>/.venv/bin/python smoke_tools.py  # tools smoke

set -euo pipefail

SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
WORKSPACE="/tmp/jaato-tensorrt-smoke"
HOST=""
MODEL=""

usage() {
    cat <<USAGE
Usage: $0 --host <url> --model <id> [--workspace <path>]

Required:
  --host <url>        trtllm-serve or Triton OpenAI URL (e.g. http://192.168.1.10:8000)
  --model <id>        Model id returned by GET /v1/models on the remote endpoint

Options:
  --workspace <path>  Workspace path (default: $WORKSPACE)
  -h, --help          Show this help

Behavior:
  1. mkdir the workspace + .jaato/{profiles,agents}/
  2. Copy smoke.py + smoke_tools.py to the workspace root
  3. Copy profile + agent templates to .jaato/{profiles,agents}/
  4. sed the host + model placeholders in the profile copies
  5. Print a hint for running the smoke

The user then runs the smoke themselves from the workspace.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)      HOST="$2";      shift 2 ;;
        --model)     MODEL="$2";     shift 2 ;;
        --workspace) WORKSPACE="$2"; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

[[ -z "$HOST"  ]] && { echo "ERROR: --host is required"  >&2; usage >&2; exit 1; }
[[ -z "$MODEL" ]] && { echo "ERROR: --model is required" >&2; usage >&2; exit 1; }

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

cat <<NEXT

Bootstrap complete. To run the smoke:

  cd $WORKSPACE
  <repo>/.venv/bin/python smoke.py         # chat smoke
  <repo>/.venv/bin/python smoke_tools.py   # tools smoke

(Daemon must already be listening on /tmp/jaato.sock.)
NEXT
