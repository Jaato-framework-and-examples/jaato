#!/usr/bin/env bash
# Bootstrap a self-contained smoke install at the target workspace.
#
# Copies the smoke harness scripts, profile/agent templates, and
# .env.example into the workspace.  The workspace .env (where the
# user fills in ANTHROPIC_API_KEY) is only created when it does not
# already exist — re-running bootstrap will never clobber edits.
#
# Configuration model:
#   - Profile knobs (model, plugins, GC) live in the profile YAMLs.
#     Bake your choice in there; copy + edit the profile if you want
#     a variant.
#   - Deployment knobs (credentials) live in the workspace .env,
#     referenced from the profile as ${ANTHROPIC_API_KEY}.  Edit the
#     workspace .env after bootstrap.
#
# Optional: with --run chat | --run tools, also invokes the harness
# from the workspace dir after bootstrap (cwd-fallback resolves
# workspace_path).
#
# Examples:
#   # Bootstrap only — user runs the smoke themselves.
#   ./bootstrap.sh
#
#   # Bootstrap + immediately run the chat smoke.
#   ./bootstrap.sh --run chat
#
#   # Bootstrap + run the tools smoke into a non-default workspace.
#   ./bootstrap.sh --workspace /var/tmp/my-smoke --run tools

set -euo pipefail

SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# smoke/ → anthropic/ → model_provider/ → plugins/ → shared/ → jaato-server/ → repo root
JAATO_REPO="$(cd "$SMOKE_DIR/../../../../../.." && pwd)"

# Defaults
WORKSPACE="/tmp/jaato-anthropic-smoke"
PYTHON="${PYTHON:-$JAATO_REPO/.venv/bin/python}"
RUN=""

usage() {
    cat <<USAGE
Usage: $0 [options]

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
  4. Copy .env.example to <workspace>/.env if .env does not already exist

Behavior (only with --run):
  5. cd into the workspace and exec the chosen harness

After bootstrap, edit <workspace>/.env to set ANTHROPIC_API_KEY.
The daemon must be listening on /tmp/jaato.sock before running the smoke.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace) WORKSPACE="$2"; shift 2 ;;
        --run)       RUN="$2";       shift 2 ;;
        --python)    PYTHON="$2";    shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

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
cp -f "$SMOKE_DIR/.jaato.example/profiles/"*.yaml "$WORKSPACE/.jaato/profiles/"
cp -f "$SMOKE_DIR/.jaato.example/agents/"*.md "$WORKSPACE/.jaato/agents/"

if [[ -f "$WORKSPACE/.env" ]]; then
    echo "==> Preserving existing $WORKSPACE/.env (edit it if ANTHROPIC_API_KEY needs to change)"
else
    cp -f "$SMOKE_DIR/.env.example" "$WORKSPACE/.env"
    echo "==> Created $WORKSPACE/.env from .env.example — edit ANTHROPIC_API_KEY to set your credential"
fi

if [[ -n "$HARNESS" ]]; then
    echo "==> Running $HARNESS from $WORKSPACE"
    cd "$WORKSPACE"
    exec "$PYTHON" "$WORKSPACE/$HARNESS"
fi

cat <<NEXT

Bootstrap complete.  Next:

  1. Edit  $WORKSPACE/.env  to set ANTHROPIC_API_KEY.
  2. Run the smoke:
       cd $WORKSPACE
       $PYTHON smoke.py         # chat smoke
       $PYTHON smoke_tools.py   # tools smoke
     (Or re-invoke this script with --run chat / --run tools.)

(Daemon must already be listening on /tmp/jaato.sock.)
NEXT
