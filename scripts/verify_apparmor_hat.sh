#!/usr/bin/env bash
# verify_apparmor_hat.sh
#
# Empirical verification of the proposed 0.6.55 hat segregation
# (per-tenant integrity write-deny on user-authored config + tool-hat
# read-deny on the same set, with reads otherwise open and plugin-
# runtime subdirs writable everywhere).
#
# What this script proves:
#
#   1. ``base`` context (no hat):
#       - reads of ``.jaato/agents/<x>.md`` succeed (framework loading)
#       - writes to ``.jaato/agents/<x>.md`` are denied (integrity)
#       - reads + writes to ``.jaato/sessions/`` succeed (plugin runtime)
#
#   2. ``tool_hat`` context (hat-stacked):
#       - reads of ``.jaato/agents/<x>.md`` are denied (info-isolation)
#       - reads + writes to ``.jaato/sessions/`` succeed (plugin runtime
#         inherits from base — confirms additive hat semantics)
#       - writes outside ``.jaato/`` succeed (workspace rwkl inherits)
#       - writes to ``.jaato/agents/`` denied (integrity inherits)
#
# Run as: sudo ./scripts/verify_apparmor_hat.sh
# (sudo needed for apparmor_parser).
#
# Exit code: 0 if all assertions pass, 1 otherwise.  Per-test
# pass/fail is printed inline.
#
# Cleanup: removes the test profile + tmpdir on exit.  If the script
# is killed mid-run, ``apparmor_parser -R`` may need a manual call;
# ``sudo apparmor_parser -R /tmp/jaato-hat-verify-*/profile`` clears it.

set -euo pipefail

if [[ "$EUID" -ne 0 ]]; then
    echo "ERROR: must run as root (need apparmor_parser)" >&2
    exit 1
fi

TMPDIR=$(mktemp -d -t jaato-hat-verify-XXXXXX)
PROFILE_NAME="jaato-hat-verify-$$"
WS="$TMPDIR/workspace"
PROFILE_FILE="$TMPDIR/$PROFILE_NAME"

cleanup() {
    apparmor_parser -R "$PROFILE_FILE" 2>/dev/null || true
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

# ---- fixture: workspace with .jaato structure ----
mkdir -p "$WS/.jaato/agents"
mkdir -p "$WS/.jaato/profiles"
mkdir -p "$WS/.jaato/scripts"
mkdir -p "$WS/.jaato/sessions"
mkdir -p "$WS/.jaato/logs"
mkdir -p "$WS/.jaato/services/_discovered"
mkdir -p "$WS/.jaato/services/sinco"
mkdir -p "$WS/.jaato/memory"
mkdir -p "$WS/output"

echo "agent persona content" > "$WS/.jaato/agents/test_agent.md"
echo "{\"name\": \"test\"}"   > "$WS/.jaato/profiles/test_profile.json"
echo "def render(c, a): pass" > "$WS/.jaato/scripts/test_prefetch.py"
echo "{\"id\": \"sess-1\"}"   > "$WS/.jaato/sessions/sess-1.json"
echo "log line"               > "$WS/.jaato/logs/session_x.log"
echo "{\"discovered\": {}}"   > "$WS/.jaato/services/_discovered/cache.json"
echo "name: sinco"            > "$WS/.jaato/services/sinco/_service.yaml"
echo "{\"memories\": []}"     > "$WS/.jaato/memory/m1.json"

# ---- profile with the proposed shape ----
cat > "$PROFILE_FILE" <<EOF
#include <tunables/global>

profile $PROFILE_NAME flags=(attach_disconnected) {
  #include <abstractions/base>

  # Workspace rw + integrity write-deny on user-authored config.
  # Applies in both base and hat contexts.
  $WS/   rw,
  $WS/** rwkl,
  audit deny $WS/.jaato/agents/**             w,
  audit deny $WS/.jaato/profiles/**           w,
  audit deny $WS/.jaato/prompts/**            w,
  audit deny $WS/.jaato/scripts/**            w,
  audit deny $WS/.jaato/services/*/           w,
  audit deny $WS/.jaato/reactors.json         w,
  audit deny $WS/.jaato/completion_schemas/** w,
  audit deny $WS/.jaato/spawn_schemas/**      w,
  audit deny $WS/.jaato/instructions/**       w,
  audit deny $WS/.jaato/references/**         w,

  # Per-thread profile transition support — same as production template.
  change_profile -> unconfined,
  /proc/self/attr/current      w,
  /proc/self/task/*/attr/current w,
  capability sys_admin,

  # Tool execution sub-profile (not a hat).
  #
  # IMPORTANT — empirically confirmed (2026-05-06): AppArmor hats do
  # NOT inherit the base profile's rules; the hat's ruleset replaces
  # the base's.  Hats are also entered via aa_change_hat(2) (not
  # change_profile), which doesn't compose with aa-exec testing.
  # Sub-profiles solve both problems: they're entered via
  # change_profile, support aa-exec via the parent//child syntax, and
  # have the same self-contained ruleset semantics (must redeclare all
  # allows + denies they want to honor).
  profile tool_hat {
    #include <abstractions/base>

    # Workspace allow (redeclared from base)
    $WS/   rw,
    $WS/** rwkl,

    # Integrity write-denies (redeclared from base — same set)
    audit deny $WS/.jaato/agents/**             w,
    audit deny $WS/.jaato/profiles/**           w,
    audit deny $WS/.jaato/prompts/**            w,
    audit deny $WS/.jaato/scripts/**            w,
    audit deny $WS/.jaato/services/*/           w,
    audit deny $WS/.jaato/reactors.json         w,
    audit deny $WS/.jaato/completion_schemas/** w,
    audit deny $WS/.jaato/spawn_schemas/**      w,
    audit deny $WS/.jaato/instructions/**       w,
    audit deny $WS/.jaato/references/**         w,

    # Sub-profile-specific: info-isolation read-denies on user-authored
    # config (the new restriction on top of base's write-denies)
    audit deny $WS/.jaato/agents/**             r,
    audit deny $WS/.jaato/profiles/**           r,
    audit deny $WS/.jaato/prompts/**            r,
    audit deny $WS/.jaato/scripts/**            r,
    audit deny $WS/.jaato/completion_schemas/** r,
    audit deny $WS/.jaato/spawn_schemas/**      r,
    audit deny $WS/.jaato/instructions/**       r,
    audit deny $WS/.jaato/reactors.json         r,

    # Per-thread profile transition support (redeclared from base)
    change_profile -> unconfined,
    /proc/self/attr/current      w,
    /proc/self/task/*/attr/current w,
    capability sys_admin,
  }
}
EOF

# ---- load profile ----
apparmor_parser -r "$PROFILE_FILE"

PASS=0
FAIL=0

# Test runner: expects either ``allow`` or ``deny`` for the given command
# under the given profile context (base or hat).
#
# Usage: assert <expected> <context> <description> <command...>
# context is "base" or "hat".
assert() {
    local expected="$1"
    local context="$2"
    local description="$3"
    shift 3
    local profile_spec
    if [[ "$context" == "hat" ]]; then
        # Sub-profile syntax: parent//child enters the sub-profile via
        # change_profile.  Different from "//&hat" which is the older
        # change_hat invocation form (not supported by aa-exec).
        profile_spec="$PROFILE_NAME//tool_hat"
    else
        profile_spec="$PROFILE_NAME"
    fi
    local rc=0
    aa-exec -p "$profile_spec" -- "$@" >/dev/null 2>&1 || rc=$?
    local actual
    if [[ $rc -eq 0 ]]; then actual="allow"; else actual="deny"; fi
    if [[ "$actual" == "$expected" ]]; then
        echo "  PASS [$context] $description"
        PASS=$((PASS+1))
    else
        echo "  FAIL [$context] $description (expected=$expected, actual=$actual, rc=$rc)"
        FAIL=$((FAIL+1))
    fi
}

echo
echo "=== BASE context (no hat) — framework lifecycle, reactors, session-init ==="
assert allow base "read user-authored agent .md (framework loading)" \
    cat "$WS/.jaato/agents/test_agent.md"
assert allow base "read user-authored profile JSON" \
    cat "$WS/.jaato/profiles/test_profile.json"
assert allow base "read user-authored prefetch script" \
    cat "$WS/.jaato/scripts/test_prefetch.py"
assert deny base "write to user-authored agent .md (integrity)" \
    sh -c "echo poisoned > $WS/.jaato/agents/test_agent.md"
assert deny base "write to user-authored profile JSON" \
    sh -c "echo {} > $WS/.jaato/profiles/test_profile.json"
assert allow base "read plugin-runtime sessions journal" \
    cat "$WS/.jaato/sessions/sess-1.json"
assert allow base "write plugin-runtime sessions journal (daemon save)" \
    sh -c "echo updated > $WS/.jaato/sessions/sess-1.json"
assert allow base "write plugin-runtime log file" \
    sh -c "echo log >> $WS/.jaato/logs/session_x.log"
assert allow base "write workspace file outside .jaato" \
    sh -c "echo content > $WS/output/result.txt"

echo
echo "=== HAT context (tool_hat) — LLM-driven tool execution ==="
assert deny hat "read user-authored agent .md (info-isolation)" \
    cat "$WS/.jaato/agents/test_agent.md"
assert deny hat "read user-authored profile JSON" \
    cat "$WS/.jaato/profiles/test_profile.json"
assert deny hat "read user-authored prefetch script" \
    cat "$WS/.jaato/scripts/test_prefetch.py"
assert deny hat "write to user-authored agent .md (integrity inherits from base)" \
    sh -c "echo poisoned > $WS/.jaato/agents/test_agent.md"
assert allow hat "read plugin-runtime sessions journal (inherits from base)" \
    cat "$WS/.jaato/sessions/sess-1.json"
assert allow hat "write plugin-runtime sessions journal (inherits from base)" \
    sh -c "echo updated > $WS/.jaato/sessions/sess-1.json"
assert allow hat "read plugin-runtime memory state" \
    cat "$WS/.jaato/memory/m1.json"
assert allow hat "read plugin-runtime services/_discovered" \
    cat "$WS/.jaato/services/_discovered/cache.json"
assert allow hat "read curated services/<name>/" \
    cat "$WS/.jaato/services/sinco/_service.yaml"
assert allow hat "write workspace file outside .jaato (workspace rwkl inherits)" \
    sh -c "echo content > $WS/output/from_tool.txt"

echo
echo "=== Summary ==="
echo "PASS: $PASS"
echo "FAIL: $FAIL"
if [[ $FAIL -ne 0 ]]; then
    exit 1
fi
echo "All assertions passed — proposed 0.6.55 hat segregation works as designed."
