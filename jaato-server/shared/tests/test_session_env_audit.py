"""Audit test — guard against regressions on the workspace-tied env-var rule.

Per the framework principle (memory:
``feedback_session_env_from_workspace_dotenv_only.md``):

    Every per-session env var the framework reads (JAATO_PROFILE_SET,
    provider keys, ${SINCO_PORT}-style placeholders, etc.) must resolve
    via ``shared.session_context.get_session_env`` — which checks the
    per-session contextvar (populated from the SDK's ``env_file``)
    before falling back to ``os.environ``.

    Reading the daemon's ``os.environ`` directly breaks workspace
    isolation: two sessions on the same daemon get the same global
    value, switching profile-sets requires daemon restart, and
    workspace-scoped credentials leak across clients.

This test enforces the rule by **freezing** the current set of
``(file, env_var)`` direct-environ reads as a baseline allowlist.  The
test re-scans framework source on every run and fails on:

  - **NEW** entries (someone added a direct ``os.environ.get('JAATO_*')``
    call without going through ``get_session_env``).  The CI failure
    points the dev at the right API.

  - **REMOVED** entries (someone migrated a direct read to
    ``get_session_env``; great, but the baseline has stale entries the
    dev must delete to keep the allowlist tight).

The baseline is a snapshot of all current acknowledged reads.  Most are
genuinely boot-time / non-session (workspace-root resolution, retry
config, terminal capability detection) but a non-trivial subset is
session-tied work the team hasn't migrated yet.  Those are tech debt
the audit makes legible — the allowlist size shrinks as we migrate.

Scope: the audit only fires on env vars that *could plausibly* be
session-tied — currently the ``JAATO_*`` prefix plus a few legacy names
(``MODEL_NAME``, ``PROJECT_ID``, ``LOCATION``).  Generic OS / platform
vars (``HOME``, ``TERM``, etc.) are out of scope.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Set, Tuple


# ---------------------------------------------------------------------------
# Baseline allowlist — frozen 2026-05-01 after the JAATO_PROFILE_SET fix.
# Each entry is (relative_path_under_shared, env_var_name).
#
# Adding to this set is allowed but must be intentional: it's a
# declaration that "this env var is read directly from os.environ
# because it is genuinely NOT per-session" (boot-time config, platform
# detection, etc.).  When the answer is "this should be session-scoped",
# use ``shared.session_context.get_session_env(...)`` instead — no
# allowlist entry needed.
# ---------------------------------------------------------------------------

ALLOWLIST: Set[Tuple[str, str]] = {
    ("shared/jaato_client.py",                                    "JAATO_PROVIDER"),
    ("shared/jaato_client.py",                                    "MODEL_NAME"),
    ("shared/plugins/clarification/channels.py",                  "JAATO_CLARIFICATION_TIMEOUT"),
    ("shared/plugins/environment/plugin.py",                      "JAATO_KERBEROS_PROXY"),
    ("shared/plugins/environment/plugin.py",                      "JAATO_NO_PROXY"),
    ("shared/plugins/environment/plugin.py",                      "JAATO_SESSION_ID"),
    ("shared/plugins/environment/plugin.py",                      "JAATO_SSL_VERIFY"),
    ("shared/plugins/gc/base.py",                                 "JAATO_GC_PRESSURE"),
    ("shared/plugins/gc/base.py",                                 "JAATO_GC_TARGET"),
    ("shared/plugins/gc/base.py",                                 "JAATO_GC_THRESHOLD"),
    ("shared/plugins/mermaid_formatter/backends/__init__.py",     "JAATO_MERMAID_BACKEND"),
    ("shared/plugins/mermaid_formatter/plugin.py",                "JAATO_MERMAID_BACKEND"),
    ("shared/plugins/mermaid_formatter/plugin.py",                "JAATO_MERMAID_SCALE"),
    ("shared/plugins/mermaid_formatter/plugin.py",                "JAATO_MERMAID_THEME"),
    ("shared/plugins/mermaid_formatter/plugin.py",                "JAATO_VISION_DIR"),
    ("shared/plugins/mermaid_formatter/renderer.py",              "JAATO_KROKI_URL"),
    ("shared/plugins/model_provider/__init__.py",                 "JAATO_WORKSPACE_ROOT"),
    ("shared/plugins/model_provider/anthropic/env.py",            "JAATO_ANTHROPIC_ENABLE_CACHING"),
    ("shared/plugins/model_provider/anthropic/env.py",            "JAATO_ANTHROPIC_ENABLE_THINKING"),
    ("shared/plugins/model_provider/anthropic/env.py",            "JAATO_ANTHROPIC_THINKING_BUDGET"),
    ("shared/plugins/model_provider/antigravity/env.py",          "JAATO_ANTIGRAVITY_AUTO_ROTATE"),
    ("shared/plugins/model_provider/antigravity/env.py",          "JAATO_ANTIGRAVITY_ENDPOINT"),
    ("shared/plugins/model_provider/antigravity/env.py",          "JAATO_ANTIGRAVITY_PROJECT_ID"),
    ("shared/plugins/model_provider/antigravity/env.py",          "JAATO_ANTIGRAVITY_QUOTA"),
    ("shared/plugins/model_provider/antigravity/env.py",          "JAATO_ANTIGRAVITY_RETRY_EMPTY"),
    ("shared/plugins/model_provider/antigravity/env.py",          "JAATO_ANTIGRAVITY_SESSION_RECOVERY"),
    ("shared/plugins/model_provider/antigravity/env.py",          "JAATO_ANTIGRAVITY_THINKING_BUDGET"),
    ("shared/plugins/model_provider/antigravity/env.py",          "JAATO_ANTIGRAVITY_THINKING_LEVEL"),
    ("shared/plugins/model_provider/antigravity/provider.py",     "JAATO_WORKSPACE_ROOT"),
    ("shared/plugins/model_provider/claude_cli/env.py",           "JAATO_CLAUDE_CLI_MAX_TURNS"),
    ("shared/plugins/model_provider/claude_cli/env.py",           "JAATO_CLAUDE_CLI_MODE"),
    ("shared/plugins/model_provider/claude_cli/env.py",           "JAATO_CLAUDE_CLI_PATH"),
    ("shared/plugins/model_provider/claude_cli/env.py",           "JAATO_CLAUDE_CLI_PERMISSION_MODE"),
    ("shared/plugins/model_provider/github_models/provider.py",   "JAATO_WORKSPACE_ROOT"),
    ("shared/plugins/notebook/plugin.py",                         "JAATO_TOOL_BINDINGS"),
    ("shared/plugins/permission/channels.py",                     "JAATO_PERMISSION_TIMEOUT"),
    ("shared/plugins/permission/config_loader.py",                "JAATO_PERMISSION_TIMEOUT"),
    ("shared/plugins/session/config_loader.py",                   "JAATO_SESSION_CONFIG"),
    ("shared/plugins/subagent/plugin.py",                         "JAATO_PROVIDER_TRACE"),
    ("shared/plugins/subagent/plugin.py",                         "JAATO_TRACE_LOG"),
    ("shared/plugins/subagent/plugin.py",                         "JAATO_WORKSPACE_ROOT"),
    ("shared/plugins/subagent/plugin.py",                         "LOCATION"),
    ("shared/plugins/subagent/plugin.py",                         "MODEL_NAME"),
    ("shared/plugins/subagent/plugin.py",                         "PROJECT_ID"),
    ("shared/plugins/table_formatter/plugin.py",                  "JAATO_AMBIGUOUS_WIDTH"),
    ("shared/plugins/telemetry/__init__.py",                      "JAATO_TELEMETRY_BACKEND"),
    ("shared/plugins/telemetry/__init__.py",                      "JAATO_TELEMETRY_ENABLED"),
    ("shared/plugins/telemetry/__init__.py",                      "JAATO_TELEMETRY_EXPORTER"),
    ("shared/plugins/telemetry/__init__.py",                      "JAATO_TELEMETRY_REDACT_CONTENT"),
    ("shared/plugins/telemetry/otel_plugin.py",                   "JAATO_TELEMETRY_FILE"),
    ("shared/terminal_caps.py",                                   "JAATO_GRAPHICS_PROTOCOL"),
    # ---- added 2026-06-19 (session_env_audit burndown) ----
    # session_context.py IS the get_session_env module; these are its
    # own bootstrap fallback reads — it cannot recurse into itself.
    ("shared/session_context.py",                                 "JAATO_WORKSPACE_ROOT"),
    ("shared/session_context.py",                                 "JAATO_CONFIG_ROOT"),
    # Global debug-dump toggle, not per-session.
    ("shared/plugins/model_provider/anthropic/provider.py",       "JAATO_DUMP_PROVIDER_REQUEST"),
    # Daemon-side auth/key resolution (pass:// resolved daemon-side),
    # consistent with other provider key reads.
    ("shared/plugins/model_provider/zhipuai/env.py",              "JAATO_ZHIPUAI_API_KEY"),
}

# Vars NOT considered session-tied — generic OS / platform / non-jaato.
# Reads of these via ``os.environ.get`` are out of scope.
_SESSION_TIED_PREFIX = "JAATO_"
_SESSION_TIED_LITERAL = {"MODEL_NAME", "PROJECT_ID", "LOCATION"}

# Files we don't audit (test scaffolding, third-party shims, etc.).
_SKIP_PATH_FRAGMENTS = ("/tests/", "/test_", "_test.py")

# Pattern: ``os.environ.get('VAR'`` / ``os.environ['VAR']`` / ``os.getenv('VAR'``
_PATTERN = re.compile(
    r"""os\.(?:environ\.get|getenv)\s*\(\s*['"]([A-Z_][A-Z0-9_]*)['"]"""
    r"""|os\.environ\[\s*['"]([A-Z_][A-Z0-9_]*)['"]"""
)


def _is_session_tied(var: str) -> bool:
    """Return True if this var is in scope for the audit."""
    return var.startswith(_SESSION_TIED_PREFIX) or var in _SESSION_TIED_LITERAL


def _scan_shared() -> Set[Tuple[str, str]]:
    """Walk shared/ and return every (relpath, var) where session-tied
    env vars are read via ``os.environ`` directly."""
    repo_root = Path(__file__).resolve().parents[2]  # jaato-server/
    shared_dir = repo_root / "shared"
    assert shared_dir.is_dir(), f"expected {shared_dir} to exist"

    hits: Set[Tuple[str, str]] = set()
    for py_path in shared_dir.rglob("*.py"):
        path_str = str(py_path)
        if any(frag in path_str for frag in _SKIP_PATH_FRAGMENTS):
            continue
        try:
            text = py_path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for m in _PATTERN.finditer(line):
                var = m.group(1) or m.group(2)
                if _is_session_tied(var):
                    rel = py_path.relative_to(repo_root).as_posix()
                    hits.add((rel, var))
    return hits


def test_no_new_direct_environ_reads_for_session_tied_vars():
    """Audit fails on direct os.environ reads of session-tied vars not in ALLOWLIST."""
    current = _scan_shared()

    new_unallowlisted = current - ALLOWLIST
    if new_unallowlisted:
        details = "\n  ".join(
            f"{path}: {var}" for path, var in sorted(new_unallowlisted)
        )
        raise AssertionError(
            "New direct os.environ reads of session-tied env vars detected:\n  "
            f"{details}\n\n"
            "Per the workspace-tied env-var rule, framework code that runs in\n"
            "session context must read env vars via\n"
            "    from shared.session_context import get_session_env\n"
            "    value = get_session_env('JAATO_FOO')\n"
            "rather than os.environ.get / os.getenv directly.  get_session_env\n"
            "checks the per-session contextvar (populated from the SDK's\n"
            "env_file) before falling back to os.environ — preserving\n"
            "workspace isolation when multiple sessions run on one daemon.\n\n"
            "If THIS particular access is genuinely NOT per-session (boot-time\n"
            "config, platform detection, etc.), add it to the ALLOWLIST in\n"
            "this file with a comment explaining why."
        )


def test_allowlist_has_no_stale_entries():
    """Audit fails when ALLOWLIST entries no longer exist in source.

    Rationale: stale entries hide migrations.  When a dev moves a read
    from os.environ to get_session_env they should remove the stale
    allowlist entry to keep the audit's surface minimal.
    """
    current = _scan_shared()

    stale = ALLOWLIST - current
    if stale:
        details = "\n  ".join(
            f"{path}: {var}" for path, var in sorted(stale)
        )
        raise AssertionError(
            "ALLOWLIST entries no longer present in source — likely "
            "migrated to get_session_env:\n  "
            f"{details}\n\n"
            "Remove these lines from ALLOWLIST in this file."
        )
