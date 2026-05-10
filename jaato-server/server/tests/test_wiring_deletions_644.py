"""Regression-pin tests for §7c step 6.6.4.4 — narrow WIRING
deletions (safe-only).

Per the 6.6.4.4 implementation-review audit (split decided per
"always split" policy, F2): 3 of 6 WIRING sites are safely
deletable post-6.6.4.3b seat-flip.  The other 3 (`configure_*`
calls) collapse with 6.6.4.5's atomic field removal because of
cascading downstream daemon-side read dependencies.

This file pins the 3 deleted call sites stay deleted via AST-
based call-site detection (docstrings + comments referencing
the deleted patterns are safe).

Sites covered:

- ``self._jaato.set_gc_plugin(...)`` ×2 (initialize + auth-completion)
- ``self._jaato.set_session_plugin(...)``

The deferred 3 (``configure_plugins_only`` + ``configure_tools``
×2) are explicitly NOT pinned here — they're expected to survive
until 6.6.4.5.  The pin would fail prematurely otherwise.

Audit-discipline: also pins the description-callback hook stays
in place daemon-side (broken pre-6.6.4.4 by 6.6.4.3b's seat-flip;
fix planned via NotificationFrame extension in a follow-up step).
"""

from __future__ import annotations

import ast
import inspect

import server.core as core_module
from server.core import JaatoServer


def _module_calls_with_attr(attr_name: str) -> list:
    """Walk the entire ``server.core`` module AST and return all
    ``Call`` nodes whose function is ``<anything>.<attr_name>``."""
    src = inspect.getsource(core_module)
    tree = ast.parse(src)
    return [
        n for n in ast.walk(tree)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr_name
        )
    ]


def _module_calls_jaato_method(method_name: str) -> list:
    """Walk the AST and return ``Call`` nodes shaped like
    ``<anything>._jaato.<method_name>(...)`` — the WIRING pattern."""
    src = inspect.getsource(core_module)
    tree = ast.parse(src)
    found = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if not isinstance(f, ast.Attribute) or f.attr != method_name:
            continue
        inner = f.value
        if isinstance(inner, ast.Attribute) and inner.attr == "_jaato":
            found.append(n)
    return found


# ----------------------------------------------------------------------
# Deleted-site regression pins (the 3 safe-only WIRINGs)
# ----------------------------------------------------------------------


def test_no_jaato_set_gc_plugin_calls_remain() -> None:
    """``self._jaato.set_gc_plugin(...)`` was a daemon-side
    WIRING call propagating GC plugin to the daemon-side
    ``_session`` — dead-weight post-6.6.4.3b (GC trigger path
    is runner-side now).  Both call sites (initialize +
    auth-completion mirror) deleted in §7c step 6.6.4.4."""
    calls = _module_calls_jaato_method("set_gc_plugin")
    assert calls == [], (
        f"§7c step 6.6.4.4 regression: ``self._jaato.set_gc_plugin(...)`` "
        f"WIRING re-introduced in server/core.py ({len(calls)} call sites). "
        f"GC trigger path is runner-side post-6.6.4.3b — daemon-side "
        f"propagation is dead-weight."
    )


def test_no_jaato_set_session_plugin_calls_remain() -> None:
    """``self._jaato.set_session_plugin(...)`` propagated session
    plugin to daemon-side ``_session`` — dead-weight post-6.6.4.3b
    (daemon-side enrichment pipeline is no longer the live one).
    Deleted in §7c step 6.6.4.4."""
    calls = _module_calls_jaato_method("set_session_plugin")
    assert calls == [], (
        f"§7c step 6.6.4.4 regression: ``self._jaato.set_session_plugin(...)`` "
        f"WIRING re-introduced in server/core.py ({len(calls)} call sites). "
        f"Daemon-side enrichment pipeline is no longer the live one."
    )


# ----------------------------------------------------------------------
# Deferred-site presence pins (the 3 unsafe WIRINGs that stay
# until 6.6.4.5)
# ----------------------------------------------------------------------


def test_configure_tools_calls_still_present_until_645() -> None:
    """Pin ``self._jaato.configure_tools(...)`` is NOT yet deleted —
    it has cascading downstream daemon-side read deps (~15 sites).
    Per the 6.6.4.4 audit narrowing, this drops with the atomic
    field removal in §7c step 6.6.4.5 alongside read-site migration.
    If this test starts failing, 6.6.4.5 has landed and the audit
    note in the design doc should be updated."""
    calls = _module_calls_jaato_method("configure_tools")
    assert len(calls) >= 2, (
        f"Expected ≥2 ``self._jaato.configure_tools(...)`` call sites "
        f"(initialize + auth-completion mirror); found {len(calls)}. "
        f"If this fails because count went to 0, 6.6.4.5 has landed "
        f"and this pin can be deleted."
    )


def test_configure_plugins_only_call_still_present_until_645() -> None:
    """Pin ``self._jaato.configure_plugins_only(...)`` is NOT yet
    deleted — pre-auth user-command listing depends on the daemon-
    side runtime/session it creates.  Drops with 6.6.4.5."""
    calls = _module_calls_jaato_method("configure_plugins_only")
    assert len(calls) >= 1, (
        f"Expected ≥1 ``self._jaato.configure_plugins_only(...)`` call "
        f"site; found {len(calls)}. "
        f"If this fails because count went to 0, 6.6.4.5 has landed "
        f"and this pin can be deleted."
    )


# ----------------------------------------------------------------------
# Behavior-preservation pin: GC config UI fields still flow
# ----------------------------------------------------------------------


def test_gc_config_ui_fields_still_assigned_in_initialize() -> None:
    """Even with ``set_gc_plugin`` WIRING deleted, the daemon
    still extracts ``gc_threshold`` / ``gc_strategy`` /
    ``gc_target_percent`` / ``gc_continuous_mode`` from
    ``gc_config`` so they can be stored on AgentState (UI
    surfaces them in the toolbar).  Pin this assignment block
    survived the deletion."""
    src = inspect.getsource(JaatoServer.initialize)
    # All four UI-bound fields must still be assigned from gc_config.
    for field in (
        "gc_threshold = gc_config.threshold_percent",
        "gc_target_percent = gc_config.target_percent",
        "gc_continuous_mode = gc_config.continuous_mode",
    ):
        assert field in src, (
            f"§7c step 6.6.4.4 regression: GC UI field assignment "
            f"{field!r} disappeared from initialize().  These must "
            f"persist daemon-side — they feed AgentState toolbar "
            f"display, not the GC trigger path."
        )


# ----------------------------------------------------------------------
# Audit Finding 2 acknowledgment: description-callback hook still wired
# ----------------------------------------------------------------------


def test_description_callback_still_wired_daemon_side() -> None:
    """Pin the daemon-side ``set_description_callback`` wiring
    survives the §7c step 6.6.4.4 narrowing.  Note: this hook is
    silently broken post-6.6.4.3b (Audit Finding 2: model invokes
    set_description runner-side, fires runner-side instance's
    callback, daemon never sees it).  Fix is planned via
    NotificationFrame extension in a follow-up step.  Until then,
    the wiring stays in place — removing it now would mask the
    intended-but-broken contract."""
    src = inspect.getsource(JaatoServer._setup_session_plugin)
    assert "set_description_callback" in src, (
        "§7c step 6.6.4.4 regression: daemon-side description-"
        "callback wiring removed prematurely.  Audit Finding 2 "
        "documents this hook is silently broken post-6.6.4.3b, "
        "but removing the wiring before the runner-side fix lands "
        "would erase the intended contract."
    )
