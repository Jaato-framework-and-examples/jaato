"""Every session-construction route must wire the SessionManager into plugins.

``set_session_manager`` is how a daemon-tier plugin gets the handle its tools
need (``list_siblings`` / ``send_to_sibling`` / ``session_ops``).  It used to
be called from ``_create_session_impl`` only — so a session reached by
**attach** (which runs disk-restore) came back with its tools present and
their daemon wiring absent.

The failure mode is the nasty part: the tools answered

    "send_to_sibling is unavailable: no session manager is attached
     (this build routes it daemon-side)"

— the same words a genuine misconfiguration produces.  A wiring gap and a
build that legitimately has no manager were indistinguishable from the
outside.

Reported by the cascade-coordination probe, discriminated on ONE daemon and
ONE build with attach as the only variable.  It mattered because attach is the
route to making a sibling COLD, so the answer to "how do I test
``sibling_cold``" disabled the verb needed to observe it.

The wiring now lives in ``_construct_and_initialize_server`` — the single
sanctioned construction funnel — so create and disk-restore share the line.
"""

import ast
import pathlib

import pytest

from server.session_manager import SessionManager
from server.tests.test_bootstrap_partition import PERMITTED_CONSTRUCTION_SITES


SM_PATH = pathlib.Path("jaato-server/server/session_manager.py")
WIRE = "_wire_session_manager_into_plugins"


def _fn_calling(tree, needle):
    """Names of functions containing a call to *needle*."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr == needle):
                    out.add(node.name)
    return out


def test_the_construction_funnel_wires_the_manager():
    """Tied to the SAME allow-list the bootstrap-partition test maintains.

    A contributor adding a construction site must extend that list — and this
    asserts the SessionManager-side site on it also wires, so the two facts
    cannot drift apart.
    """
    tree = ast.parse(SM_PATH.read_text(encoding="utf-8"))
    callers = _fn_calling(tree, WIRE)

    sanctioned = {
        fn for mod, fn in PERMITTED_CONSTRUCTION_SITES
        if mod.endswith("session_manager.py")
    }
    assert sanctioned, "the allow-list no longer names a session_manager site"
    assert sanctioned <= callers, (
        f"construction site(s) {sanctioned - callers} build a server without "
        f"calling {WIRE} — sessions from that route lose their daemon wiring"
    )


def test_the_wiring_is_not_duplicated_at_a_caller():
    """One site, so a route cannot be wired 'sometimes'.

    The bug was precisely that wiring lived at a CALLER of the funnel rather
    than in it; a second caller-side copy would let the two drift.
    """
    tree = ast.parse(SM_PATH.read_text(encoding="utf-8"))
    assert len(_fn_calling(tree, WIRE)) == 1, (
        "wiring must be called from the construction funnel ONLY"
    )


# ----------------------------------------------------------------------
# Behaviour of the wiring itself
# ----------------------------------------------------------------------

class _Plugin:
    def __init__(self):
        self.manager = None

    def set_session_manager(self, m):
        self.manager = m


class _NoHook:
    pass


class _Raises:
    def set_session_manager(self, m):
        raise RuntimeError("nope")


def _server(**plugins):
    reg = type("R", (), {
        "list_exposed": lambda self: list(plugins),
        "get_plugin": lambda self, n: plugins.get(n),
    })()
    return type("S", (), {"registry": reg})()


def test_every_asking_plugin_receives_the_manager():
    sm = SessionManager.__new__(SessionManager)
    a, b = _Plugin(), _Plugin()
    sm._wire_session_manager_into_plugins(_server(alpha=a, beta=b))
    assert a.manager is sm and b.manager is sm


def test_a_plugin_without_the_hook_is_skipped_not_crashed():
    """Duck-typed: most plugins never grow this hook."""
    sm = SessionManager.__new__(SessionManager)
    sm._wire_session_manager_into_plugins(_server(plain=_NoHook()))


def test_one_failing_plugin_does_not_block_the_others():
    """Session construction must not die because one plugin's hook raised."""
    sm = SessionManager.__new__(SessionManager)
    good = _Plugin()
    sm._wire_session_manager_into_plugins(_server(bad=_Raises(), good=good))
    assert good.manager is sm, "a raising plugin aborted the rest of the wiring"


def test_a_server_without_a_registry_is_tolerated():
    sm = SessionManager.__new__(SessionManager)
    sm._wire_session_manager_into_plugins(type("S", (), {"registry": None})())
