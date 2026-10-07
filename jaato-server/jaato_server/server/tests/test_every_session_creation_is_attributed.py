"""No ``SessionManager.create_session`` call may omit ``created_by``.

``test_attribution_reaches_both_transports`` proves the value arrives on
the three paths it drives.  This proves there is no FOURTH path, which a
behavioural test cannot: it would have to know about the call site to
exercise it, and the failure being guarded against is precisely a call
site nobody thought about.

The defect this freezes is a silent one.  ``_create_session_impl``
declares ``created_by: Optional[str] = None``, so a call that omits it is
not a TypeError, not a warning, and not visible in the session that
results -- the record simply carries no creator, and every downstream
consumer (#859's ledger rows, the #951 DECISION line, the OpenInference
``user.id`` span) omits the key rather than reporting an absence.  A path
added later inherits that silence by default.  ``session.default`` is the
one that already did, on both transports, for as long as attribution has
existed.

WHY A RECEIVER ALLOW-LIST RATHER THAN A SKIP-LIST.  ``create_session``
is also a method of ``JaatoRuntime`` -- a different call with no
``created_by`` parameter at all (``server/runner/session.py``, spawning
subagent sessions).  A guard that skipped receivers it did not recognise
would silently stop covering the daemon the day someone spells the
receiver differently, which is the shape of an inert guard.  So an
UNRECOGNISED receiver fails, and the fix is to classify it here.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Dict, List, Optional, Tuple

from jaato_server.shared.tests.reversion import Reversion

_ROUTER = "jaato-server/jaato_server/server/command_router.py"

REVERSIONS = [
    Reversion(
        target=_ROUTER,
        find="""            env_overrides={
                "JAATO_PROVIDER": provider_name,
                "MODEL_NAME": model_name,
            },
            created_by=self._event_sink.get_client_user(client_id),""",
        replace="""            env_overrides={
                "JAATO_PROVIDER": provider_name,
                "MODEL_NAME": model_name,
            },""",
        test="test_every_manager_create_site_passes_created_by",
        because=(
            "the post-auth create stops attributing its session, and no "
            "behavioural test drives that path -- which is the case this "
            "guard exists for rather than the ones already covered"
        ),
    ),
]

_SERVER = pathlib.Path(__file__).resolve().parents[1]

#: Receiver expressions that ARE a ``SessionManager``.  ``self`` qualifies
#: only inside the class's own module; everywhere else it is some other
#: object entirely.
_MANAGER_RECEIVERS = {
    "self._session_manager",
    "self",                    # only in session_manager.py -- see below
}

#: Receiver expressions that are NOT a ``SessionManager``, with why.
#: Listed rather than skipped so the classification is auditable and a
#: new spelling cannot join them by accident.
_OTHER_RECEIVERS = {
    "runtime": (
        "JaatoRuntime.create_session -- the subagent-session factory, "
        "which has no created_by parameter and whose attribution is "
        "inherited from the parent (SessionManager._creator_of)"
    ),
}

#: ``(module, enclosing function)`` allowed to omit it, with the reason.
#: An exemption is a decision recorded, not a gap: the next author reads
#: why rather than inferring that omitting is normal.
_EXEMPT: Dict[Tuple[str, str], str] = {
    ("session_manager.py", "create_headless_session"): (
        "the client is the synthetic _HEADLESS_CLIENT_ID -- there is no "
        "transport connection behind it, so no sink can answer and a "
        "value here would be invented rather than authenticated. Whether "
        "a reactor-spawned stage should INHERIT its cascade driver's "
        "creator (the idiom _create_subagent_session already uses) is a "
        "real question and a wider change than this guard"
    ),
}


class _Site:
    """One ``x.create_session(...)`` call found in the daemon source."""

    def __init__(
        self, module: str, lineno: int, receiver: Optional[str],
        function: str, has_created_by: bool,
    ) -> None:
        self.module = module
        self.lineno = lineno
        self.receiver = receiver
        self.function = function
        self.has_created_by = has_created_by

    def __repr__(self) -> str:      # pragma: no cover - failure output
        return f"{self.module}:{self.lineno} in {self.function}()"


def _enclosing(tree: ast.AST, node: ast.AST) -> str:
    """The name of the function lexically containing *node*."""
    best = "<module>"
    for parent in ast.walk(tree):
        if not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(parent, "end_lineno", None) or parent.lineno
        if parent.lineno <= node.lineno <= end:
            # The innermost enclosing function wins.
            if best == "<module>" or parent.lineno >= _line_of(tree, best):
                best = parent.name
    return best


def _line_of(tree: ast.AST, name: str) -> int:
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and n.name == name:
            return n.lineno
    return -1


def _sites() -> List[_Site]:
    """Every ``*.create_session(...)`` call in non-test daemon source."""
    found: List[_Site] = []
    for path in sorted(_SERVER.rglob("*.py")):
        if "tests" in path.parts:
            continue
        source = path.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError:         # pragma: no cover - not our concern
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "create_session":
                continue
            found.append(_Site(
                module=path.name,
                lineno=node.lineno,
                receiver=ast.get_source_segment(source, node.func.value),
                function=_enclosing(tree, node),
                has_created_by=any(
                    k.arg == "created_by" for k in node.keywords
                ),
            ))
    return found


def _manager_sites() -> List[_Site]:
    """The subset whose receiver is a ``SessionManager``."""
    out = []
    for site in _sites():
        if site.receiver in _OTHER_RECEIVERS:
            continue
        if site.receiver == "self" and site.module != "session_manager.py":
            continue
        out.append(site)
    return out


# ------------------------------------------------------------- the guard

def test_every_receiver_is_classified() -> None:
    """An unknown receiver is a finding, never a skip.

    Without this, a rename of ``_session_manager`` would quietly take
    every router call site out of the guard's sight while it kept
    reporting success.
    """
    unknown = sorted({
        site.receiver for site in _sites()
        if site.receiver not in _MANAGER_RECEIVERS
        and site.receiver not in _OTHER_RECEIVERS
    })
    assert not unknown, (
        f"unclassified create_session receiver(s): {unknown}. Add each to "
        f"_MANAGER_RECEIVERS (it is a SessionManager, so its calls must "
        f"attribute) or to _OTHER_RECEIVERS with the reason it is not."
    )


def test_every_manager_create_site_passes_created_by() -> None:
    """The guard itself."""
    missing = [
        site for site in _manager_sites()
        if not site.has_created_by
        and (site.module, site.function) not in _EXEMPT
    ]
    assert not missing, (
        f"create_session call(s) omitting created_by: {missing}. "
        f"_create_session_impl defaults it to None, so the session is "
        f"created with no creator and the record, the ledger and the "
        f"permission DECISION line all omit the user -- silently. Read "
        f"the identity from the EventSink at the call site (the sink is "
        f"the only thing that knows, and the event body must never be "
        f"able to claim it), or add an entry to _EXEMPT saying why this "
        f"path has nobody to attribute to."
    )


def test_the_exemptions_are_live() -> None:
    """A stale exemption is how a guard quietly narrows.

    The complexity ratchet's rule, applied here: an entry naming a site
    that no longer omits the field -- or no longer exists -- must be
    removed rather than left as permission nobody needs.
    """
    omitting = {
        (site.module, site.function)
        for site in _manager_sites() if not site.has_created_by
    }
    stale = sorted(set(_EXEMPT) - omitting)
    assert not stale, (
        f"exemption(s) for sites that now attribute correctly or are "
        f"gone: {stale}. Delete the entry."
    )


def test_the_scan_is_not_vacuous() -> None:
    """A guard that matches nothing passes for the wrong reason.

    Both halves matter: the walk must find the daemon's manager calls,
    and it must find the ``runtime`` call it is meant to EXCLUDE -- if
    the latter disappeared, the receiver classification above would be
    asserting over an empty set.
    """
    sites = _sites()
    assert len(_manager_sites()) >= 4, (
        f"expected at least the four known SessionManager.create_session "
        f"sites, found {_manager_sites()}"
    )
    assert any(s.receiver in _OTHER_RECEIVERS for s in sites), (
        "no non-manager create_session call found, so _OTHER_RECEIVERS "
        "is untested classification"
    )
