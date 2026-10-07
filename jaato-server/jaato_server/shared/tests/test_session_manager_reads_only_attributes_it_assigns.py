"""An attribute `SessionManager` READ and nobody ever ASSIGNED (#1176, #1056).

``SessionManager`` built its daemon-owned session index in ``__init__`` and
bound it to ``self._session_workspace_index``.  Three sites then read it back
under a name that is assigned nowhere in the package::

    self._session_index.record_identity(...)   # :4714, in try/except -> WARNING
    self._session_index.record_identity(...)   # :4779, in try/except -> DEBUG
    stored = self._session_index.identity(...) # :4812, UNGUARDED -> AttributeError

A misspelling, introduced by ``789e4abd`` (the #812 commit) and shipped in
every release from ``jaato-server-0.14.0`` onward.  The consequence is not
cosmetic: **#812's third record -- the daemon-owned workspace index, the
cross-workspace lookup for someone holding only a session id -- has never
been written in any release.**  A runner identity was stamped on the live
``Session`` and persisted into the session record, and the one store that
answers "which process ran the session whose id is all I have" got nothing.

WHY A SCAN, AND NOT VIGILANCE
=============================

Because the defect is *invisible in a working tree*.  Python resolves an
attribute at read time, so a misspelling is not a syntax error, not an
import error and not something a linter following the class's ``__init__``
can see.  Two of the three sites sit inside ``except Exception`` blocks
written to keep a diagnostic from failing a spawn -- correct in themselves,
and between them they swallow the ``AttributeError`` on the two paths that
run *every session*.  What is left is one WARNING per session in a daemon
log, phrased as though the index were merely unwritable ("runner identity
not written to the workspace index"), which is exactly what an operator
would expect to see on a read-only volume.  The third site raises for real,
on the cold-lookup path a person reaches only after something has already
gone wrong.

So the failure mode is: shipped for six releases, in a method with tests,
under a heading in ``CLAUDE.md`` asserting it worked.  It was not found by
running the code.  It was found by reading it.

Nothing else in the tree asks the question a scan makes trivial -- *is this
name ever assigned?* -- and the question is decidable from the source alone.

WHAT THIS MODULE PINS
=====================

``test_every_self_attribute_read_is_assigned_somewhere``
    Every ``self.<name>`` read inside ``class SessionManager`` resolves to
    something the class body defines (a method, a property, a class
    attribute) or to something assigned to ``self`` somewhere in the class.
    A name read but never assigned fails, naming the lines that read it.

``test_the_scan_notices_a_misspelled_attribute``
    The discrimination test.  The scan above is only worth running if it
    can fail, and an AST walk that quietly stops matching (a future Python
    node shape, a refactor that moves the class) would go on reporting
    success forever.  This drives the scanner over a fabricated class
    carrying exactly #1176's shape and requires it to be caught.

THE ALLOW-LIST
==============

Empty, and that is the measurement rather than an aspiration: on the fixed
tree the scan finds 183 distinct reads against 184 class-body definitions
and 48 assignment targets, with **zero** unassigned.  ``SessionManager``
has no base class, declares no nested class and deletes no attribute, so
the three obvious sources of false positive do not arise here.

If one does arise, add it to ``_ALLOWED_UNASSIGNED`` *with its reason* --
do not loosen the scan until it passes.  And if the list grows past a
handful, the scan's premise is wrong for this class and it should be said
so rather than kept limping.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Dict, List, Set, Tuple

from jaato_server.shared.tests.reversion import Reversion

_MANAGER = "jaato-server/jaato_server/server/session_manager.py"
_MANAGER_PY = pathlib.Path(__file__).resolve().parents[2] / (
    "server/session_manager.py")

#: Attributes read on ``self`` that are legitimately never assigned in the
#: class body, each with the reason it is fine.  EMPTY on the tree as it
#: stands -- see "THE ALLOW-LIST" above before adding to it.
#:
#: The shapes that would earn an entry:
#:   * set via ``setattr(self, name, ...)``, which an AST scan for
#:     ``Attribute(ctx=Store)`` cannot see;
#:   * inherited from a base class (``SessionManager`` has none today);
#:   * assigned by a collaborator from outside the class.
#: Each of those is a real hazard in its own right, which is why an entry
#: has to be argued rather than added.
_ALLOWED_UNASSIGNED: Dict[str, str] = {}


REVERSIONS = [
    Reversion(
        target=_MANAGER,
        find=(
            "            self._session_workspace_index.record_identity("
            "session_id, identity.to_dict())"
        ),
        replace=(
            "            self._session_index.record_identity("
            "session_id, identity.to_dict())"
        ),
        test="test_every_self_attribute_read_is_assigned_somewhere",
        because=(
            "#812's workspace-index record goes unwritten again, and "
            "silently: this site sits in an except-Exception block, so the "
            "AttributeError becomes one WARNING per session that reads as "
            "an unwritable index rather than as a name that does not exist"
        ),
    ),
]


def _session_manager_class() -> ast.ClassDef:
    """The ``SessionManager`` class node, parsed from source.

    Parsed rather than imported: the question is about the SOURCE (which
    names does this class write?), and importing the module would pull in
    the whole server package for a text fact.
    """
    tree = ast.parse(_MANAGER_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SessionManager":
            return node
    raise AssertionError(
        f"no `class SessionManager` in {_MANAGER}.  The scan in this module "
        f"is anchored on that name; if the class moved or was renamed, "
        f"re-anchor it rather than deleting this guard."
    )


def _class_body_names(cls: ast.ClassDef) -> Set[str]:
    """Names the class body itself defines: methods, properties, class attrs.

    ``self.foo()`` is an attribute READ that resolves here rather than to
    anything assigned on the instance, so these have to be collected or
    every method call would be reported as unassigned.
    """
    names: Set[str] = set()
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(
                node.target, ast.Name):
            names.add(node.target.id)
    return names


def _self_attribute_use(
    cls: ast.ClassDef,
) -> Tuple[Dict[str, List[int]], Set[str]]:
    """Split every ``self.<name>`` in *cls* into reads and assignments.

    Returns:
        ``(reads, assigned)`` -- reads maps each name to the lines that read
        it (so a failure can point at them), assigned is the set of names
        that appear as an assignment TARGET anywhere in the class.

    ``ast.Store`` covers plain assignment, augmented and annotated
    assignment, ``for`` targets, ``with ... as self.x`` and tuple
    unpacking, because the parser marks every one of those the same way.
    ``ast.Del`` is deliberately NOT counted as an assignment: an attribute
    only ever deleted was never created either, which is the same defect
    wearing a different verb.
    """
    reads: Dict[str, List[int]] = {}
    assigned: Set[str] = set()
    for node in ast.walk(cls):
        if not isinstance(node, ast.Attribute):
            continue
        if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
            continue
        if isinstance(node.ctx, ast.Store):
            assigned.add(node.attr)
        elif isinstance(node.ctx, ast.Load):
            reads.setdefault(node.attr, []).append(node.lineno)
    return reads, assigned


def _unassigned_reads(cls: ast.ClassDef) -> Dict[str, List[int]]:
    """Names read on ``self`` that nothing in the class ever assigns."""
    reads, assigned = _self_attribute_use(cls)
    known = assigned | _class_body_names(cls) | set(_ALLOWED_UNASSIGNED)
    return {name: lines for name, lines in reads.items() if name not in known}


def test_every_self_attribute_read_is_assigned_somewhere():
    """No ``self.<name>`` in ``SessionManager`` is read but never assigned.

    THE GUARD.  A name in the report is either a misspelling of a real
    attribute (#1176's ``_session_index`` for ``_session_workspace_index``)
    or a read of something that is never created -- and Python will not tell
    you which until the line runs, if it ever does.
    """
    cls = _session_manager_class()
    missing = _unassigned_reads(cls)
    assert not missing, (
        "SessionManager reads attributes nothing assigns:\n  "
        + "\n  ".join(
            f"self.{name}  (read at line{'s' if len(lines) > 1 else ''} "
            f"{', '.join(str(n) for n in sorted(set(lines)))})"
            for name, lines in sorted(missing.items())
        )
        + "\n\nThis is #1176: `self._session_index` was read at three sites "
          "and assigned nowhere, because the attribute is spelled "
          "`_session_workspace_index`.  Two of the three sites sat inside "
          "`except Exception`, so it shipped in six releases as one WARNING "
          "per session.\n"
          "Fix the name.  If the attribute really is created from outside "
          "the class body (setattr, a base class, a collaborator), add it to "
          "_ALLOWED_UNASSIGNED with the reason -- do not widen the scan."
    )


def test_the_scan_notices_a_misspelled_attribute():
    """The scan can FAIL, so a passing run above means something.

    Not a restatement of the guard: this drives the scanner over source it
    owns, carrying #1176's exact shape -- one attribute assigned in
    ``__init__``, a near-miss of that name read in a method.  Without it, an
    ``ast.walk`` that silently stopped matching (a node shape that changes,
    a class that moves) would report success for as long as nobody looked,
    which is the failure this whole module is about.
    """
    cls = next(
        node for node in ast.parse(
            "class SessionManager:\n"
            "    def __init__(self):\n"
            "        self._session_workspace_index = Index()\n"
            "    def helper(self):\n"
            "        return None\n"
            "    def read(self):\n"
            "        self.helper()\n"
            "        return self._session_index.identity('s')\n"
        ).body if isinstance(node, ast.ClassDef)
    )
    missing = _unassigned_reads(cls)
    assert set(missing) == {"_session_index"}, (
        f"the scanner no longer detects #1176's shape; it reported "
        f"{sorted(missing)}.  A guard that cannot fail proves nothing -- "
        f"repair the scan rather than this expectation."
    )
    # ...and it must not cry wolf over the two legitimate resolutions.
    assert "helper" not in missing          # resolves to the class body
    assert "_session_workspace_index" not in missing   # assigned in __init__
