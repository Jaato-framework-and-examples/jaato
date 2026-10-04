"""A stack with ``unconfined`` is confinement by its one real profile (#1509).

With ``kernel.apparmor_restrict_unprivileged_unconfined = 1`` (the Ubuntu
default) the kernel converts an UNPRIVILEGED, unconfined task's
``change_profile`` into a stack.  A runner dropped by
``--runner-uid-policy`` (#1168) changes profile after the drop, so it reads

    jaato-ws-<id>//&unconfined (enforce)

and the bootstrap verify refused it, because the parser stripped only the
mode.  A stack is the intersection of its members and ``unconfined``
restricts nothing, so the task is confined by exactly ``jaato-ws-<id>``.
The kernel leaves ``unconfined`` out of the printed mode
(``label_modename``), so ``(enforce)`` is that profile's own mode.

Any other stack is NOT the requested profile: it is bounded by a second
profile jaato did not ask for, and every reader must keep refusing it.

No kernel needed: every case is a fabricated ``attr/current`` value.
"""

from __future__ import annotations

import pytest

from jaato_server.shared.apparmor_label import (
    parse_label,
    profile_name_ignoring_mode,
)

try:  # pragma: no cover - import shape differs per invocation
    from jaato_server.shared.tests.reversion import Reversion
except Exception:  # pragma: no cover
    Reversion = None  # type: ignore[assignment]

_LABEL = "jaato-server/jaato_server/shared/apparmor_label.py"
_BOOTSTRAP = "jaato-server/jaato_server/server/runner/bootstrap.py"
_SESSION = "jaato-server/jaato_server/server/runner/session.py"

REVERSIONS = [] if Reversion is None else [
    Reversion(
        target=_LABEL,
        find="    stack = tuple(c.strip() for c in name.split(STACK_SEPARATOR))\n",
        replace="    stack = (name,)\n",
        test="test_a_stack_with_unconfined_verifies_as_the_profile",
        because="the stacked label is read as one profile name and the "
                "dropped runner is refused again (#1509)",
    ),
    Reversion(
        target=_LABEL,
        find="    if len(real) == 1:\n        return real[0]\n    return name\n",
        replace="    return real[0]\n",
        test="test_a_stack_of_two_real_profiles_is_refused",
        because="a stack with a second real profile is reported as the "
                "requested one, so a foreign boundary passes the verify",
    ),
    Reversion(
        target=_LABEL,
        find="            and not self.foreign_stack\n",
        replace="",
        test="test_a_foreign_stack_is_not_enforced",
        because="a foreign stack's printed mode is claimed as the "
                "requested profile's enforcement",
    ),
    Reversion(
        target=_BOOTSTRAP,
        find='    return name.startswith(expected + "//") and STACK_SEPARATOR not in name',
        replace='    return name.startswith(expected + "//")',
        test="test_thread_verify_sees_a_foreign_stack_as_divergent",
        because="'P//&Q' starts with 'P//' and a thread bounded by Q too "
                "reads as a sub-profile of P",
    ),
    Reversion(
        target=_SESSION,
        find="    if parse_label(exc.actual).stacked:\n",
        replace="    if False:\n",
        test="test_the_mismatch_message_names_stacking",
        because="the refusal blames a missing change_profile rule when "
                "the transition happened as a stack",
    ),
]

P = "jaato-ws-runtime-51d7b8d982d0"


# ----------------------------------------------------------------------
# The parser
# ----------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    f"{P}//&unconfined (enforce)",
    f"unconfined//&{P} (enforce)",       # the kernel sorts by name
    f"{P}//&unconfined (enforce)\x00\n",
])
def test_a_stack_with_unconfined_parses_as_the_profile(raw):
    label = parse_label(raw)
    assert label.profile == P
    assert label.enforced
    assert label.stacked and not label.foreign_stack
    assert profile_name_ignoring_mode(raw) == P


def test_a_child_stacked_with_unconfined_is_the_child():
    label = parse_label(f"{P}//child//&unconfined (enforce)")
    assert label.profile == f"{P}//child"
    assert label.enforced


def test_a_complain_stack_is_not_enforced():
    label = parse_label(f"{P}//&unconfined (complain)")
    assert label.profile == P
    assert label.complaining and not label.enforced


def test_a_foreign_stack_is_not_enforced():
    label = parse_label(f"{P}//&other (enforce)")
    assert label.foreign_stack
    assert label.profile != P
    assert not label.enforced


def test_an_unstacked_label_is_unchanged():
    label = parse_label(f"{P} (enforce)")
    assert (label.profile, label.mode, label.stack) == (P, "enforce", (P,))
    assert profile_name_ignoring_mode("unconfined") == "unconfined"
    assert not parse_label("unconfined").confined


# ----------------------------------------------------------------------
# The bootstrap verify
# ----------------------------------------------------------------------

def _fake_libapparmor():
    class _Fn:
        argtypes = None
        restype = None

        def __call__(self, _profile):
            return 0

    class _Lib:
        aa_change_profile = _Fn()

    return _Lib()


def _attr(tmp_path, contents):
    path = tmp_path / "attr_current"
    path.write_text(contents + "\n")
    return str(path)


def test_a_stack_with_unconfined_verifies_as_the_profile(tmp_path):
    from jaato_server.server.runner.bootstrap import confine_to_profile

    label = confine_to_profile(
        P,
        libapparmor=_fake_libapparmor(),
        proc_attr_path=_attr(tmp_path, f"{P}//&unconfined (enforce)"),
        require_enforce=True,
    )
    assert label.enforced and label.profile == P


def test_a_stack_of_two_real_profiles_is_refused(tmp_path):
    from jaato_server.server.runner.bootstrap import (
        ConfinementMismatchError,
        confine_to_profile,
    )

    with pytest.raises(ConfinementMismatchError) as excinfo:
        confine_to_profile(
            P,
            libapparmor=_fake_libapparmor(),
            proc_attr_path=_attr(tmp_path, f"{P}//&other (enforce)"),
        )
    assert "stack of more than the requested profile" in str(excinfo.value)


def test_the_mismatch_message_names_stacking():
    from jaato_server.server.runner.bootstrap import ConfinementMismatchError
    from jaato_server.server.runner.session import _confinement_mismatch_cause

    exc = ConfinementMismatchError(expected=P, actual=f"{P}//&other (enforce)")
    cause = _confinement_mismatch_cause("unconfined", exc)
    assert "stacked" in cause
    assert "apparmor_restrict_unprivileged_unconfined" in cause
    assert "other than 'unconfined' is refused" in cause


# ----------------------------------------------------------------------
# #1023: sibling threads read the same stacked labels
# ----------------------------------------------------------------------

def _proc_tree(tmp_path, labels):
    for tid, label in labels.items():
        attr = tmp_path / str(tid) / "attr"
        attr.mkdir(parents=True, exist_ok=True)
        (attr / "current").write_text(label + "\n")
    return str(tmp_path)


def test_thread_verify_accepts_stacked_siblings(tmp_path):
    from jaato_server.server.runner.bootstrap import verify_thread_confinement

    tree = _proc_tree(tmp_path, {
        101: f"{P}//&unconfined (enforce)",
        102: f"{P}//&unconfined (enforce)",
        103: f"{P}//child//&unconfined (enforce)",
    })
    scan = verify_thread_confinement(P, task_dir=tree, grace_seconds=0)
    assert not scan.divergent
    assert len(scan.matched) == 3


def test_thread_verify_sees_a_foreign_stack_as_divergent(tmp_path):
    from jaato_server.server.runner.bootstrap import (
        ThreadConfinementDivergence,
        verify_thread_confinement,
    )

    tree = _proc_tree(tmp_path, {
        101: f"{P}//&unconfined (enforce)",
        102: f"{P}//&other (enforce)",
    })
    with pytest.raises(ThreadConfinementDivergence):
        verify_thread_confinement(P, task_dir=tree, grace_seconds=0)


def test_thread_verify_sees_unconfined_as_divergent(tmp_path):
    from jaato_server.server.runner.bootstrap import (
        ThreadConfinementDivergence,
        verify_thread_confinement,
    )

    tree = _proc_tree(tmp_path, {
        101: f"{P}//&unconfined (enforce)",
        102: "unconfined",
    })
    with pytest.raises(ThreadConfinementDivergence):
        verify_thread_confinement(P, task_dir=tree, grace_seconds=0)


# ----------------------------------------------------------------------
# #1323: the notebook kernel in //child stacked with unconfined
# ----------------------------------------------------------------------

def test_the_notebook_counts_a_stacked_child_as_its_boundary(monkeypatch):
    from jaato_server.shared.plugins.notebook import kernel_sandbox

    monkeypatch.setattr(
        kernel_sandbox, "try_read_label",
        lambda *a, **k: parse_label(f"{P}//child//&unconfined (enforce)"),
    )
    assert kernel_sandbox.cell_boundary_profile() == f"{P}//child"


def test_the_notebook_does_not_count_a_stacked_base_profile(monkeypatch):
    from jaato_server.shared.plugins.notebook import kernel_sandbox

    monkeypatch.setattr(
        kernel_sandbox, "try_read_label",
        lambda *a, **k: parse_label(f"{P}//&unconfined (enforce)"),
    )
    assert kernel_sandbox.cell_boundary_profile() is None
