"""Complain mode is not confinement (#1014).

``JAATO_APPARMOR_COMPLAIN=1`` stamps ``flags=(complain)`` on the whole
profile chain.  The kernel then LOGS each denial and ALLOWS the syscall:
there is a profile and there is no boundary.  Four of the five places that
asked "is this session confined?" answered without consulting the mode —
two by prefix-matching the profile NAME, one from the fact that
provisioning succeeded, one from the presence of a callback — so all four
said yes, and the session record persisted a positive claim,
``sandbox_mode: "apparmor"``, that an operator reads weeks later as
evidence of enforcement.

These tests pin the four asks:

1. one predicate — :mod:`shared.apparmor_label` is the single definition,
   and the deliberately mode-TOLERANT match is named as such;
2. the record distinguishes ``apparmor`` from ``apparmor-complain``;
3. complain-mode generation announces itself at WARNING;
4. ``interactive_shell.require_confinement`` means *enforce*.

No AppArmor LSM is needed to run any of this: every site is exercised from
a fabricated ``attr/current`` value, which is the point of routing the
decision through one parser.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from unittest.mock import patch

import pytest

from shared.apparmor_label import (
    COMPLAIN_ENV_VAR,
    MODE_COMPLAIN,
    MODE_ENFORCE,
    SANDBOX_MODE_APPARMOR,
    SANDBOX_MODE_APPARMOR_COMPLAIN,
    SANDBOX_MODE_SOFT,
    complain_mode_requested,
    enforced_profile_name,
    label_is_enforced,
    parse_label,
    profile_name_ignoring_mode,
    sandbox_mode_for_profile,
    sandbox_mode_is_apparmor,
    sandbox_mode_is_enforced,
    try_read_label,
)


# ----------------------------------------------------------------------
# Reversions: the one change that must make each guard go red.
# ----------------------------------------------------------------------
#
# Read by ``shared/tests/test_every_guard_detects_its_own_reversion.py``,
# which puts each defect back in a DISPOSABLE COPY of the checkout and
# fails if the named test still passes.  That is what stops the tests
# below becoming decorative: a guard nobody has seen fail is a guard
# nobody knows works.
#
# Every anchor here is deliberately tight -- the call or the condition
# itself, never a run of lines reaching down to the next ``def``.  A span
# that crosses a gap goes stale the moment anything is inserted into it,
# and a stale reversion reports BLOCKED: the guard is then neither
# known-good nor known-broken.  This change did exactly that to #1023's
# own reversion, by inserting a helper between a call and the ``def``
# that followed it -- which is why the rule is written down here.

try:  # pragma: no cover - import shape differs per invocation
    from shared.tests.test_every_guard_detects_its_own_reversion import Reversion
except Exception:  # pragma: no cover
    Reversion = None  # type: ignore[assignment]

_BOOTSTRAP = "jaato-server/server/runner/bootstrap.py"
_APPARMOR = "jaato-server/server/apparmor.py"
_LABEL = "jaato-server/shared/apparmor_label.py"
_ISHELL = "jaato-server/shared/plugins/interactive_shell/plugin.py"
_KERNEL_SANDBOX = "jaato-server/shared/plugins/notebook/kernel_sandbox.py"

REVERSIONS = [] if Reversion is None else [
    # Ask 1 -- the load-bearing readback.  Accepting any attached profile
    # is the pre-#1014 behaviour: the INFO line says "runner confined to"
    # about a kernel that is enforcing nothing, and no warning is emitted.
    Reversion(
        target=_BOOTSTRAP,
        find='''    if label.enforced:
        logger.info(
            "runner confined to AppArmor profile %s (kernel reports: %s)",''',
        replace='''    if True:
        logger.info(
            "runner confined to AppArmor profile %s (kernel reports: %s)",''',
        test="test_confine_to_profile_warns_rather_than_claiming_confinement",
        because=(
            "a complain-mode profile is logged as confinement again, which "
            "is the line the #1014 operator read"
        ),
    ),
    # Ask 1 -- one predicate.  BEHAVIOUR is unchanged by this reversion;
    # what comes back is a second parser of the mode string, living
    # outside the one module allowed to know that vocabulary.  That is the
    # finding itself: the tree held five opinions, four were wrong, and
    # nothing failed when a sixth was added.
    Reversion(
        target=_KERNEL_SANDBOX,
        find='''    label = try_read_label()
    return label.profile if label.enforced else None''',
        replace='''    label = try_read_label()
    mode = label.raw.partition(" (")[2].rstrip(")").strip()
    return label.profile if mode == "enforce" else None''',
        test="test_no_second_mode_parser_survives_in_the_tree",
        because=(
            "a module parses the enforcement mode itself again, so the tree "
            "holds two definitions of 'the kernel is enforcing'"
        ),
    ),
    # Ask 2 -- the persisted record.  A complain-mode session goes back to
    # claiming ``sandbox_mode: "apparmor"``: a durable positive claim about
    # a boundary the kernel was not applying.
    Reversion(
        target=_LABEL,
        find=(
            "    return SANDBOX_MODE_APPARMOR_COMPLAIN if complain "
            "else SANDBOX_MODE_APPARMOR"
        ),
        replace="    return SANDBOX_MODE_APPARMOR",
        test="test_complain_provisioning_records_the_mode_not_a_boundary_claim",
        because=(
            "the session record asserts enforcement for a session that had "
            "no kernel boundary"
        ),
    ),
    # Ask 3 -- announce it.  Generation goes silent again, which is the
    # state in which grepping ``complain`` against ``logger|warn`` in
    # ``server/apparmor.py`` returned nothing at all.
    Reversion(
        target=_APPARMOR,
        find='''        complain = complain_mode_requested()
        announce_complain_mode_once()
        self._complain_profiles[session_id] = complain''',
        replace='''        complain = complain_mode_requested()
        self._complain_profiles[session_id] = complain''',
        test="test_rendering_a_complain_profile_announces_and_records",
        because=(
            "complain-mode profile generation stops announcing itself -- the "
            "one weakened boundary in this tree that was silent"
        ),
    ),
    # Ask 4 -- the strictest fail-closed knob in the tree.  Reverting to
    # "a transition callback is installed, therefore confined" is exactly
    # what let ``require_confinement: true`` pass while the kernel blocked
    # nothing.
    Reversion(
        target=_ISHELL,
        find='''            if not self._require_confinement:
                return None
            label = read_thread_label()
            if label.enforced:
                return None''',
        replace='''            return None''',
        test="test_require_confinement_refuses_a_complain_mode_child",
        because=(
            "require_confinement is satisfied by a transition into a "
            "complain-mode child profile, which enforces nothing"
        ),
    ),
]

# ----------------------------------------------------------------------
# Ask 1 — one predicate, and a named mode-tolerant sibling
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,profile,mode",
    [
        ("jaato-ws-a (enforce)", "jaato-ws-a", MODE_ENFORCE),
        ("jaato-ws-a (complain)", "jaato-ws-a", MODE_COMPLAIN),
        ("jaato-ws-a//child (complain)", "jaato-ws-a//child", MODE_COMPLAIN),
        ("jaato-ws-a//tool_hat (enforce)", "jaato-ws-a//tool_hat", MODE_ENFORCE),
        # procfs NUL-terminates, and the terminator does not always arrive
        # with the newline (#1026 measured a bare ``kernel\x00``).
        ("jaato-ws-a (enforce)\x00\n", "jaato-ws-a", MODE_ENFORCE),
        ("unconfined", "", None),
        ("unconfined\n", "", None),
        ("", "", None),
        # A bare name with no annotation: not evidence of enforcement.
        ("jaato-ws-a", "jaato-ws-a", None),
    ],
)
def test_parse_label_separates_profile_from_mode(raw, profile, mode):
    label = parse_label(raw)
    assert label.profile == profile
    assert label.mode == mode


def test_only_enforce_is_a_boundary():
    assert label_is_enforced("jaato-ws-a (enforce)")
    assert not label_is_enforced("jaato-ws-a (complain)")
    assert not label_is_enforced("jaato-ws-a")          # no mode reported
    assert not label_is_enforced("unconfined")
    assert not label_is_enforced("")


def test_enforced_profile_name_matches_the_notebook_helper_it_replaced():
    """The notebook kernel was the one component that got this right, and
    its behaviour is the contract the shared helper had to preserve."""
    assert enforced_profile_name("jaato-ws-a//child (enforce)") == "jaato-ws-a//child"
    assert enforced_profile_name("jaato-ws-a//child (complain)") is None
    assert enforced_profile_name("jaato-ws-a") is None
    assert enforced_profile_name("unconfined") is None


def test_mode_tolerant_match_is_named_for_what_it_does():
    """The legitimate mode-blind question — *which profile is this task
    in* — keeps its answer, and the name says it discards the mode so it
    cannot be read as an enforcement assertion."""
    assert profile_name_ignoring_mode("jaato-ws-a (complain)") == "jaato-ws-a"
    assert profile_name_ignoring_mode("jaato-ws-a (enforce)") == "jaato-ws-a"
    # "unconfined" is the ANSWER to that question, not a missing one.
    assert profile_name_ignoring_mode("unconfined") == "unconfined"


def test_describe_never_says_confined_about_a_complain_profile():
    """#1014's log line read ``runner confined to AppArmor profile X
    (kernel reports: X (complain))`` — the truth in the parenthetical of a
    line whose leading words said the opposite.  Nobody greps a
    parenthetical."""
    complain = parse_label("jaato-ws-a (complain)").describe()
    assert "NOT a kernel boundary" in complain
    assert parse_label("jaato-ws-a (enforce)").describe() == "jaato-ws-a (enforce)"


def test_the_definition_is_pure_stdlib():
    """The module is importable from ``server.runner.bootstrap``, which
    must stay loadable before plugin discovery.  A jaato import here would
    pull plugin code into the unconfined window the bootstrap exists to
    close."""
    import ast
    import pathlib

    import shared.apparmor_label as mod

    tree = ast.parse(pathlib.Path(mod.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert imported <= {"os", "dataclasses", "typing", "__future__"}, imported


#: A comparison against the enforcement-mode token — ``== "enforce"``,
#: ``!= 'enforce'``, either way round.  Prose mentioning ``(enforce)`` is
#: deliberately not matched: the finding is a second PARSER, not a second
#: mention.
_MODE_COMPARISON = re.compile(
    r"""(==|!=|\bis\b)\s*["']enforce["']|["']enforce["']\s*(==|!=)"""
)


def test_no_second_mode_parser_survives_in_the_tree():
    """Ask 1 is "one predicate": a module that compares a label against
    the enforcement-mode token itself is a sixth opinion by construction,
    whatever it currently concludes."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    allowed = {root / "shared" / "apparmor_label.py"}    # the definition
    offenders = []
    for path in root.rglob("*.py"):
        if path in allowed or "/tests/" in path.as_posix():
            continue
        text = path.read_text(errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if _MODE_COMPARISON.search(line):
                offenders.append(f"{path}:{lineno}: {line.strip()}")
    assert not offenders, (
        "enforcement mode is parsed outside shared/apparmor_label.py:\n"
        + "\n".join(offenders)
        + "\n\nResolve through parse_label() / label_is_enforced() instead."
    )


# ----------------------------------------------------------------------
# Ask 1 — the load-bearing readback, bootstrap.confine_to_profile
# ----------------------------------------------------------------------


def _fake_libapparmor(rc: int = 0):
    class _Fn:
        argtypes = None
        restype = None

        def __call__(self, _profile):
            return rc

    class _Lib:
        aa_change_profile = _Fn()

    return _Lib()


def _attr_file(tmp_path, contents: str) -> str:
    p = tmp_path / "attr_current"
    p.write_text(contents)
    return str(p)


def test_confine_to_profile_reports_enforce_as_confined(tmp_path, caplog):
    from server.runner.bootstrap import confine_to_profile

    path = _attr_file(tmp_path, "jaato-ws-s1 (enforce)\n")
    with caplog.at_level(logging.INFO, logger="server.runner.bootstrap"):
        label = confine_to_profile(
            "jaato-ws-s1",
            libapparmor=_fake_libapparmor(),
            proc_attr_path=path,
        )
    # Assert the OBSERVABLE behaviour before the return value, so this
    # test's verdict is about what the function does rather than about
    # its signature.
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "runner confined to" in r.getMessage()
        for r in caplog.records if r.levelno == logging.INFO
    )
    assert label.enforced


def test_confine_to_profile_warns_rather_than_claiming_confinement(tmp_path, caplog):
    """The #1014 state.  The transition succeeded, the right profile is
    attached, and the kernel is enforcing nothing — so the line an operator
    reads must not lead with "confined"."""
    from server.runner.bootstrap import confine_to_profile

    path = _attr_file(tmp_path, "jaato-ws-s1 (complain)\n")
    with caplog.at_level(logging.INFO, logger="server.runner.bootstrap"):
        label = confine_to_profile(
            "jaato-ws-s1",
            libapparmor=_fake_libapparmor(),
            proc_attr_path=path,
        )

    # The log assertions come first and deliberately: they are what an
    # operator had to go on, and they are what the unfixed tree gets
    # wrong.  Before #1014 this emitted INFO "runner confined to AppArmor
    # profile jaato-ws-s1 (kernel reports: jaato-ws-s1 (complain))" and no
    # warning at all.
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert not any("runner confined to" in m for m in infos), infos

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "a profile with no kernel boundary must announce itself"
    text = warnings[0].getMessage()
    assert COMPLAIN_ENV_VAR in text
    assert "NOT confined" in text

    assert label.complaining and not label.enforced


def test_require_enforce_refuses_a_complain_profile(tmp_path):
    """The hook #1013's ``JAATO_APPARMOR_BEHAVIOR=require`` needs.  Built
    on the mode-blind readback this replaces, ``require`` would have been
    satisfied by exactly the posture it exists to refuse."""
    from server.runner.bootstrap import ConfinementModeError, confine_to_profile

    path = _attr_file(tmp_path, "jaato-ws-s1 (complain)\n")
    with pytest.raises(ConfinementModeError) as excinfo:
        confine_to_profile(
            "jaato-ws-s1",
            libapparmor=_fake_libapparmor(),
            proc_attr_path=path,
            require_enforce=True,
        )
    assert excinfo.value.label.mode == MODE_COMPLAIN
    assert COMPLAIN_ENV_VAR in str(excinfo.value)


def test_require_enforce_accepts_an_enforcing_profile(tmp_path):
    from server.runner.bootstrap import confine_to_profile

    path = _attr_file(tmp_path, "jaato-ws-s1 (enforce)\n")
    label = confine_to_profile(
        "jaato-ws-s1",
        libapparmor=_fake_libapparmor(),
        proc_attr_path=path,
        require_enforce=True,
    )
    assert label.enforced


def test_a_different_profile_is_still_a_mismatch_not_a_mode_problem(tmp_path):
    """The #1026/§6.5 condition is untouched: which-profile and which-mode
    are separate verdicts and must stay separately diagnosable."""
    from server.runner.bootstrap import ConfinementMismatchError, confine_to_profile

    path = _attr_file(tmp_path, "unconfined\n")
    with pytest.raises(ConfinementMismatchError):
        confine_to_profile(
            "jaato-ws-s1",
            libapparmor=_fake_libapparmor(),
            proc_attr_path=path,
        )


def test_read_current_profile_strips_the_nul_terminator(tmp_path):
    from server.runner.bootstrap import read_current_profile

    path = _attr_file(tmp_path, "jaato-ws-s1 (enforce)\x00\n")
    assert read_current_profile(path) == "jaato-ws-s1 (enforce)"


# ----------------------------------------------------------------------
# Ask 1 — the idempotency skip in runner/session.py
# ----------------------------------------------------------------------


def test_idempotency_skip_stays_mode_tolerant_but_announces(caplog):
    """Re-entering the same profile would not change its mode, so the skip
    itself is correctly mode-blind.  What it must not do is let "already
    confined" stand in for "there is a boundary" — the commonest cascade
    path would otherwise be the one that says nothing."""
    from server.runner.session import _maybe_self_confine
    from shared.session_envelope import SessionInitEnvelope

    envelope = SessionInitEnvelope(
        session_id="s1", workspace_path="/tmp/ws", profile_name="jaato-ws-s1",
        provider_name="stub", model_name="stub",
    )
    with patch(
        "server.runner.bootstrap.current_confinement",
        return_value=parse_label("jaato-ws-s1 (complain)"),
    ), patch(
        "server.runner.bootstrap.verify_thread_confinement",
    ), patch(
        "server.runner.bootstrap.confine_to_profile",
    ) as confine_mock, caplog.at_level(
        logging.INFO, logger="server.runner.session",
    ):
        _maybe_self_confine(envelope)

    confine_mock.assert_not_called()           # still idempotent
    warnings = [
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    ]
    assert any("NOT confined" in m for m in warnings), warnings


def test_idempotency_skip_is_quiet_when_enforcing(caplog):
    from server.runner.session import _maybe_self_confine
    from shared.session_envelope import SessionInitEnvelope

    envelope = SessionInitEnvelope(
        session_id="s1", workspace_path="/tmp/ws", profile_name="jaato-ws-s1",
        provider_name="stub", model_name="stub",
    )
    with patch(
        "server.runner.bootstrap.current_confinement",
        return_value=parse_label("jaato-ws-s1 (enforce)"),
    ), patch(
        "server.runner.bootstrap.verify_thread_confinement",
    ), patch(
        "server.runner.bootstrap.confine_to_profile",
    ), caplog.at_level(logging.INFO, logger="server.runner.session"):
        _maybe_self_confine(envelope)

    assert not [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "NOT confined" in r.getMessage()
    ]


# ----------------------------------------------------------------------
# Ask 2 — the persisted record
# ----------------------------------------------------------------------


def test_sandbox_mode_vocabulary_distinguishes_the_two_postures():
    assert sandbox_mode_for_profile(complain=False) == SANDBOX_MODE_APPARMOR
    assert sandbox_mode_for_profile(complain=True) == SANDBOX_MODE_APPARMOR_COMPLAIN

    # "was a profile provisioned" — true of both.
    assert sandbox_mode_is_apparmor(SANDBOX_MODE_APPARMOR)
    assert sandbox_mode_is_apparmor(SANDBOX_MODE_APPARMOR_COMPLAIN)
    assert not sandbox_mode_is_apparmor(SANDBOX_MODE_SOFT)
    assert not sandbox_mode_is_apparmor(None)

    # "was there a boundary" — true of one.
    assert sandbox_mode_is_enforced(SANDBOX_MODE_APPARMOR)
    assert not sandbox_mode_is_enforced(SANDBOX_MODE_APPARMOR_COMPLAIN)


def test_sandbox_mode_round_trips_through_the_session_record():
    """A new value in an existing string field, not a new key — so the
    record needs no version bump, and an older reader comparing
    ``== "apparmor"`` degrades to "not confined", which is TRUE."""
    from shared.plugins.session.base import SessionState
    from shared.plugins.session.serializer import (
        deserialize_session_state,
        serialize_session_state,
    )

    state = SessionState(
        session_id="s1",
        history=[],
        created_at=datetime(2026, 9, 13),
        updated_at=datetime(2026, 9, 13),
        sandbox_mode=SANDBOX_MODE_APPARMOR_COMPLAIN,
    )
    data = serialize_session_state(state)
    assert data["sandbox_mode"] == SANDBOX_MODE_APPARMOR_COMPLAIN
    assert deserialize_session_state(data).sandbox_mode == (
        SANDBOX_MODE_APPARMOR_COMPLAIN
    )
    assert data["sandbox_mode"] != SANDBOX_MODE_APPARMOR


def test_complain_provisioning_records_the_mode_not_a_boundary_claim(tmp_path):
    """The daemon-side half: a session whose profile was rendered in
    complain mode must not persist ``sandbox_mode: "apparmor"``."""
    from server.session_manager import SessionManager

    class _Mgr:
        def __init__(self, complain):
            self._complain = complain

        def is_available(self):
            return True

        def provision_profile(self, *a, **kw):
            return True

        def get_profile_name(self, session_id):
            return f"jaato-ws-{session_id}"

        def profile_is_complain_mode(self, session_id):
            return self._complain

    sm = SessionManager()
    sm._emit_to_client = lambda cid, ev: None

    for complain, expected in (
        (False, SANDBOX_MODE_APPARMOR),
        (True, SANDBOX_MODE_APPARMOR_COMPLAIN),
    ):
        sm._apparmor_manager = _Mgr(complain)
        name, mode = sm._provision_apparmor_for_session(
            session_id="s1",
            workspace_path=str(tmp_path),
            client_id="c1",
            config_root=None,
            env_file=None,
        )
        assert name == "jaato-ws-s1"
        assert mode == expected


def test_revive_rearms_confinement_for_either_apparmor_mode():
    """A complain-mode session still WANTED a profile; the mode is
    re-decided from the environment at the next provisioning, so the
    revive gate asks "was a profile provisioned", not "was it enforced"."""
    assert sandbox_mode_is_apparmor(SANDBOX_MODE_APPARMOR_COMPLAIN)


# ----------------------------------------------------------------------
# Ask 3 — announce it
# ----------------------------------------------------------------------


def test_complain_mode_requested_reads_the_documented_spellings(monkeypatch):
    for value in ("1", "true", "TRUE", "yes", "on", " 1 "):
        monkeypatch.setenv(COMPLAIN_ENV_VAR, value)
        assert complain_mode_requested(), value
    for value in ("", "0", "false", "no", "off"):
        monkeypatch.setenv(COMPLAIN_ENV_VAR, value)
        assert not complain_mode_requested(), value
    monkeypatch.delenv(COMPLAIN_ENV_VAR, raising=False)
    assert not complain_mode_requested()


def test_complain_generation_warns_once_per_process(monkeypatch, caplog):
    import server.apparmor as apparmor_mod

    monkeypatch.setenv(COMPLAIN_ENV_VAR, "1")
    monkeypatch.setattr(apparmor_mod, "_complain_announced", __import__(
        "threading").Event())

    with caplog.at_level(logging.WARNING, logger="server.apparmor"):
        apparmor_mod.announce_complain_mode_once()
        apparmor_mod.announce_complain_mode_once()

    warnings = [
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    ]
    assert len(warnings) == 1, warnings
    assert COMPLAIN_ENV_VAR in warnings[0]
    assert SANDBOX_MODE_APPARMOR_COMPLAIN in warnings[0]


def test_no_announcement_when_the_knob_is_unset(monkeypatch, caplog):
    import server.apparmor as apparmor_mod

    monkeypatch.delenv(COMPLAIN_ENV_VAR, raising=False)
    monkeypatch.setattr(apparmor_mod, "_complain_announced", __import__(
        "threading").Event())
    with caplog.at_level(logging.WARNING, logger="server.apparmor"):
        apparmor_mod.announce_complain_mode_once()
    assert not caplog.records


def test_rendering_a_complain_profile_announces_and_records(monkeypatch, tmp_path):
    """Wired at the generation site, not merely available: the env read
    that used to be an inline literal now goes through the one predicate,
    and the manager remembers what it rendered."""
    import threading

    import server.apparmor as apparmor_mod

    monkeypatch.setenv(COMPLAIN_ENV_VAR, "1")
    monkeypatch.setattr(apparmor_mod, "_complain_announced", threading.Event())

    mgr = apparmor_mod.AppArmorManager(
        workspace_root=str(tmp_path), profile_dir=str(tmp_path / "profiles"),
    )
    rendered = mgr._render_profile("s1", str(tmp_path))

    assert "flags=(attach_disconnected, complain)" in rendered
    assert mgr.profile_is_complain_mode("s1") is True
    assert apparmor_mod._complain_announced.is_set()


def test_rendering_an_enforcing_profile_records_no_complain(monkeypatch, tmp_path):
    import threading

    import server.apparmor as apparmor_mod

    monkeypatch.delenv(COMPLAIN_ENV_VAR, raising=False)
    monkeypatch.setattr(apparmor_mod, "_complain_announced", threading.Event())

    mgr = apparmor_mod.AppArmorManager(
        workspace_root=str(tmp_path), profile_dir=str(tmp_path / "profiles"),
    )
    rendered = mgr._render_profile("s1", str(tmp_path))

    assert "complain" not in rendered.split("flags=(")[1].split(")")[0]
    assert mgr.profile_is_complain_mode("s1") is False
    assert not apparmor_mod._complain_announced.is_set()


# ----------------------------------------------------------------------
# Ask 4 — require_confinement means enforce
# ----------------------------------------------------------------------


def _shell_plugin(tmp_path, **config):
    from shared.plugins.interactive_shell.plugin import create_plugin

    plugin = create_plugin()
    plugin.initialize({"workspace_root": str(tmp_path), **config})
    return plugin


def test_require_confinement_refuses_a_complain_mode_child(tmp_path):
    """The sharpest row of the issue's table: the strictest fail-closed
    knob in the tree passed while the kernel blocked nothing.  The
    transition into ``//child (complain)`` succeeds — that is precisely
    what made the callback's presence worthless as evidence."""
    plugin = _shell_plugin(tmp_path, require_confinement=True)
    plugin.set_apparmor_child_transition_callback(lambda: None)
    try:
        with patch(
            # ``create=True``: this asserts BEHAVIOUR against a tree that
            # has no such seam.  On the unfixed code the patched name is
            # simply unused and ``_confinement_refusal`` returns None for
            # every label — which IS #1014's fourth row.
            "shared.plugins.interactive_shell.plugin.read_thread_label",
            create=True,
            return_value=parse_label("jaato-ws-s1//child (complain)"),
        ):
            refusal = plugin._confinement_refusal()
        assert refusal is not None
        assert COMPLAIN_ENV_VAR in refusal["error"]
        assert "not enforcing" in refusal["error"]
    finally:
        plugin.shutdown()


def test_require_confinement_refuses_a_mode_less_label(tmp_path):
    """Absence of a reported mode is not evidence of a boundary, and the
    strictest knob in the tree is the last place to be generous about it."""
    plugin = _shell_plugin(tmp_path, require_confinement=True)
    plugin.set_apparmor_child_transition_callback(lambda: None)
    try:
        with patch(
            # ``create=True``: this asserts BEHAVIOUR against a tree that
            # has no such seam.  On the unfixed code the patched name is
            # simply unused and ``_confinement_refusal`` returns None for
            # every label — which IS #1014's fourth row.
            "shared.plugins.interactive_shell.plugin.read_thread_label",
            create=True,
            return_value=parse_label("jaato-ws-s1//child"),
        ):
            assert plugin._confinement_refusal() is not None
    finally:
        plugin.shutdown()


def test_require_confinement_allows_an_enforcing_child(tmp_path):
    plugin = _shell_plugin(tmp_path, require_confinement=True)
    plugin.set_apparmor_child_transition_callback(lambda: None)
    try:
        with patch(
            # ``create=True``: this asserts BEHAVIOUR against a tree that
            # has no such seam.  On the unfixed code the patched name is
            # simply unused and ``_confinement_refusal`` returns None for
            # every label — which IS #1014's fourth row.
            "shared.plugins.interactive_shell.plugin.read_thread_label",
            create=True,
            return_value=parse_label("jaato-ws-s1//child (enforce)"),
        ):
            assert plugin._confinement_refusal() is None
    finally:
        plugin.shutdown()


def test_without_require_confinement_the_label_is_never_read(tmp_path):
    """The default posture must not gain a ``/proc`` read per spawn: the
    knob is opt-in, and the unconfined WARNING already covers the rest."""
    plugin = _shell_plugin(tmp_path)
    plugin.set_apparmor_child_transition_callback(lambda: None)
    try:
        with patch(
            # ``create=True``: this asserts BEHAVIOUR against a tree that
            # has no such seam.  On the unfixed code the patched name is
            # simply unused and ``_confinement_refusal`` returns None for
            # every label — which IS #1014's fourth row.
            "shared.plugins.interactive_shell.plugin.read_thread_label",
            create=True,
        ) as read_mock:
            assert plugin._confinement_refusal() is None
        read_mock.assert_not_called()
    finally:
        plugin.shutdown()


def test_no_transition_still_refuses_with_its_own_wording(tmp_path):
    """Two different failures, two different next moves: "there is no
    profile" and "there is a profile the kernel ignores"."""
    plugin = _shell_plugin(tmp_path, require_confinement=True)
    try:
        refusal = plugin._confinement_refusal()
        assert refusal is not None
        assert "no AppArmor child profile is active" in refusal["error"]
    finally:
        plugin.shutdown()


# ----------------------------------------------------------------------
# The component that already got it right keeps its answer
# ----------------------------------------------------------------------


def test_notebook_helper_behaviour_is_unchanged(tmp_path):
    from shared.plugins.notebook.kernel_sandbox import apparmor_enforced_profile

    cases = {
        "jaato-ws-a//child (enforce)": "jaato-ws-a//child",
        "jaato-ws-a//child (complain)": None,
        "jaato-ws-a": None,
        "unconfined": None,
        "": None,
    }
    for raw, expected in cases.items():
        with patch(
            "shared.plugins.notebook.kernel_sandbox.try_read_label",
            return_value=parse_label(raw),
        ):
            assert apparmor_enforced_profile() == expected, raw


def test_unreadable_proc_is_not_a_boundary(tmp_path):
    """Absence of evidence never reads as enforcement."""
    assert try_read_label(str(tmp_path / "nope")).enforced is False
    assert try_read_label(str(tmp_path / "nope")).confined is False
