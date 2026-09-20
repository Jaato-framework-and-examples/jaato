"""The first-interaction announcement -- EU AI Act, Article 50(1) (#1116).

``test_ai_disclosure_piece.py`` pins the half the MODEL reads: a prompt
constant telling it to answer truthfully when asked whether it is an AI.
That is the fallback.  Article 50(1) asks for something stronger -- that
the system be "designed and developed in such a way that the natural
persons concerned are informed" -- and the design half is the framework
saying so, unprompted, before the first turn.

``disclosure_announcement()`` rendered the text and NOTHING CALLED IT, so
a profile declaring ``interacts_with_persons: true`` started exactly as
any other.  This module pins the emission.

Five properties, each attached to a way it could silently stop holding:

A. one predicate.  The daemon's emit site and ``explain oversight`` both
   call :func:`~shared.ai_disclosure.announcement_for`, so a page saying
   "this profile announces" and a session that stays silent cannot
   disagree -- the two-implementations failure this tree keeps finding
   (#735, #950);
B. absent is not false.  A profile that declared nothing announces
   nothing: announcing on its behalf would put a legal statement in front
   of every session in every existing workspace;
C. the client can decline it, and only the client can.  The Act's "unless
   this is obvious from the point of view of a natural person" clause is
   asserted by the party that can see the screen -- and for that reason
   ``validate`` does not read the flag;
D. once per session, at CREATION -- never per turn, never on a revive, and
   the text reaches an attaching client through the state snapshot rather
   than through a second emission;
E. the text is deterministic from the profile, so this asserts it
   byte-for-byte rather than describing its shape.
"""

from __future__ import annotations

import ast
from pathlib import Path

from shared.ai_disclosure import (
    CLIENT_DISCLOSES,
    DECLARED_NO_PERSONS,
    NOT_DECLARED,
    announcement_for,
)
from shared.tests.reversion import Reversion

_MANAGER = "jaato-server/server/session_manager.py"
_CORE = "jaato-server/server/core.py"
_DISCLOSURE = "jaato-server/shared/ai_disclosure.py"
_EXPLAIN = "jaato-server/shared/scaffold/explain.py"

REVERSIONS = [
    Reversion(
        target=_MANAGER,
        find="        self._announce_ai_interaction(client_id, session)",
        replace="        pass  # reversion: the announcement is never emitted",
        because=(
            "a text nobody emits informs nobody -- the state #1116 reports, "
            "where disclosure_announcement() rendered the sentence and no "
            "call site existed"
        ),
        test="test_only_the_creation_path_announces",
    ),
    Reversion(
        target=_DISCLOSURE,
        find="    interacts = getattr(regulatory, \"interacts_with_persons\", None)\n    if interacts is not True:",
        replace="    interacts = getattr(regulatory, \"interacts_with_persons\", None)\n    if interacts is False:",
        because=(
            "absent is not false: a profile that declared nothing has made no "
            "determination, and announcing for it would put a legal statement "
            "in front of every existing workspace's sessions"
        ),
        test="test_an_undeclared_profile_announces_nothing",
    ),
    Reversion(
        target=_DISCLOSURE,
        find="    if client_discloses_ai:\n        return None, CLIENT_DISCLOSES",
        replace="    if False:\n        return None, CLIENT_DISCLOSES",
        because=(
            "the Act's 'unless this is obvious' clause is the client's to "
            "assert; a flag nothing reads is a flag a client cannot use"
        ),
        test="test_a_client_that_discloses_already_suppresses_it",
    ),
    Reversion(
        target=_EXPLAIN,
        find='        "announcement": _announcement(prof),',
        replace='        "announcement": {"text": None, "withheld_reason": None},',
        because=(
            "explain oversight must read the SAME predicate the daemon does, "
            "or the page and the behaviour drift apart silently"
        ),
        test="test_explain_oversight_reads_the_same_predicate",
    ),
]


# --------------------------------------------------------------- fixtures

class _Reg:
    """A duck-typed ``regulatory:`` block.

    ``announcement_for`` reads its argument with ``getattr`` on purpose --
    ``shared/ai_disclosure.py`` is stdlib-only so the runtime can append
    the instruction piece before plugin discovery -- so a stand-in is the
    honest way to exercise it.  ``test_the_real_block_satisfies_the_duck``
    below pins that the real dataclass still fits.
    """

    def __init__(self, interacts=None, provider_name=None, text=None):
        self.interacts_with_persons = interacts
        self.provider_name = provider_name
        self.disclosure_text = text


class _Server:
    """The two things ``JaatoServer.disclosure_announcement`` reads."""

    def __init__(self, reg, client_discloses=False):
        self._profile = type("P", (), {"regulatory": reg})()
        self._presentation_context = type(
            "PC", (), {"client_discloses_ai": client_discloses})()

    disclosure_announcement = None  # replaced in __init__ below


def _server(reg, client_discloses=False):
    """A real ``JaatoServer.disclosure_announcement`` bound to a stand-in.

    The METHOD under test is the framework's; only its two inputs are
    fabricated.  Constructing a whole ``JaatoServer`` would need a
    provider, a registry and a workspace, none of which this question
    involves.
    """
    from server.core import JaatoServer
    srv = _Server(reg, client_discloses)
    # #1157: the text accessor delegates to the (text, reason) one, so a
    # stand-in carries both -- the same predicate, both halves of its answer.
    srv.disclosure_decision = JaatoServer.disclosure_decision.__get__(srv, _Server)
    return JaatoServer.disclosure_announcement.__get__(srv, _Server)


# ------------------------------------------------------------ A. one predicate

def test_explain_oversight_reads_the_same_predicate():
    """Through ``_profile_oversight``, the dict the page renders.

    Asserting on ``_announcement`` alone would be blind to the one way
    this drifts: the helper staying correct while the page stops calling
    it.  The reversion meta-guard says so -- it reported this test as
    decorative until it went through the assembler.
    """
    from shared.plugins.subagent.config import RegulatoryProfileConfig
    from shared.scaffold.explain import _profile_oversight

    # The REAL block here, not the duck: `_profile_oversight` renders the
    # whole regulatory dict beside the announcement, so this also pins that
    # the two readings of one block agree.
    reg = RegulatoryProfileConfig.from_dict(
        {"interacts_with_persons": True, "provider": {"name": "Acme GmbH"}})
    prof = type("Prof", (), {"regulatory": reg})()
    expected = ("You are interacting with an AI system operated by "
                "Acme GmbH, not with a human being.")
    assert _profile_oversight(prof)["announcement"] == {
        "text": expected, "withheld_reason": None}

    # And it is the SAME sentence the daemon would emit for that profile.
    assert _server(reg)() == expected

    silent = type("Prof", (), {"regulatory": None})()
    assert _profile_oversight(silent)["announcement"] == {
        "text": None, "withheld_reason": NOT_DECLARED}


def test_the_page_names_which_of_the_three_reasons_withheld_it():
    from shared.scaffold.explain import _ANNOUNCEMENT_REASONS, _announcement_lines

    # A page that printed a bare "none" would leave an author unable to tell
    # a profile that DECLINED to declare from one that declared false.
    assert set(_ANNOUNCEMENT_REASONS) == {
        NOT_DECLARED, DECLARED_NO_PERSONS, CLIENT_DISCLOSES}
    for reason, expected in _ANNOUNCEMENT_REASONS.items():
        rendered = "\n".join(
            _announcement_lines({"text": None, "withheld_reason": reason}))
        assert expected in rendered

    spoken = "\n".join(_announcement_lines(
        {"text": "You are talking to a bot.", "withheld_reason": None}))
    assert "You are talking to a bot." in spoken
    assert "client_discloses_ai" in spoken, (
        "the page must name the per-connection half it cannot evaluate")


def test_the_real_block_satisfies_the_duck():
    from shared.plugins.subagent.config import RegulatoryProfileConfig

    reg = RegulatoryProfileConfig.from_dict(
        {"interacts_with_persons": True, "provider": {"name": "Acme GmbH"}})
    text, reason = announcement_for(reg)
    assert reason is None
    assert text == ("You are interacting with an AI system operated by "
                    "Acme GmbH, not with a human being.")


# --------------------------------------------------------- B. absent is not false

def test_an_undeclared_profile_announces_nothing():
    assert announcement_for(None) == (None, NOT_DECLARED)
    assert announcement_for(_Reg()) == (None, NOT_DECLARED)
    assert _server(_Reg())() is None


def test_a_profile_that_declared_false_announces_nothing_and_says_so():
    # Distinguishable from the above: `validate` reports the first as
    # `disclosure_absent` and has nothing to say about the second.
    assert announcement_for(_Reg(False)) == (None, DECLARED_NO_PERSONS)


# ------------------------------------------------------ C. the client's clause

def test_a_client_that_discloses_already_suppresses_it():
    reg = _Reg(True, "Acme GmbH")
    assert announcement_for(reg, client_discloses_ai=True) == (
        None, CLIENT_DISCLOSES)
    assert _server(reg, client_discloses=True)() is None
    assert _server(reg, client_discloses=False)() is not None


def test_the_flag_defaults_to_not_disclosing():
    from jaato_sdk.events import PresentationContext

    # A client that has not said it discloses has not disclosed, so the
    # framework announces.  The safe direction: the cost of being wrong is
    # a redundant sentence, not an undisclosed AI.
    assert PresentationContext().client_discloses_ai is False


def test_validate_does_not_read_the_per_connection_flag():
    # A per-connection assertion cannot answer a question about a profile,
    # so `disclosure_absent` must stay exactly what it was.
    src = Path("jaato-server/shared/scaffold/validate.py").read_text()
    assert "client_discloses_ai" not in src


# ------------------------------------------------------------- D. once, at create

def test_a_session_that_interacts_with_persons_announces_at_creation():
    from jaato_sdk.events import AgentOutputEvent
    from server.session_manager import SessionManager

    emitted = []
    mgr = SessionManager.__new__(SessionManager)
    mgr._emit_to_client = lambda cid, ev: emitted.append((cid, ev))

    session = type("S", (), {})()
    session.server = type("Srv", (), {
        "disclosure_announcement": lambda self=None: "You are talking to a bot.",
    })()
    SessionManager._announce_ai_interaction(mgr, "client_1", session)

    assert len(emitted) == 1
    client_id, event = emitted[0]
    assert client_id == "client_1"
    assert isinstance(event, AgentOutputEvent)
    assert event.source == "system"
    assert event.text == "You are talking to a bot."


def test_a_session_that_announces_nothing_emits_nothing():
    from server.session_manager import SessionManager

    emitted = []
    mgr = SessionManager.__new__(SessionManager)
    mgr._emit_to_client = lambda cid, ev: emitted.append(ev)
    session = type("S", (), {})()
    session.server = type("Srv", (), {
        "disclosure_announcement": lambda self=None: None})()
    SessionManager._announce_ai_interaction(mgr, "client_1", session)
    assert emitted == []


def test_only_the_creation_path_announces():
    """No second call site, and none on a revive.

    Source-level, because the property is *absence*: a behavioural test
    would have to know about a call site to exercise it, and the failure
    guarded against is a call site nobody thought about.  A revived
    session continues a conversation that was already disclosed to, and a
    second announcement would tell a person something they were told
    before the transcript they are looking at begins.
    """
    tree = ast.parse(Path(_MANAGER).read_text())
    callers = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_announce_ai_interaction"
    ]
    assert len(callers) == 1, (
        f"expected exactly one emit site, found {len(callers)}")

    enclosing = [
        fn.name for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(c is callers[0] for c in ast.walk(fn))
    ]
    assert "_create_session_impl" in enclosing, (
        f"the one emit site must be the CREATE path, found in {enclosing}")


def test_an_attaching_client_reads_the_text_off_the_state_snapshot():
    from jaato_sdk.events import SessionInfoEvent
    from server.session_manager import _disclosure_announcement_of

    assert "disclosure_announcement" in SessionInfoEvent.model_fields
    assert SessionInfoEvent().disclosure_announcement is None

    srv = type("Srv", (), {
        "disclosure_announcement": lambda self=None: "You are talking to a bot."})()
    assert _disclosure_announcement_of(srv) == "You are talking to a bot."


def test_a_server_without_the_accessor_gets_the_pre_1_15_behaviour():
    """#881's rule: a duck-typed double must not raise inside the snapshot.

    ``_build_session_info_event`` builds the state every client waits on,
    so an ``AttributeError`` there costs the caller its session, not a
    field.  A real ``JaatoServer`` never takes this path.
    """
    from server.session_manager import _disclosure_announcement_of

    assert _disclosure_announcement_of(None) is None
    assert _disclosure_announcement_of(object()) is None

    class Exploding:
        def disclosure_announcement(self):
            raise RuntimeError("boom")

    assert _disclosure_announcement_of(Exploding()) is None


# ------------------------------------------------------------- E. the text

def test_the_text_is_deterministic_from_the_profile():
    assert announcement_for(_Reg(True, "Acme GmbH"))[0] == (
        "You are interacting with an AI system operated by Acme GmbH, "
        "not with a human being.")
    assert announcement_for(_Reg(True))[0] == (
        "You are interacting with an AI system, not with a human being.")
    assert announcement_for(_Reg(True, "Acme", "  Hi, I am a bot.  "))[0] == (
        "Hi, I am a bot.")
