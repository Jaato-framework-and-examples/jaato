"""A topic this venv does not have is asked of the daemon that does.

``jaato-scaffold explain`` introspects the framework installed in the CALLING
process.  That is the whole answer while the CLI and the daemon share a
virtualenv, and silently wrong the moment they do not — which is the normal
shape of a deployed application: ``jaato-sdk`` (and ``jaato-server``) in the
application's own ``.venv``, driving a daemon owned by a different user over
IPC.  There are then TWO installs, and a topic contributed through
``jaato.scaffold_topics`` by a package only the daemon has exists in one of
them.

Measured on exactly that pair, before this seam::

    $ jaato-scaffold explain reactors
    unknown explain scope 'reactors' — one of: plugins | plugin <name> | ...

The refusal is indistinguishable from *no such topic exists*, so it sends a
reader looking for a feature they already have.  The entry-point seam is not
the gap — it works, in the process that has the package.  What was missing was
a way to ask the other process.

What this file guards is the set of properties that make asking honest, each
attached to the way it goes wrong without them:

* **one dispatch, not two** — the daemon renders through the same
  ``render_topic`` the CLI calls, so the two installs cannot grow separate
  opinions about what a topic answers;
* **a topic this venv CAN answer is answered locally** — quietly giving an
  offline introspection an egress changes what running it means, the argument
  ``explain releases`` already makes about being its own topic;
* **``reached`` is not ``ok``** — *the daemon answered and does not have it
  either* and *no daemon could be asked* are different facts, and collapsing
  them reproduces the confusion the seam removes;
* **a remote answer says whose install produced it** — on this deployment the
  two installs are two machines' worth of different packages, and a reader who
  cannot tell them apart cannot tell which to change;
* **the diagnostic never SPAWNS a daemon** — a report about a process the
  report created is a report about the wrong process.
"""

import ast
from pathlib import Path

from shared.scaffold import remote as _remote
from shared.scaffold.__main__ import render_topic
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

_ROUTER = "jaato-server/server/command_router.py"
_REMOTE = "jaato-server/shared/scaffold/remote.py"
_MAIN = "jaato-server/shared/scaffold/__main__.py"

REVERSIONS = [
    Reversion(
        target=_REMOTE,
        find='''    client = IPCClient(socket_path, client_type=ClientType.API,
                       auto_start=False)''',
        replace='''    client = IPCClient(socket_path, client_type=ClientType.API)''',
        test="test_asking_a_daemon_never_starts_one",
        because="a diagnostic spawns a daemon as a side effect of being asked "
                "a question, and then answers about the process it just "
                "created rather than the one serving the reader's application",
    ),
    Reversion(
        target=_MAIN,
        find='''        if _scope_renderer(scope) is None:
            rc, note = _render_from_daemon(''',
        replace='''        if True:
            rc, note = _render_from_daemon(''',
        test="test_only_a_topic_this_venv_lacks_is_worth_a_socket",
        because="a usage error about the caller's own command line is taken to "
                "a daemon, which answers a question nobody posed",
    ),
    Reversion(
        target=_MAIN,
        find='''        # On the FALLBACK, the daemon's refusal must not replace the local
        # one: its list is the DAEMON's topics, and a reader shown that list
        # concludes a topic their own install has does not exist.  The local
        # refusal is the one they can act on without a socket, so it prints,
        # and the note below keeps "we asked and it is not there either"
        # distinguishable from "nobody looked" — the distinction `reached`
        # exists to preserve, one layer out.
        return None, (f"(also asked the daemon at {answer.socket_path} "''',
        replace='''        print(answer.error or "", file=sys.stderr)
        return 2, (f"(also asked the daemon at {answer.socket_path} "''',
        test="test_a_daemon_refusal_never_replaces_the_local_topic_list",
        because="a daemon's refusal replaces the local one on the FALLBACK, "
                "so a reader who typos is shown the DAEMON's topic list and "
                "concludes a topic their own install has does not exist",
    ),
    Reversion(
        target=_ROUTER,
        find="            from shared.scaffold.__main__ import render_topic, scope_catalog",
        replace="            from shared.scaffold.explain import overview as render_topic, "
                "installed_plugins as scope_catalog",
        test="test_the_daemon_renders_through_the_one_dispatch",
        because="the daemon grows a second dispatch, free to disagree with the "
                "CLI's about what a topic answers — the failure the seam "
                "exists to remove, reproduced over a socket",
    ),
]


def _router_source() -> str:
    return (Path(__file__).resolve().parents[2]
            / "server" / "command_router.py").read_text()


def _main_source() -> str:
    return (Path(__file__).resolve().parents[1]
            / "scaffold" / "__main__.py").read_text()


def _remote_source() -> str:
    return (Path(__file__).resolve().parents[1]
            / "scaffold" / "remote.py").read_text()


# --------------------------------------------------------------- one dispatch

def _handler_node() -> ast.FunctionDef:
    """The daemon-side handler, as AST.

    Source rather than import: the properties below are about what the code
    SAYS, and a behavioural probe passes against a second dispatch that
    happens to agree today.
    """
    tree = ast.parse(_router_source())
    node = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_handle_scaffold_explain"),
        None)
    assert node is not None, "_handle_scaffold_explain is gone"
    return node


def test_the_daemon_renders_through_the_one_dispatch():
    """The handler calls ``render_topic``; it does not re-implement it.

    Asserted on the SOURCE rather than by driving the handler, because a
    behavioural test passes against a second dispatch that happens to agree
    today — and agreeing today is exactly what a second dispatch does until
    somebody edits one of them.
    """
    handler = _handler_node()
    imported = {
        alias.name
        for n in ast.walk(handler)
        if isinstance(n, ast.ImportFrom) and n.module == "shared.scaffold.__main__"
        for alias in n.names
    }
    assert "render_topic" in imported, (
        "the daemon must render through shared.scaffold.__main__.render_topic — "
        "the same function the CLI calls")

    called = {n.func.id for n in ast.walk(handler)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "render_topic" in called, "render_topic is imported and never called"


def test_the_answer_carries_the_daemon_s_own_catalog():
    """A refusal lists the topics THIS install has, not the caller's.

    The caller's catalog is by construction the wrong one — it is the one that
    just failed to answer — so a refusal that echoed it would repeat the
    unhelpful local message over a socket.
    """
    handler = _handler_node()
    called = {n.func.id for n in ast.walk(handler)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "scope_catalog" in called, (
        "the daemon must read its OWN topic catalog")

    # Every answer must carry it, so the field is bound once in the `answer`
    # helper rather than per call site — which is what makes the promise hold
    # for the failure paths too, where a per-site keyword is what gets
    # forgotten.
    answer = next((n for n in ast.walk(handler)
                   if isinstance(n, ast.FunctionDef) and n.name == "answer"),
                  None)
    assert answer is not None, "the one-door `answer` helper is gone"
    bound = {kw.arg for n in ast.walk(answer)
             if isinstance(n, ast.Call) for kw in n.keywords}
    assert "topics" in bound, (
        "the catalog must be bound in `answer`, so a refusal carries it too")


# ------------------------------------------------- local answers stay local

def test_only_a_topic_this_venv_lacks_is_worth_a_socket():
    """The fallback is gated on the topic being unresolvable HERE.

    A usage error — a topic that needs a name, given none — is about the
    caller's own command line, and taking it to a daemon answers a question
    nobody posed while paying a connect for it.
    """
    src = (Path(__file__).resolve().parents[1]
           / "scaffold" / "__main__.py").read_text()
    body = src.split("def _cmd_explain(", 1)[1]
    fallback = body.split("_render_from_daemon(None", 1)[0]
    assert "_scope_renderer(scope) is None" in fallback, (
        "the daemon fallback must be gated on this venv not having the topic")


def test_a_daemon_refusal_never_replaces_the_local_topic_list():
    """On the FALLBACK, the local refusal prints — the daemon's does not.

    The two refusals list different topic sets, and only one of them is the
    set the reader can use without a socket.  A reader who typos and is shown
    the DAEMON's list concludes a topic their own install *has* does not
    exist — a wrong answer produced by the fix, and one that appears only on
    a machine that happens to be running a daemon, so it is asserted rather
    than left to ambient state.

    ``--connect`` is the deliberate exception and is not covered here: there
    the reader named that daemon, so its list is the one they asked about.
    """
    body = _main_source().split("def _render_from_daemon(", 1)[1]
    body = body.split("\ndef ", 1)[0]
    fallback = body.split("if not answer.ok:", 1)[1]
    fallback = fallback.split("if required:", 1)[1]
    # The `required` arm ends at its own return; what follows is the fallback.
    fallback = fallback.split("return 2,", 1)[1].split("\n", 1)[1]
    assert "return None," in fallback, (
        "the fallback must hand control back so the LOCAL refusal prints")
    assert "sys.stderr" not in fallback, (
        "the fallback must not print the daemon's topic list over the "
        "reader's own")


def test_the_fallback_still_says_the_daemon_was_asked():
    """*Asked and not there either* must stay distinct from *nobody looked*.

    ``reached`` keeps that distinction inside :mod:`remote`; dropping the note
    would lose it again at the one surface a person reads.
    """
    assert "also asked the daemon at" in _main_source()


def test_a_locally_answerable_topic_is_rendered_locally():
    """The control: ``render_topic`` answers without any transport at all."""
    ok, data, text, error = render_topic("paths", None, ".")
    assert ok and text and not error


# ------------------------------------- reached is not ok, and says which

def test_unreachable_is_not_the_same_answer_as_unknown():
    """An unaskable daemon must not read as *the topic does not exist*."""
    answer = _remote.ask_daemon("/nonexistent/jaato-guard.sock", "reactors",
                                None, timeout=1.0)
    assert answer.reached is False
    assert answer.ok is False
    assert answer.unreachable, "an unreachable daemon must say why"
    assert not answer.error, (
        "`error` is what the daemon SAID; nothing was said, so it stays empty "
        "— a caller reading it would report a refusal nobody made")


def test_asking_a_daemon_never_starts_one():
    """``auto_start`` is off, so the probe cannot spawn what it reports on.

    Asserted on the CALL, by AST, and read off the file beside this test.
    Two drafts of this one test were decorative and the reversion meta-guard
    caught both, which is worth recording because each failed differently:
    ``inspect.getsource`` reads the module the editable install pins — the
    real checkout, whatever the interpreter's cwd — so it never saw the
    sabotage at all; and a substring test for ``auto_start=False`` was then
    satisfied by the *comment* two lines above the call, which survives
    removing the keyword.  A guard on prose is a guard on nothing.
    """
    tree = ast.parse(_remote_source())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "IPCClient"]
    assert calls, "the probe no longer builds an IPCClient"
    for call in calls:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        assert "auto_start" in kwargs, (
            "the probe must not spawn a daemon: it would then answer about "
            "the process it created rather than the one serving the "
            "application")
        assert isinstance(kwargs["auto_start"], ast.Constant)
        assert kwargs["auto_start"].value is False


# ------------------------------------------------------------- attribution

def test_a_remote_answer_names_the_install_that_produced_it():
    """Two installs answering differently is the NORMAL state here."""
    answer = _remote.RemoteAnswer(reached=True, ok=True, topic="reactors",
                                  socket_path="/tmp/x.sock",
                                  server_version="9.9.9")
    line = _remote.attribution(answer)
    assert "/tmp/x.sock" in line and "9.9.9" in line
    assert "not this venv" in line


def test_attribution_survives_a_daemon_that_named_no_version():
    """A missing version degrades the byline; it never removes it."""
    line = _remote.attribution(
        _remote.RemoteAnswer(reached=True, ok=True, socket_path="/tmp/x.sock"))
    assert "/tmp/x.sock" in line and "unknown version" in line
