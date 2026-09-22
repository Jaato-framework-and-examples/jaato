"""The scaffold TOPIC-extension seam: facade surface + discovery + dispatch.

The verb seam (``test_extension_verbs.py``) lets an external package add a
SUBCOMMAND.  This is its sibling for the one subcommand an agent actually
reads: ``explain``.  It exists because a package that contributes a daemon
EXTENSION — premium's reactor engine is the worked case — has something to say
and nothing to run, and the verb seam could only ever have offered it a
subcommand nobody would think to type.

No external package is required: a fake topic / fake entry point stands in, so
these pin the CONTRACT rather than premium's use of it.

Two shapes are covered because they fail differently.  An OWN topic must reach
every derived surface (the help line, the banner catalog, the dispatch) or it
is #994 again one layer out — dispatched and advertised nowhere.  An
EXTENSION must leave the built-in's own answer intact, including when it
raises, because the alternative is that installing a package deletes the
framework's documentation of itself.
"""

import json

import pytest

from jaato_server.shared.scaffold import api
from jaato_server.shared.scaffold import __main__ as cli


# --------------------------------------------------------------------------- #
# The stable facade
# --------------------------------------------------------------------------- #

def test_facade_exports_the_topic_surface():
    assert api.TOPIC_ENTRY_POINT_GROUP == "jaato.scaffold_topics"
    for name in ("ExplainTopic", "TopicRequest", "Rendered"):
        assert hasattr(api, name), name


def test_the_api_version_advertises_the_topic_seam():
    """A contributor gates on this; 1.0 predates topics entirely."""
    major, _, minor = api.SCAFFOLD_EXTENSION_API.partition(".")
    assert (int(major), int(minor)) >= (1, 1)


def test_topic_request_defaults_to_a_usable_workspace():
    """Every topic is HANDED a workspace, so none has to ask whether it got one."""
    assert api.TopicRequest(topic="paths").workspace == "."


def test_explaintopic_protocol_is_runtime_checkable():
    class Ok:
        name, help, arg, extends, reads_workspace = "x", "h", "", "", False
        def render(self, request): return {}, ""

    assert isinstance(Ok(), api.ExplainTopic)

    class Missing:
        name = "y"
    assert not isinstance(Missing(), api.ExplainTopic)


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #

class _OwnTopic:
    name = "faketopic"
    help = "a fake contributed topic"
    arg = "[<filter>]"
    extends = ""
    reads_workspace = True

    def render(self, request):
        return ({"topic": request.topic, "name": request.name,
                 "workspace": request.workspace},
                f"faketopic: name={request.name} ws={request.workspace}")


class _NeedsAName:
    name = "picky"
    help = ""
    arg = "<name>"
    extends = ""
    reads_workspace = False

    def render(self, request):
        if not request.name:
            return {"error": "no name"}, "usage: explain picky <name>"
        return {"name": request.name}, f"picky: {request.name}"


class _Section:
    name = "fakesection"
    help = ""
    arg = ""
    extends = "paths"
    reads_workspace = False

    def render(self, request):
        return {"extra": True}, "EXTRA PATHS SECTION"


class _Minimal:
    """Only what the loader REQUIRES — name and render."""
    name = "minimal"

    def render(self, request):
        return {"ok": True}, "minimal topic"


class _FakeEP:
    def __init__(self, obj, name="fake"):
        self._obj, self.name = obj, name

    def load(self):
        return self._obj


@pytest.fixture
def topics(monkeypatch):
    """Install a set of fake contributed topics for one test."""
    def _install(*objs):
        monkeypatch.setattr("importlib.metadata.entry_points",
                            lambda *a, **k: [_FakeEP(o) for o in objs])
        cli.reset_external_topics()
    yield _install
    cli.reset_external_topics()


# --------------------------------------------------------------------------- #
# An own topic reaches every derived surface
# --------------------------------------------------------------------------- #

def test_a_contributed_topic_dispatches(topics, capsys):
    topics(_OwnTopic)                                   # class -> instantiated
    rc = cli.main(["explain", "faketopic", "somefilter", "--workspace", "/w"])
    assert rc == 0
    assert "faketopic: name=somefilter ws=/w" in capsys.readouterr().out


def test_an_instance_entry_point_is_accepted(topics, capsys):
    topics(_OwnTopic())                                 # already an instance
    assert cli.main(["explain", "faketopic"]) == 0
    assert "faketopic:" in capsys.readouterr().out


def test_a_contributed_topic_is_advertised_where_it_dispatches(topics):
    """#994's round trip, for the seam: dispatched <-> advertised."""
    topics(_OwnTopic)
    assert "faketopic [<filter>]" in cli._all_scopes_help()
    catalog = {row["scope"]: row for row in cli.scope_catalog()}
    assert "faketopic" in catalog
    assert catalog["faketopic"]["blurb"] == "a fake contributed topic"
    assert catalog["faketopic"]["reads_workspace"] is True
    assert cli._scope_renderer("faketopic") is not None


def test_the_unknown_scope_error_lists_contributed_topics(topics, capsys):
    """A reader who typos a topic is shown every topic that EXISTS."""
    topics(_OwnTopic)
    rc = cli.main(["explain", "nonsense"])
    assert rc == 2
    assert "faketopic" in capsys.readouterr().err


def test_a_contributed_topic_renders_json(topics, capsys):
    topics(_OwnTopic)
    cli.main(["explain", "faketopic", "--json", "--workspace", "/w"])
    assert json.loads(capsys.readouterr().out)["workspace"] == "/w"


def test_a_topic_signals_a_missing_name_the_way_a_builtin_does(topics, capsys):
    """An ``error`` key is exit 2 + stderr, never documentation on stdout."""
    topics(_NeedsAName)
    rc = cli.main(["explain", "picky"])
    out = capsys.readouterr()
    assert rc == 2
    assert "usage: explain picky <name>" in out.err
    assert out.out == ""
    assert cli.main(["explain", "picky", "here"]) == 0


def test_only_name_and_render_are_required(topics, capsys):
    """The smallest useful topic is two attributes and one method."""
    topics(_Minimal)
    assert cli.main(["explain", "minimal"]) == 0
    assert "minimal topic" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Extensions append; they never replace
# --------------------------------------------------------------------------- #

def test_a_section_is_appended_to_the_builtin_topic(topics, capsys):
    topics(_Section)
    assert cli.main(["explain", "paths"]) == 0
    out = capsys.readouterr().out
    assert "paths & isolation model:" in out          # the built-in, intact
    assert "EXTRA PATHS SECTION" in out               # and the section after it
    assert out.index("paths & isolation") < out.index("EXTRA PATHS SECTION")


def test_a_section_lands_under_extensions_never_in_the_builtins_keys(
        topics, capsys):
    """A contributor cannot redefine a documented key for a reader branching on it."""
    topics(_Section)
    cli.main(["explain", "paths", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["extensions"]["fakesection"] == {"extra": True}
    assert "daemon_global" in data and "per_session" in data


def test_a_raising_section_does_not_take_the_builtin_down(topics, capsys):
    """Installing a package must not be able to delete the framework's own docs."""
    class _Boom:
        name, help, arg, extends, reads_workspace = "boom", "", "", "paths", False
        def render(self, request): raise RuntimeError("kaboom")

    topics(_Boom)
    assert cli.main(["explain", "paths"]) == 0
    out = capsys.readouterr().out
    assert "paths & isolation model:" in out
    assert "kaboom" in out          # reported in place, named, not swallowed


def test_a_section_naming_an_unknown_topic_is_inert(topics, capsys):
    """Not an error: a topic a later release adds is not a mistake today."""
    class _Orphan:
        name, help, arg, extends, reads_workspace = "orphan", "", "", "nosuch", False
        def render(self, request): return {}, "ORPHAN"

    topics(_Orphan)
    assert cli.main(["explain", "paths"]) == 0
    assert "ORPHAN" not in capsys.readouterr().out
    assert "orphan" not in cli._all_scopes_help()     # and it is not a topic


# --------------------------------------------------------------------------- #
# Failure isolation
# --------------------------------------------------------------------------- #

def test_a_builtin_topic_wins_a_name_collision(topics, capsys):
    """An installed package cannot replace the framework's answer about itself."""
    class _Hijack:
        name, help, arg, extends, reads_workspace = "paths", "", "", "", False
        def render(self, request): return {}, "HIJACKED"

    topics(_Hijack)
    assert cli.main(["explain", "paths"]) == 0
    out = capsys.readouterr().out
    assert "HIJACKED" not in out
    assert "paths & isolation model:" in out


def test_a_topic_that_does_not_satisfy_the_protocol_is_skipped(topics):
    class _Broken:
        name = "broken"                 # no render()

    topics(_Broken())
    assert cli.main([]) == 0
    assert "broken" not in cli._all_scopes_help()


def test_an_entry_point_that_raises_on_load_is_skipped(topics, monkeypatch):
    class _Raises:
        name = "kaboom"
        def load(self): raise RuntimeError("boom")

    monkeypatch.setattr("importlib.metadata.entry_points",
                        lambda *a, **k: [_Raises()])
    cli.reset_external_topics()
    assert cli.main([]) == 0


def test_one_broken_contributor_does_not_hide_a_working_one(topics, capsys):
    """A diagnostic that cannot survive a broken contributor is not a diagnostic."""
    class _Broken:
        name = "broken"

    topics(_Broken(), _OwnTopic)
    assert cli.main(["explain", "faketopic"]) == 0
    assert "faketopic:" in capsys.readouterr().out


def test_with_no_contributors_every_surface_is_unchanged(topics):
    """The hard requirement: an unextended install behaves exactly as before."""
    topics()
    assert cli._all_scopes_help() == cli._SCOPES_HELP
    assert [r["scope"] for r in cli.scope_catalog()] == list(cli._SCOPES)
