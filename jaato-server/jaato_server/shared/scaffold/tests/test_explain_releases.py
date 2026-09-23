"""``jaato-scaffold explain releases`` — the long-form half of the notifier.

``jaato-doctor`` reports one preflight line; this renders the same
:mod:`jaato_sdk.release_channels` report with every channel's answer spelled
out.  They are two drawings of one report, which is what stops a version
reading as "newest" on one surface and "behind" on the other — so what is
pinned here is the DRAWING and the fact that the topic is reachable, never
the ordering or the fetching (``test_release_channels.py`` owns those).

**No test here reaches the network**: the report is substituted.
"""

import json

import pytest

from jaato_sdk import release_channels as rc
from jaato_server.shared.scaffold import __main__ as cli
from jaato_server.shared.scaffold import releases as mod


def _report(*, enabled=True, rows=(), errors=()):
    by_name = {c.name: c for c in rc.CHANNELS}
    report = rc.ReleaseReport(enabled=enabled, errors=list(errors))
    for name, installed, channels in rows:
        dist = rc.DistStatus(name=name, installed=installed)
        for channel_name, latest, verdict, error, unparseable in channels:
            dist.channels.append(rc.ChannelStatus(
                channel=by_name[channel_name], latest=latest, verdict=verdict,
                error=error, unparseable=list(unparseable)))
        report.distributions.append(dist)
    return report


@pytest.fixture
def served(monkeypatch):
    def install(report):
        monkeypatch.setattr(mod._rc, "check_releases", lambda **kw: report)
    return install


_BOTH_STALE = [("jaato-sdk", "0.22.0",
                [("pypi", "0.23.0", "update", None, []),
                 ("testpypi", "0.24.0rc1", "update", None, [])])]


# --------------------------------------------------------------------------
# The topic exists and is dispatched
# --------------------------------------------------------------------------

def test_the_topic_is_registered_and_advertised():
    """A renderer no scope dispatches is a page nobody can reach."""
    scopes = {row["scope"] for row in cli.scope_catalog()}
    assert "releases" in scopes
    assert "releases" in cli._SIMPLE_SCOPES


def test_it_is_not_folded_into_explain_dependencies():
    """`dependencies` is an OFFLINE read of the installed tree and stays one.

    Giving that verb an egress would change what running it means, which is
    why the network-touching answer is its own topic.
    """
    import inspect
    from jaato_server.shared.scaffold import dependencies as deps
    source = inspect.getsource(deps)
    assert "release_channels" in source, "it may share the distribution set"
    assert "check_releases" not in source, (
        "explain dependencies must not ask an index — that belongs to "
        "`explain releases`, which says so"
    )


# --------------------------------------------------------------------------
# What it draws
# --------------------------------------------------------------------------

def test_it_names_both_channels_and_what_a_version_there_means(served):
    served(_report(rows=_BOTH_STALE))
    _, text = mod.releases()
    assert "production release" in text and "release candidate" in text
    assert "PyPI" in text and "TestPyPI" in text


def test_an_update_is_rendered_with_the_command_that_installs_it(served):
    served(_report(rows=_BOTH_STALE))
    _, text = mod.releases()
    assert "0.23.0" in text and "0.24.0rc1" in text
    assert "pip install -U jaato-sdk" in text          # production
    assert "--pre" in text and "test.pypi.org" in text  # candidate


def test_both_installers_are_offered_for_every_stale_channel(served):
    """A uv user must not have to translate the candidate command themselves.

    It is not a rename: uv's index precedence is the reverse of pip's, so the
    obvious translation resolves the PyPI stable and says nothing about it.
    """
    served(_report(rows=_BOTH_STALE))
    _, text = mod.releases()
    assert "uv pip install -U jaato-sdk" in text                  # production
    assert "--prerelease allow" in text                           # candidate
    assert "--index-strategy unsafe-best-match" in text


def test_the_renderer_names_no_installer_of_its_own(served):
    """Both surfaces loop over the channel, so neither can drift from it.

    A renderer that spelled `pip` and `uv` itself would keep rendering two
    when a channel documents three — the mock-spoke-the-client's-vocabulary
    shape this repository keeps finding.
    """
    import inspect
    source = inspect.getsource(mod)
    assert "install_commands" in source
    assert "uv_install_command" not in source, (
        "the renderer must not reach for one installer by name"
    )


def test_nothing_newer_says_so_rather_than_printing_an_empty_section(served):
    served(_report(rows=[("jaato-sdk", "0.22.0",
                          [("pypi", "0.22.0", "current", None, []),
                           ("testpypi", "0.22.0", "current", None, [])])]))
    _, text = mod.releases()
    assert "nothing newer is published" in text
    assert "to upgrade" not in text


def test_an_unreachable_index_shows_its_reason_and_claims_nothing(served):
    served(_report(rows=[("jaato-sdk", "0.22.0",
                          [("pypi", None, "unknown",
                            "cannot reach https://pypi.org (timed out)", [])])],
                   errors=["cannot reach https://pypi.org (timed out)"]))
    _, text = mod.releases()
    assert "no verdict" in text
    assert "timed out" in text
    assert "this is the newest build here" not in text


def test_a_version_it_could_not_order_is_named_not_dropped(served):
    """The release most worth naming is the one in a spelling we do not know."""
    served(_report(rows=[("jaato-sdk", "0.22.0",
                          [("pypi", "0.22.0", "current", None,
                            ["2026.09.18-nightly"])])]))
    _, text = mod.releases()
    assert "2026.09.18-nightly" in text
    assert "not ordered" in text


def test_it_says_how_to_switch_the_asking_off(served):
    served(_report(rows=_BOTH_STALE))
    _, text = mod.releases()
    assert rc.ENV_SWITCH in text


def test_disabled_renders_the_switch_rather_than_an_empty_table(served):
    served(_report(enabled=False))
    data, text = mod.releases()
    assert data["enabled"] is False
    assert rc.ENV_SWITCH in text
    assert "disabled" in text


def test_no_distributions_is_stated_not_rendered_as_success(served):
    served(_report(errors=["no jaato distributions are installed here"]))
    _, text = mod.releases()
    assert "no jaato distributions are installed here" in text
    assert "nothing newer is published" not in text


# --------------------------------------------------------------------------
# --json
# --------------------------------------------------------------------------

def test_the_json_view_is_serialisable_and_keyed_by_channel(served):
    served(_report(rows=_BOTH_STALE))
    data, _ = mod.releases()
    json.dumps(data)                                   # must not raise
    channels = data["distributions"][0]["channels"]
    assert [c["channel"] for c in channels] == ["pypi", "testpypi"]
    assert [c["verdict"] for c in channels] == ["update", "update"]


def test_the_renderer_passes_its_knobs_through(monkeypatch):
    """A timeout the caller sets and the module ignores is a knob that lies."""
    seen = {}

    def record(**kw):
        seen.update(kw)
        return _report(errors=["none installed"])

    monkeypatch.setattr(mod._rc, "check_releases", record)
    mod.releases(timeout=2.5, refresh=True)
    assert seen["timeout"] == 2.5 and seen["refresh"] is True
    seen.clear()
    mod.releases()
    assert seen["timeout"] == rc.DEFAULT_TIMEOUT and seen["refresh"] is False
