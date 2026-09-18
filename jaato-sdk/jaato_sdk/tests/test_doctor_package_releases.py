"""The doctor's package-releases check — the preflight half of the notifier.

jaato publishes production releases to PyPI and stages release candidates on
TestPyPI, and nothing in the framework asked either index anything, so the
only way to learn a release existed was to open the project page.  This check
is where a developer who runs one command before debugging finds out.

What is pinned here is the RENDERING and the STATUS, not the ordering or the
fetching — those belong to :mod:`jaato_sdk.release_channels` and are pinned
in ``test_release_channels.py``.  The two are one report drawn twice, and
this suite is about the drawing.

**No test here reaches the network**: the report is substituted.  The one
case that would is skipped unless the environment opts in, because a
diagnostic's test suite that needs PyPI to be up is a suite that goes red for
reasons its authors did not write.
"""

import pytest

from jaato_sdk import doctor, release_channels as rc
from jaato_sdk.doctor import FAIL, PASS, WARN


def _report(*, enabled=True, rows=(), errors=()):
    """A :class:`ReleaseReport` stating exactly the situation under test."""
    by_name = {c.name: c for c in rc.CHANNELS}
    report = rc.ReleaseReport(enabled=enabled, errors=list(errors))
    for name, installed, channels in rows:
        dist = rc.DistStatus(name=name, installed=installed)
        for channel_name, latest, verdict, error in channels:
            dist.channels.append(rc.ChannelStatus(
                channel=by_name[channel_name], latest=latest,
                verdict=verdict, error=error))
        report.distributions.append(dist)
    return report


@pytest.fixture
def served(monkeypatch):
    """Serve a canned report in place of anything that would use the network."""
    def install(report):
        monkeypatch.setattr(doctor._releases, "check_releases",
                            lambda **kw: report)
    return install


def _one(**kw):
    checks = doctor.check_package_releases(**kw)
    assert len(checks) == 1, "the check reports exactly one line"
    return checks[0]


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------

def test_it_never_fails_whatever_the_indexes_say(served):
    """A release is news, not a defect.

    ``jaato-doctor`` exits non-zero on any FAIL and is documented as usable
    as a CI gate, so failing because somebody shipped would break every
    harness using it the documented way.
    """
    for report in (
        _report(rows=[("jaato-sdk", "0.1.0",
                       [("pypi", "9.9.9", "update", None)])]),
        _report(rows=[("jaato-sdk", "0.1.0",
                       [("pypi", None, "unknown", "cannot reach it")])]),
        _report(enabled=False),
        _report(errors=["no jaato distributions are installed here"]),
    ):
        served(report)
        assert _one().status is not FAIL


def test_a_newer_build_warns_and_names_version_channel_and_command(served):
    """The notification has to be actionable without a second command."""
    served(_report(rows=[("jaato-sdk", "0.22.0",
                          [("pypi", "0.22.0", "current", None),
                           ("testpypi", "0.23.0rc4", "update", None)])]))
    check = _one()
    assert check.status == WARN
    assert "0.22.0" in check.detail and "0.23.0rc4" in check.detail
    assert "release candidate" in check.detail          # WHICH channel
    assert "--pre" in check.detail                      # and how to get it


def test_everything_current_passes_and_says_what_it_compared(served):
    served(_report(rows=[("jaato-sdk", "0.22.0",
                          [("pypi", "0.22.0", "current", None),
                           ("testpypi", "0.22.0", "current", None)])]))
    check = _one()
    assert check.status == PASS
    assert "jaato-sdk 0.22.0" in check.detail


def test_an_unreleased_checkout_build_is_called_ahead_not_current(served):
    """The normal state of this repository, and the one most easily mislabelled.

    Telling a contributor on an unreleased build that they are "up to date"
    is how a check stops being read.
    """
    served(_report(rows=[("jaato-sdk", "0.23.0",
                          [("pypi", "0.22.0", "ahead", None),
                           ("testpypi", "0.23.0rc4", "ahead", None)])]))
    check = _one()
    assert check.status == PASS
    assert "AHEAD" in check.detail


def test_ahead_is_pluralised_by_how_many_are_ahead(served):
    served(_report(rows=[("jaato-sdk", "0.23.0",
                          [("pypi", "0.22.0", "ahead", None)]),
                         ("jaato-server", "0.16.0",
                          [("pypi", "0.15.0", "ahead", None)])]))
    assert "are AHEAD" in _one().detail
    served(_report(rows=[("jaato-sdk", "0.23.0",
                          [("pypi", "0.22.0", "ahead", None)])]))
    assert "is AHEAD" in _one().detail


def test_an_unreachable_index_warns_and_refuses_to_claim_currency(served):
    """The wording matters as much as the status.

    "cannot check" is what ``check_mcp_sdk`` says for the same situation, and
    the detail states outright that this is not a verdict — an operator who
    reads it as one has been told the opposite of the truth.
    """
    served(_report(rows=[("jaato-sdk", "0.22.0",
                          [("pypi", None, "unknown",
                            "cannot reach https://pypi.org (timed out)")])]))
    check = _one()
    assert check.status == WARN
    assert "cannot check" in check.detail
    assert "not a verdict" in check.detail
    assert rc.ENV_SWITCH in check.detail        # and how to stop being asked


def test_an_update_on_one_channel_still_reports_the_silent_one(served):
    """A partial answer presented as a whole one is the other failure mode."""
    served(_report(rows=[("jaato-sdk", "0.22.0",
                          [("pypi", "0.23.0", "update", None),
                           ("testpypi", None, "unknown", "cannot reach it")])]))
    check = _one()
    assert check.status == WARN
    assert "0.23.0" in check.detail
    assert "not every channel answered" in check.detail


# --------------------------------------------------------------------------
# The off switch
# --------------------------------------------------------------------------

def test_disabled_by_flag_contacts_nothing_and_says_so(monkeypatch):
    def explode(**kw):                                 # pragma: no cover
        raise AssertionError("--no-release-check must prevent the request")

    monkeypatch.setattr(doctor._releases, "check_releases", explode)
    check = _one(enabled=False)
    assert check.status == PASS
    assert "--no-release-check" in check.detail
    assert rc.ENV_SWITCH in check.detail


def test_disabled_by_env_is_reported_as_disabled(served):
    served(_report(enabled=False))
    check = _one()
    assert check.status == PASS
    assert rc.ENV_SWITCH in check.detail


# --------------------------------------------------------------------------
# Wiring — a correct check nothing calls is still no notification
# --------------------------------------------------------------------------

def test_run_checks_includes_it_and_honours_the_flags(monkeypatch):
    """The knobs must reach the module, not just exist on the parser."""
    seen = {}

    def record(**kw):
        seen.update(kw)
        return _report(rows=[("jaato-sdk", "0.1.0",
                              [("pypi", "0.1.0", "current", None)])])

    monkeypatch.setattr(doctor._releases, "check_releases", record)
    checks = doctor.run_checks(
        socket_path="/nonexistent/jaato-doctor-test.sock",
        pidfile="/nonexistent/jaato-doctor-test.pid",
        workspace=".", config_root=None, env_file=None, secret=None,
        auto_start=False, release_timeout=1.5, refresh_releases=True)
    assert any(c.name == "package releases" for c in checks)
    assert seen == {"timeout": 1.5, "refresh": True}


def test_the_cli_flags_reach_run_checks(monkeypatch):
    """The last link: typed on the command line, arriving at the check.

    Asserted through ``main`` rather than by inspecting the parser, because
    a flag argparse accepts and nobody forwards is exactly the shape of a
    knob that does nothing — which is the defect class this repository
    keeps finding.
    """
    seen = {}

    def record(**kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(doctor, "run_checks", record)
    monkeypatch.setattr(doctor, "_print", lambda checks: 0)
    assert doctor.main(["--no-release-check", "--refresh-release-check",
                        "--release-check-timeout", "1.5"]) == 0
    assert seen["release_check"] is False
    assert seen["refresh_releases"] is True
    assert seen["release_timeout"] == 1.5

    seen.clear()
    assert doctor.main([]) == 0
    assert seen["release_check"] is True, "the check is on by default"
    assert seen["release_timeout"] == rc.DEFAULT_TIMEOUT


@pytest.mark.skipif(True, reason="reaches the network; run by hand with -k")
def test_live_against_the_real_indexes():          # pragma: no cover
    """Kept as a runnable probe, never as a CI dependency.

    A diagnostic's suite that needs PyPI to be up goes red for reasons
    nobody wrote, so this is skipped by construction; drop the skipif
    locally to see what the indexes actually carry.
    """
    check = _one(timeout=10)
    assert check.status in (PASS, WARN)
