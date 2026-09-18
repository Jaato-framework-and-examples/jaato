"""The release notifier must actually notice — and must not invent currency.

``jaato-doctor`` and ``jaato-scaffold explain releases`` tell you when a
production release is on PyPI or a release candidate is staged on TestPyPI.
Before :mod:`jaato_sdk.release_channels` nothing in the tree asked either
index, so the failure mode this guards is not "it crashed" — it is the
notifier quietly answering wrongly, which looks exactly like a notifier
working.

Four ways that happens, and each is a :data:`REVERSIONS` entry below, so the
meta-suite puts the defect back and this file has to catch it.  Broader
coverage of the module lives in
``jaato-sdk/jaato_sdk/tests/test_release_channels.py``; what is here is the
subset whose reversion must be exercised, and it lives in ``shared/tests``
because the meta-suite walks only that directory and ``server/tests``
(``test_doctor_detects_checkout_skew_823.py`` is the precedent for a guard
sitting here while its subject is in the SDK).

No test here touches the network: every fetch goes through the ``opener``
seam and every cache through ``tmp_path``.
"""

import json
import textwrap

from jaato_sdk import doctor
from jaato_sdk import release_channels as rc
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

REVERSIONS = [
    Reversion(
        target="jaato-sdk/jaato_sdk/release_channels.py",
        find='    releases = payload.get("releases")',
        replace=('    return ([(payload.get("info") or {}).get("version")], True)\n'
                 '    releases = payload.get("releases")'),
        test="test_the_candidate_channel_sees_the_candidate",
        because=("trusting the index's own `latest`, which on TestPyPI is the "
                 "newest STABLE version and so cannot see a release candidate"),
    ),
    Reversion(
        target="jaato-sdk/jaato_sdk/release_channels.py",
        find='            return ChannelStatus(channel=channel, verdict="unknown",',
        replace='            return ChannelStatus(channel=channel, verdict="current",',
        test="test_an_index_that_did_not_answer_is_not_a_clean_bill_of_health",
        because="reading an unreachable index as 'you are up to date'",
    ),
    Reversion(
        target="jaato-sdk/jaato_sdk/release_channels.py",
        find='    epoch = int(m.group("epoch") or 0)',
        replace='    return (text,)\n    epoch = int(m.group("epoch") or 0)',
        test="test_versions_are_ordered_as_versions_not_as_strings",
        because="ordering versions lexicographically, so 0.9.0 beats 0.21.0",
    ),
    # The whole call, not its first line: a reversion must leave the target
    # PARSEABLE, or the module fails to import, pytest cannot collect, and the
    # meta-suite reports BLOCKED — which is not the same evidence as a guard
    # noticing its defect.
    Reversion(
        target="jaato-sdk/jaato_sdk/doctor.py",
        find="""    checks += check_package_releases(timeout=release_timeout,
                                     refresh=refresh_releases,
                                     enabled=release_check)""",
        replace="    checks += []",
        test="test_the_preflight_actually_runs_the_check",
        because="a correct check nothing calls, which notifies nobody",
    ),
]


def _payload(*versions, info_version=None):
    """A PyPI-shaped body listing *versions*, with its own `latest` field."""
    return json.dumps({
        "info": {"version": info_version or (versions[-1] if versions else "")},
        "releases": {v: [{"filename": f"x-{v}.whl", "yanked": False}]
                     for v in versions},
    }).encode()


def _opener(by_host):
    def open_url(url, timeout):
        for host, answer in by_host.items():
            if host in url:
                if isinstance(answer, BaseException):
                    raise answer
                return answer
        raise AssertionError(f"unexpected url {url}")
    return open_url


def test_the_candidate_channel_sees_the_candidate(tmp_path):
    """TestPyPI's ``info.version`` is its newest STABLE — not the candidate.

    Measured live on 2026-09-18: with ``jaato-sdk`` 0.23.0rc4 published to
    TestPyPI, that index served ``info.version=0.21.0``.  A notifier reading
    that field reports the release-candidate channel as two releases behind
    the candidate it is carrying — silently, and about the one channel the
    feature exists for.  The payload reproduces that exact shape.
    """
    report = rc.check_releases(
        {"jaato-sdk": "0.22.0"},
        opener=_opener({"test.pypi.org": _payload("0.21.0", "0.23.0rc4",
                                                  info_version="0.21.0"),
                        "pypi.org": _payload("0.22.0")}),
        cache_path=tmp_path / "c.json", now=1000.0)
    by_channel = {c.channel.name: c for c in report.distributions[0].channels}
    assert by_channel["testpypi"].latest == "0.23.0rc4", (
        "the release-candidate channel must report the candidate, which means "
        "computing from the release listing rather than trusting the index's "
        "own 'latest'"
    )
    assert by_channel["testpypi"].verdict == "update"


def test_an_index_that_did_not_answer_is_not_a_clean_bill_of_health(
        tmp_path, monkeypatch):
    """Absence of evidence is not currency.

    A notifier that reads an unreachable index as "up to date" answers the
    question wrongly rather than declining to answer it, and is worse than no
    notifier: the reader now believes something false.  Asserted on the
    verdict AND on what the preflight line then says, because the verdict is
    only worth having if the rendering preserves it.
    """
    report = rc.check_releases(
        {"jaato-sdk": "0.22.0"},
        opener=_opener({"pypi.org": OSError("network is unreachable"),
                        "test.pypi.org": OSError("network is unreachable")}),
        cache_path=tmp_path / "c.json", now=1000.0)
    for status in report.distributions[0].channels:
        assert status.verdict == "unknown", (
            "an index that did not answer must leave the verdict UNKNOWN; "
            f"got {status.verdict!r}"
        )
    assert not report.updates

    monkeypatch.setattr(doctor._releases, "check_releases", lambda **kw: report)
    detail = doctor.check_package_releases()[0].detail
    assert "cannot check" in detail and "not a verdict" in detail, (
        "the preflight line must say it could not check, not that the build "
        f"is current; it said: {detail!r}"
    )
    assert "newest on both channels" not in detail


def test_versions_are_ordered_as_versions_not_as_strings(tmp_path):
    """A string sort puts 0.9.0 after 0.21.0 — the cheapest way to be wrong.

    Asserted end to end rather than on ``parse_version`` alone, so the
    ordering is pinned where it is USED: an index listing both must report
    the higher one as latest.
    """
    assert rc.parse_version("0.21.0") > rc.parse_version("0.9.0")
    report = rc.check_releases(
        {"jaato-sdk": "0.9.0"},
        opener=_opener({"pypi.org": _payload("0.9.0", "0.21.0", "0.10.0")}),
        channels=rc.CHANNELS[:1], cache_path=tmp_path / "c.json", now=1000.0)
    assert report.distributions[0].channels[0].latest == "0.21.0"


def test_the_preflight_actually_runs_the_check():
    """A correct check nothing calls notifies nobody.

    ``run_checks`` IS ``jaato-doctor``'s preflight, so a check missing from it
    is the defect class this repository keeps finding: a mechanism that is
    implemented, documented, advertised — and never reached.

    Asserted against the SOURCE rather than by running the preflight, because
    running it means running every other check too: a daemon probe, an
    integrations scan and a ``server`` import that resolves differently
    depending on which package's tests are executing.  A guard that goes red
    for those reasons stops being read long before it catches anything.  The
    dispatch is a literal call in one function, which is exactly what a source
    read answers well — and exactly what the reversion below removes.
    """
    import ast
    import inspect

    tree = ast.parse(textwrap.dedent(inspect.getsource(doctor.run_checks)))
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "check_package_releases" in called, (
        "jaato-doctor's preflight must call the release check; the checks it "
        "does call are: " + ", ".join(sorted(n for n in called
                                             if n.startswith("check_")))
    )
