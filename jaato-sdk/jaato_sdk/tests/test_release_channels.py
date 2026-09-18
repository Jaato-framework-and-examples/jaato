"""The release check: ordering, channel semantics, and what it refuses to say.

jaato ships production releases to PyPI and stages release candidates on
TestPyPI, and until :mod:`jaato_sdk.release_channels` nothing in the tree
asked either index anything.  What this pins is less "it fetches a URL" than
the four ways a version notifier gets it wrong and stops being read:

* ordering a version by string, so ``0.9.0`` beats ``0.21.0``;
* reading the index's own ``latest``, which on TestPyPI is the newest
  **stable** version and therefore cannot see the release candidate the
  channel exists to carry;
* reporting an unreachable index as "up to date";
* reporting a checkout's unreleased build as "up to date".

**No test here touches the network or the real cache.**  Every fetch goes
through the ``opener`` seam and every cache through ``tmp_path``, so the
suite is the same offline and on, and a green run says nothing about
whether PyPI happens to be up.
"""

import json

import pytest

from jaato_sdk import release_channels as rc


# --------------------------------------------------------------------------
# PEP 440 ordering
# --------------------------------------------------------------------------

def test_release_candidate_precedes_its_own_release():
    """The ordering the whole check rests on: rc1 < rc2 < final."""
    order = ["0.23.0rc1", "0.23.0rc2", "0.23.0"]
    assert sorted(order, key=rc.parse_version) == order


def test_numeric_segments_order_numerically_not_as_strings():
    """A string sort puts 0.9.0 after 0.21.0 — the cheapest way to be wrong."""
    assert rc.parse_version("0.21.0") > rc.parse_version("0.9.0")
    assert sorted(["0.9.0", "0.21.0", "0.10.0"], key=rc.parse_version) == \
        ["0.9.0", "0.10.0", "0.21.0"]


@pytest.mark.parametrize("lower,higher", [
    ("1.0.dev1", "1.0a1"),        # a dev build precedes every pre-release
    ("1.0a1", "1.0b1"),
    ("1.0b1", "1.0rc1"),
    ("1.0rc1", "1.0"),            # no pre-release segment sorts LAST
    ("1.0", "1.0.post1"),
    ("1.0.post1", "1.0.post2"),
    ("1.0", "1!0.1"),             # an epoch outranks everything
    ("0.16.0rc2", "0.16.0rc10"),  # rc numbers are numbers
])
def test_pep440_segment_ordering(lower, higher):
    assert rc.parse_version(lower) < rc.parse_version(higher)


@pytest.mark.parametrize("a,b", [
    ("1.0", "1.0.0"),             # trailing zeros are not significant
    ("1.0rc1", "1.0c1"),          # `c` is a spelling of `rc`, not a letter
    ("1.0alpha2", "1.0a2"),
])
def test_equivalent_spellings_compare_equal(a, b):
    assert rc.parse_version(a) == rc.parse_version(b)


@pytest.mark.parametrize("text", ["", "latest", "nightly-2026-09-18",
                                  "1.0.0-SNAPSHOT", "v", None, 7])
def test_a_version_it_cannot_parse_is_refused_not_guessed(text):
    """``None`` means "I cannot order this" — never "this is old".

    Folding an unreadable version into "older" is how a notifier hides the
    release it was asked about, so the refusal is the contract.
    """
    assert rc.parse_version(text) is None
    assert rc.is_prerelease(text) is None


def test_agrees_with_packaging_where_packaging_is_available():
    """A cross-check against the reference implementation, when present.

    ``packaging`` is NOT a dependency of jaato-sdk — which is why this module
    carries its own ordering rather than importing one that may be absent.
    Where it happens to be installed it is the best oracle available, so the
    agreement is asserted pairwise rather than on a single sorted list: one
    sorted list can agree by luck about elements it never compares.
    """
    packaging_version = pytest.importorskip("packaging.version")
    corpus = ["0.9.0", "0.21.0", "0.22.0", "0.23.0rc1", "0.23.0rc4", "0.23.0",
              "1.0", "1.0.0", "1.0a1", "1.0b2", "1.0rc1", "1.0.post1",
              "1.0.dev1", "1!0.1", "2.0.0.dev3", "1.0+ubuntu.1", "10.0.1"]
    for a in corpus:
        for b in corpus:
            mine = ((rc.parse_version(a) > rc.parse_version(b))
                    - (rc.parse_version(a) < rc.parse_version(b)))
            ref = ((packaging_version.Version(a) > packaging_version.Version(b))
                   - (packaging_version.Version(a) < packaging_version.Version(b)))
            assert mine == ref, f"disagreed about {a} vs {b}"


# --------------------------------------------------------------------------
# Channel semantics
# --------------------------------------------------------------------------

def test_production_channel_ignores_release_candidates():
    """`pip install -U` gives the newest STABLE build, so PyPI's answer is it."""
    pool = ["0.22.0", "0.23.0rc4"]
    assert rc.newest(pool, allow_prereleases=False)[0] == "0.22.0"


def test_candidate_channel_reports_the_candidate():
    """Refusing pre-releases here would refuse the channel's entire point."""
    pool = ["0.22.0", "0.23.0rc4"]
    assert rc.newest(pool, allow_prereleases=True)[0] == "0.23.0rc4"


def test_the_two_shipped_channels_are_the_two_the_workflows_publish_to():
    names = {c.name for c in rc.CHANNELS}
    assert names == {"pypi", "testpypi"}
    by_name = {c.name: c for c in rc.CHANNELS}
    assert by_name["pypi"].allow_prereleases is False
    assert by_name["testpypi"].allow_prereleases is True
    # The candidate channel's command must actually reach that index, and
    # must permit a pre-release, or following it installs nothing new.
    candidate = by_name["testpypi"].install_command("jaato-sdk")
    assert "--pre" in candidate and "test.pypi.org" in candidate


def test_unparseable_versions_are_reported_rather_than_dropped():
    latest, unparseable = rc.newest(["1.0", "moonshot"], allow_prereleases=True)
    assert latest == "1.0"
    assert unparseable == ["moonshot"]


def test_a_channel_holding_only_prereleases_has_no_stable_answer():
    """``None`` here is "nothing qualifies", which is not "nothing exists"."""
    assert rc.newest(["1.0rc1"], allow_prereleases=False)[0] is None


# --------------------------------------------------------------------------
# The check, driven through the opener seam
# --------------------------------------------------------------------------

def _payload(*versions, yanked=(), empty=(), info_version=None):
    """A PyPI-shaped JSON body listing *versions*."""
    releases = {v: [{"filename": f"x-{v}.whl", "yanked": v in yanked}]
                for v in versions}
    for v in empty:
        releases[v] = []
    return json.dumps({"info": {"version": info_version or (versions[-1]
                                                            if versions else "")},
                       "releases": releases}).encode()


def _opener(by_host):
    """An `opener` answering from *by_host*: a payload, or an exception."""
    def open_url(url, timeout):
        for host, answer in by_host.items():
            if host in url:
                if isinstance(answer, BaseException):
                    raise answer
                return answer
        raise AssertionError(f"unexpected url {url}")
    return open_url


def _check(dists, by_host, tmp_path, **kw):
    return rc.check_releases(dists, opener=_opener(by_host),
                             cache_path=tmp_path / "cache.json",
                             now=kw.pop("now", 1000.0), **kw)


def test_it_reports_the_candidate_the_indexes_own_latest_would_hide(tmp_path):
    """#-the-whole-point: TestPyPI's ``info.version`` is its newest STABLE.

    Measured live: with 0.23.0rc4 published, TestPyPI served
    ``info.version=0.21.0``.  Trusting that field would report the
    release-candidate channel as behind the candidate it is carrying, so the
    payload here reproduces exactly that shape.
    """
    report = _check({"jaato-sdk": "0.22.0"},
                    {"test.pypi.org": _payload("0.21.0", "0.23.0rc4",
                                               info_version="0.21.0"),
                     "pypi.org": _payload("0.22.0")},
                    tmp_path)
    by_channel = {c.channel.name: c for c in report.distributions[0].channels}
    assert by_channel["testpypi"].latest == "0.23.0rc4"
    assert by_channel["testpypi"].verdict == "update"
    assert by_channel["pypi"].verdict == "current"


def test_an_unreachable_index_is_unknown_and_never_current(tmp_path):
    """Absence of evidence is not currency.

    A notifier that reads silence from the index as "you are up to date" is
    worse than no notifier, because it answers the question wrongly instead
    of declining to answer it.
    """
    report = _check({"jaato-sdk": "0.22.0"},
                    {"pypi.org": OSError("network is unreachable"),
                     "test.pypi.org": OSError("network is unreachable")},
                    tmp_path)
    for status in report.distributions[0].channels:
        assert status.verdict == "unknown"
        assert status.latest is None
        assert "unreachable" in (status.error or "")
    assert not report.updates
    assert report.errors                      # and it says so once, not twice


def test_a_build_newer_than_both_channels_is_ahead_not_current(tmp_path):
    """The normal state of a checkout; calling it "current" makes it noise."""
    report = _check({"jaato-sdk": "0.23.0"},
                    {"pypi.org": _payload("0.22.0"),
                     "test.pypi.org": _payload("0.23.0rc4")},
                    tmp_path)
    assert {c.verdict for c in report.distributions[0].channels} == {"ahead"}
    assert not report.updates


def test_a_yanked_or_fileless_release_is_not_offered_as_an_upgrade(tmp_path):
    """Following the install command must not hand back the version you have.

    A yanked release and one whose files were deleted are both *listed* and
    neither is installable, so reporting either as "newer is available" sends
    a reader to a command that no-ops.
    """
    report = _check({"jaato-sdk": "0.22.0"},
                    {"pypi.org": _payload("0.22.0", "0.23.0", yanked=("0.23.0",)),
                     "test.pypi.org": _payload("0.22.0", "0.24.0",
                                               empty=("0.24.0",))},
                    tmp_path)
    assert [c.latest for c in report.distributions[0].channels] == \
        ["0.22.0", "0.22.0"]
    assert not report.updates


def test_a_package_the_index_never_heard_of_says_so(tmp_path):
    import urllib.error
    missing = urllib.error.HTTPError("u", 404, "Not Found", {}, None)
    report = _check({"jaato-private": "1.0.0"},
                    {"pypi.org": missing, "test.pypi.org": missing}, tmp_path)
    status = report.distributions[0].channels[0]
    assert status.verdict == "unknown"
    assert "not published" in status.error


def test_an_unorderable_installed_version_blocks_the_verdict(tmp_path):
    """Nothing can be compared to it, so nothing is claimed about it."""
    report = _check({"jaato-sdk": "some-local-build"},
                    {"pypi.org": _payload("0.22.0"),
                     "test.pypi.org": _payload("0.23.0rc4")}, tmp_path)
    for status in report.distributions[0].channels:
        assert status.verdict == "unknown"
        assert "PEP 440" in status.error


def test_a_payload_with_no_release_listing_says_the_answer_is_partial(tmp_path):
    """The fallback is the index's own 'latest', which hides candidates."""
    bare = json.dumps({"info": {"version": "0.21.0"}}).encode()
    report = _check({"jaato-sdk": "0.20.0"},
                    {"pypi.org": bare, "test.pypi.org": bare}, tmp_path)
    by_channel = {c.channel.name: c for c in report.distributions[0].channels}
    assert by_channel["testpypi"].latest == "0.21.0"
    assert "release candidates" in (by_channel["testpypi"].error or "")
    # the production channel's own 'latest' IS the right answer there
    assert by_channel["pypi"].error is None


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def test_a_fresh_cached_answer_is_reused_without_asking_again(tmp_path):
    calls = []

    def counting(url, timeout):
        calls.append(url)
        return _payload("0.22.0")

    args = dict(opener=counting, cache_path=tmp_path / "c.json",
                channels=rc.CHANNELS[:1])
    rc.check_releases({"jaato-sdk": "0.22.0"}, now=1000.0, **args)
    assert len(calls) == 1
    rc.check_releases({"jaato-sdk": "0.22.0"}, now=1000.0 + 60, **args)
    assert len(calls) == 1, "a fresh cache entry must not re-ask the index"
    rc.check_releases({"jaato-sdk": "0.22.0"},
                      now=1000.0 + rc.DEFAULT_MAX_AGE + 1, **args)
    assert len(calls) == 2, "a stale cache entry must re-ask"


def test_refresh_reasks_even_with_a_fresh_entry(tmp_path):
    calls = []

    def counting(url, timeout):
        calls.append(url)
        return _payload("0.22.0")

    args = dict(opener=counting, cache_path=tmp_path / "c.json",
                channels=rc.CHANNELS[:1])
    rc.check_releases({"jaato-sdk": "0.22.0"}, now=1000.0, **args)
    rc.check_releases({"jaato-sdk": "0.22.0"}, now=1000.0, refresh=True, **args)
    assert len(calls) == 2


def test_a_failed_fetch_falls_back_to_the_stale_answer_and_dates_it(tmp_path):
    """Stale evidence beats none — as long as it is labelled stale."""
    cache = tmp_path / "c.json"
    rc.check_releases({"jaato-sdk": "0.22.0"}, now=1000.0, cache_path=cache,
                      channels=rc.CHANNELS[:1],
                      opener=_opener({"pypi.org": _payload("0.23.0")}))
    report = rc.check_releases(
        {"jaato-sdk": "0.22.0"}, now=1000.0 + rc.DEFAULT_MAX_AGE + 600,
        cache_path=cache, channels=rc.CHANNELS[:1],
        opener=_opener({"pypi.org": OSError("down")}))
    status = report.distributions[0].channels[0]
    assert status.latest == "0.23.0"
    assert status.from_cache is True
    assert "cached answer from" in status.error


def test_an_unwritable_cache_costs_the_cache_and_not_the_check(tmp_path):
    """A read-only HOME must not turn a courtesy into a failure."""
    blocked = tmp_path / "file-not-a-dir"
    blocked.write_text("")
    report = rc.check_releases({"jaato-sdk": "0.22.0"},
                               cache_path=blocked / "nested" / "c.json",
                               channels=rc.CHANNELS[:1], now=1000.0,
                               opener=_opener({"pypi.org": _payload("0.23.0")}))
    assert report.distributions[0].channels[0].verdict == "update"


def test_a_corrupt_cache_is_ignored_rather_than_raised(tmp_path):
    cache = tmp_path / "c.json"
    cache.write_text("{not json at all")
    report = rc.check_releases({"jaato-sdk": "0.22.0"}, cache_path=cache,
                               channels=rc.CHANNELS[:1], now=1000.0,
                               opener=_opener({"pypi.org": _payload("0.23.0")}))
    assert report.distributions[0].channels[0].latest == "0.23.0"


def test_use_cache_false_touches_no_file(tmp_path):
    cache = tmp_path / "c.json"
    rc.check_releases({"jaato-sdk": "0.22.0"}, cache_path=cache,
                      use_cache=False, channels=rc.CHANNELS[:1], now=1000.0,
                      opener=_opener({"pypi.org": _payload("0.23.0")}))
    assert not cache.exists()


# --------------------------------------------------------------------------
# The off switch
# --------------------------------------------------------------------------

def test_the_check_is_on_by_default():
    """A notification nobody enables is a notification nobody gets."""
    assert rc.release_check_enabled({}) is True


@pytest.mark.parametrize("value", ["off", "0", "no", "false", "OFF", " none ",
                                   "never"])
def test_the_documented_spellings_switch_it_off(value):
    assert rc.release_check_enabled({rc.ENV_SWITCH: value}) is False


@pytest.mark.parametrize("value", ["on", "1", "yes", "", "nope"])
def test_an_unrecognised_value_leaves_it_on(value):
    """The two failure directions are not equal.

    A typo that silently disables the notifier reproduces the state this
    module exists to fix and is invisible; a typo that leaves it on costs one
    bounded request.  So only the documented spellings turn it off.
    """
    assert rc.release_check_enabled({rc.ENV_SWITCH: value}) is True


def test_disabled_contacts_nothing_and_says_it_is_disabled(tmp_path):
    def explode(url, timeout):                       # pragma: no cover
        raise AssertionError("the off switch must prevent every request")

    report = rc.check_releases({"jaato-sdk": "0.1.0"}, opener=explode,
                               env={rc.ENV_SWITCH: "off"},
                               cache_path=tmp_path / "c.json")
    assert report.enabled is False
    assert report.distributions == []


# --------------------------------------------------------------------------
# Which packages are ours
# --------------------------------------------------------------------------

def test_the_installed_set_is_measured_not_hardcoded():
    """A jaato distribution shipped apart from this repo participates.

    #966's rule, applied to the release check: a hardcoded list is a list
    that is out of date exactly when a new package starts shipping.
    """
    found = rc.installed_distributions()
    assert "jaato-sdk" in found, "this suite runs with jaato-sdk installed"
    assert all(rc.normalize_dist_name(n).startswith("jaato-") for n in found)


def test_one_broken_metadata_entry_does_not_blank_the_answer(monkeypatch):
    """A half-removed .egg-info must cost its own row, not every row."""
    import importlib.metadata

    class Broken:
        @property
        def metadata(self):
            raise OSError("unreadable METADATA")

    class Good:
        metadata = {"Name": "jaato-made-up"}
        version = "9.9.9"

    monkeypatch.setattr(importlib.metadata, "distributions",
                        lambda: [Broken(), Good()])
    assert rc.installed_distributions() == {"jaato-made-up": "9.9.9"}


def test_the_report_is_json_safe(tmp_path):
    """``--json`` callers get a plain structure, not dataclasses."""
    report = _check({"jaato-sdk": "0.22.0"},
                    {"pypi.org": _payload("0.23.0"),
                     "test.pypi.org": _payload("0.24.0rc1")}, tmp_path)
    json.dumps(report.to_dict())                     # must not raise


# --------------------------------------------------------------------------
# What an unreachable index costs
# --------------------------------------------------------------------------

def test_a_dead_index_is_asked_once_not_once_per_package(tmp_path):
    """The deadline is per request, so the bound has to be per run.

    With several distributions installed and two channels each, re-learning
    "the network is down" per package is the difference between one deadline
    and eight — inside a preflight someone runs before debugging.
    """
    attempts = []

    def refuse(url, timeout):
        attempts.append(url)
        raise OSError("connection refused")

    report = rc.check_releases(
        {"jaato-sdk": "0.1.0", "jaato-server": "0.1.0", "jaato-tui": "0.1.0"},
        opener=refuse, cache_path=tmp_path / "c.json", now=1000.0)
    assert len(attempts) == len(rc.CHANNELS), (
        "each channel should be tried once per run once it is known dead; "
        f"tried {len(attempts)} times"
    )
    # and every package still gets a verdict, with the reason
    for dist in report.distributions:
        for status in dist.channels:
            assert status.verdict == "unknown"
            assert "connection refused" in (status.error or "")


def test_a_404_does_not_retire_the_index_for_other_packages(tmp_path):
    """A missing project says nothing about the next one.

    The server answered; only a connection-level failure is evidence about
    the index itself.  Collapsing the two would let one unpublished package
    silence the check for every published one.
    """
    import urllib.error
    attempts = []

    def sometimes(url, timeout):
        attempts.append(url)
        if "jaato-private" in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return _payload("9.9.9")

    report = rc.check_releases(
        {"jaato-private": "1.0.0", "jaato-sdk": "0.1.0"},
        opener=sometimes, channels=rc.CHANNELS[:1],
        cache_path=tmp_path / "c.json", now=1000.0)
    assert len(attempts) == 2, "both packages must be asked"
    by_name = {d.name: d for d in report.distributions}
    assert by_name["jaato-private"].channels[0].verdict == "unknown"
    assert by_name["jaato-sdk"].channels[0].verdict == "update"


def test_a_retired_channel_still_serves_a_stale_cached_answer(tmp_path):
    """Skipping the request must not also discard the evidence we have."""
    cache = tmp_path / "c.json"
    rc.check_releases({"jaato-sdk": "0.1.0", "jaato-server": "0.1.0"},
                      opener=_opener({"pypi.org": _payload("0.9.0")}),
                      channels=rc.CHANNELS[:1], cache_path=cache, now=1000.0)
    report = rc.check_releases(
        {"jaato-sdk": "0.1.0", "jaato-server": "0.1.0"},
        opener=_opener({"pypi.org": OSError("connection refused")}),
        channels=rc.CHANNELS[:1], cache_path=cache,
        now=1000.0 + rc.DEFAULT_MAX_AGE + 1)
    for dist in report.distributions:
        status = dist.channels[0]
        assert status.latest == "0.9.0"
        assert status.from_cache is True
        assert "cached answer from" in (status.error or "")
