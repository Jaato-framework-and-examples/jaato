"""Are there newer jaato packages on the indexes we publish to?

jaato ships through **two channels**, and they answer different questions:

=============  ==========================  ====================================
channel        index                       what a version there means
=============  ==========================  ====================================
``pypi``       ``https://pypi.org``        a **production release** — what a
                                           plain ``pip install -U`` gives you.
``testpypi``   ``https://test.pypi.org``   a **release candidate** — a staging
                                           build of a release that has not
                                           shipped yet.
=============  ==========================  ====================================

Nothing in the framework asked either index anything, so the only way to
learn that a release existed was to look at the project page.  This module
is the one place that asks, and :mod:`jaato_sdk.doctor` (a preflight check)
and ``jaato-scaffold explain releases`` are the two surfaces that render the
answer — they are *presentations*, not second opinions, which is the same
split the rest of the framework keeps between producing data and displaying
it.

WHY THE INDEX'S OWN "LATEST" IS NOT THE ANSWER.  Both indexes serve an
``info.version`` field and on the channel that matters it is wrong for this
question.  PyPI pins a project's "latest" to the newest **stable** version
whenever one exists — which is correct for ``pip install`` and is why the
publish workflow stages every TestPyPI build as a pre-release in the first
place.  Measured on 2026-09-18, with the repository at ``jaato-sdk`` 0.23.0:

    test.pypi.org  jaato-sdk  info.version=0.21.0   newest actually there: 0.23.0rc1

So reading ``info.version`` would have reported the release-candidate channel
as two releases *behind* the candidate it was carrying — the exact thing this
module exists to surface.  The versions are therefore computed from the
release listing, ordered by :func:`parse_version`, per channel.

THREE RULES, each attached to a way a version notifier goes wrong:

* **Positive evidence only.**  An index that could not be reached yields
  ``verdict="unknown"`` and an error string, never "you are up to date".
  Absence of evidence is not currency, and a notifier that reports silence
  as good news is worse than no notifier.
* **A version we cannot parse is never ordered.**  :func:`parse_version`
  returns ``None`` rather than guessing, the unparseable strings are carried
  on :attr:`ChannelStatus.unparseable`, and they are excluded from the
  comparison instead of being sorted as "older" — which is how a notifier
  starts hiding the release it was asked about.
* **Ahead is not current.**  A checkout of this repository normally runs a
  version *newer* than anything published, and reporting that as "up to
  date" would make the check meaningless for the people who read it most.
  It is its own verdict (``ahead``).

The module is **stdlib-only and imports nothing from jaato**, the shape
:mod:`shared.apparmor_label` and :mod:`shared.completion_nudge` already
have, so ``shared`` may read it without an import cycle and the SDK may use
it with jaato-server absent.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "CHANNELS",
    "Channel",
    "ChannelStatus",
    "DistStatus",
    "ReleaseReport",
    "check_releases",
    "installed_distributions",
    "is_prerelease",
    "newest",
    "normalize_dist_name",
    "parse_version",
    "release_check_enabled",
]


# --------------------------------------------------------------------------
# PEP 440 ordering
#
# `packaging.version` does this properly and jaato-sdk does not depend on
# `packaging` (its dependencies are python-dotenv and pydantic), so importing
# it would make this check work on the machines that happen to have pip's
# vendored copy exposed and silently do nothing on the others.  A capability
# fallback is worse still: two orderings that can disagree about which release
# is newer is precisely the bug this module would then be shipping.  So there
# is ONE implementation, here, of the subset PEP 440 specifies, and a string
# outside it is REFUSED rather than approximated.
# --------------------------------------------------------------------------

#: The canonical PEP 440 version grammar (public version + optional local
#: segment).  Index-served versions are already normalised, so the liberal
#: separators exist for a locally-installed version whose metadata is not.
_VERSION_RE = re.compile(
    r"""^\s*v?
    (?:(?P<epoch>[0-9]+)!)?
    (?P<release>[0-9]+(?:\.[0-9]+)*)
    (?P<pre>[-_.]?(?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)[-_.]?(?P<pre_n>[0-9]+)?)?
    (?P<post>(?:-(?P<post_n1>[0-9]+))
             |(?:[-_.]?(?P<post_l>post|rev|r)[-_.]?(?P<post_n2>[0-9]+)?))?
    (?P<dev>[-_.]?(?P<dev_l>dev)[-_.]?(?P<dev_n>[0-9]+)?)?
    (?:\+(?P<local>[a-z0-9]+(?:[-_.][a-z0-9]+)*))?
    \s*$""",
    re.VERBOSE | re.IGNORECASE,
)

#: PEP 440's spellings for the three pre-release phases, folded to the
#: canonical one.  ``0.24.0c1`` and ``0.24.0rc1`` are the same release, and an
#: ordering that treated them as two would order them alphabetically.
_PRE_ALIASES = {"alpha": "a", "beta": "b", "c": "rc", "pre": "rc", "preview": "rc"}


class _Boundary:
    """A sort-key element that is above or below every other element.

    PEP 440's ordering needs "no pre-release segment sorts AFTER every
    pre-release segment" (1.0 > 1.0rc1) and "a dev release of a version with
    no pre/post segment sorts BEFORE everything" (1.0.dev1 < 1.0a1).  Both are
    positions in a tuple that otherwise holds ints and strings, so they need
    an element that compares against those — which is what this is.
    """

    __slots__ = ("_high",)

    def __init__(self, high: bool):
        self._high = high

    def __repr__(self) -> str:                      # pragma: no cover - debug
        return "Infinity" if self._high else "-Infinity"

    def __hash__(self) -> int:
        return hash(("_Boundary", self._high))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Boundary) and other._high == self._high

    def __lt__(self, other: object) -> bool:
        if isinstance(other, _Boundary):
            return not self._high and other._high
        return not self._high

    def __le__(self, other: object) -> bool:
        return self == other or self < other

    def __gt__(self, other: object) -> bool:
        if isinstance(other, _Boundary):
            return self._high and not other._high
        return self._high

    def __ge__(self, other: object) -> bool:
        return self == other or self > other


_INFINITY = _Boundary(True)
_NEG_INFINITY = _Boundary(False)


def _int_or_none(value: Optional[str]) -> Optional[int]:
    return int(value) if value is not None else None


def _local_key(local: Optional[str]) -> Any:
    """Order a local version segment ('+ubuntu.1') per PEP 440.

    Numeric segments compare numerically and above alphanumeric ones, so each
    becomes ``(n, "")`` and each alphanumeric ``(-Infinity, s)``.
    """
    if local is None:
        return _NEG_INFINITY
    parts = []
    for seg in re.split(r"[-_.]", local.lower()):
        parts.append((int(seg), "") if seg.isdigit() else (_NEG_INFINITY, seg))
    return tuple(parts)


def _release_segment(raw: str) -> Tuple[int, ...]:
    """The release numbers, with insignificant trailing zeros stripped.

    PEP 440 makes ``1.0`` and ``1.0.0`` the same release, so they must produce
    the same key — otherwise the two order as different versions and a
    published ``1.0`` reads as older than the ``1.0.0`` beside it.
    """
    parts = [int(part) for part in raw.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _pre_segment(m: "re.Match") -> Optional[Tuple[str, int]]:
    """The pre-release phase and number, folded to PEP 440's canonical letter.

    ``c`` / ``pre`` / ``preview`` all mean ``rc``; leaving them apart would
    order two spellings of one release alphabetically.
    """
    if not m.group("pre_l"):
        return None
    letter = m.group("pre_l").lower()
    return (_PRE_ALIASES.get(letter, letter), int(m.group("pre_n") or 0))


def _post_segment(m: "re.Match") -> Optional[Tuple[str, int]]:
    """The post-release number, from either spelling PEP 440 allows."""
    if m.group("post_n1") is not None:
        return ("post", int(m.group("post_n1")))
    if m.group("post_l"):
        return ("post", int(m.group("post_n2") or 0))
    return None


def _dev_segment(m: "re.Match") -> Optional[Tuple[str, int]]:
    """The development-release number, if this version carries one."""
    if not m.group("dev_l"):
        return None
    return ("dev", int(m.group("dev_n") or 0))


def _pre_key(pre: Optional[Tuple[str, int]],
             post: Optional[Tuple[str, int]],
             dev: Optional[Tuple[str, int]]) -> Any:
    """Where a version with no pre-release segment sorts.

    Two boundaries rather than one, and which applies depends on the other
    segments: ``1.0.dev1`` precedes every ``1.0a1`` (it is a build of a
    version not yet begun), while a plain ``1.0`` follows every ``1.0rc``.
    """
    if pre is not None:
        return pre
    if post is None and dev is not None:
        return _NEG_INFINITY              # 1.0.dev1 precedes 1.0a1
    return _INFINITY                      # 1.0 follows every 1.0<pre>


def parse_version(text: str) -> Optional[Tuple]:
    """Return a PEP 440 sort key for *text*, or ``None`` if it is not one.

    ``None`` is the whole point of the signature: it means *this string is
    not a version I can order*, which callers must carry as an unknown rather
    than fold into "older".  Nothing here ever guesses.

    The segment helpers above own one field each, so this reads as the
    grammar's shape — epoch, release, pre, post, dev, local — rather than as
    a run of conditionals.

    Args:
        text: A version string as an index or installed metadata spells it.

    Returns:
        A tuple that orders correctly against any other key this function
        returns, or ``None`` when *text* does not match PEP 440.
    """
    if not isinstance(text, str):
        return None
    m = _VERSION_RE.match(text)
    if m is None:
        return None

    pre = _pre_segment(m)
    post = _post_segment(m)
    dev = _dev_segment(m)
    return (int(m.group("epoch") or 0),
            _release_segment(m.group("release")),
            _pre_key(pre, post, dev),
            _NEG_INFINITY if post is None else post,
            _INFINITY if dev is None else dev,
            _local_key(m.group("local")))


def is_prerelease(text: str) -> Optional[bool]:
    """Is *text* a pre-release (``rc``/``a``/``b``/``dev``)?

    Returns:
        ``True`` / ``False``, or ``None`` when *text* is not a PEP 440
        version at all — the same refusal :func:`parse_version` makes, for
        the same reason.
    """
    if not isinstance(text, str):
        return None
    m = _VERSION_RE.match(text)
    if m is None:
        return None
    return bool(m.group("pre_l") or m.group("dev_l"))


def newest(versions: Iterable[str],
           *, allow_prereleases: bool) -> Tuple[Optional[str], List[str]]:
    """The highest version in *versions*, and the ones that could not be read.

    Args:
        versions: Version strings, in any order.
        allow_prereleases: Whether a pre-release may win.  ``False`` is what
            ``pip install -U`` does and so what the production channel means;
            ``True`` is the release-candidate channel, where refusing
            pre-releases would refuse the entire point of the channel.

    Returns:
        ``(highest, unparseable)``.  ``highest`` is ``None`` when nothing
        qualified — an empty listing, or one holding only pre-releases while
        *allow_prereleases* is False.  ``unparseable`` lists every string
        this module declined to order, so a caller can say so rather than
        quietly dropping it.
    """
    best: Optional[str] = None
    best_key: Optional[Tuple] = None
    unparseable: List[str] = []
    for raw in versions:
        key = parse_version(raw)
        if key is None:
            unparseable.append(str(raw))
            continue
        if not allow_prereleases and is_prerelease(raw):
            continue
        if best_key is None or key > best_key:
            best, best_key = raw, key
    return best, sorted(unparseable)


# --------------------------------------------------------------------------
# Channels
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Channel:
    """One index jaato publishes to, and what a version on it means.

    Attributes:
        name: Stable machine token (``"pypi"`` / ``"testpypi"``) — what a
            caller keys on and what ``--json`` emits.
        label: What a release HERE is, in the words the publish workflow
            uses: a production release, or a release candidate.
        base_url: Index root; the JSON metadata endpoint is derived from it,
            so a channel cannot name a host and read another.
        allow_prereleases: Whether a pre-release may be this channel's
            answer.  See :func:`newest`.
        install_hint: The ``pip`` command that installs from this channel,
            rendered with the distribution name.  A notification that does
            not say how to act on it is half a notification.
        uv_install_hint: The same thing for ``uv``.  Carried rather than
            derived — see :meth:`install_commands`.
    """

    name: str
    label: str
    base_url: str
    allow_prereleases: bool
    install_hint: str
    uv_install_hint: str

    def metadata_url(self, dist: str) -> str:
        """The JSON metadata endpoint for *dist* on this index."""
        return f"{self.base_url.rstrip('/')}/pypi/{dist}/json"

    def install_command(self, dist: str) -> str:
        """How to install *dist* from this channel with ``pip``."""
        return self.install_hint.format(dist=dist)

    def uv_install_command(self, dist: str) -> str:
        """How to install *dist* from this channel with ``uv``."""
        return self.uv_install_hint.format(dist=dist)

    def install_commands(self, dist: str) -> Tuple[Tuple[str, str], ...]:
        """``((installer, command), ...)`` for every installer we document.

        Renderers loop over this rather than naming ``pip`` and ``uv``
        themselves, so a third installer is one field here and no edit in
        ``jaato-doctor`` or ``explain releases`` — which is what stops the
        two surfaces documenting different sets.
        """
        return (("pip", self.install_command(dist)),
                ("uv", self.uv_install_command(dist)))


#: The channels, in the order a reader should consider them: what has shipped
#: first, what is staged second.
#:
#: THE ``uv`` FORM OF THE CANDIDATE COMMAND IS NOT A FLAG RENAME, and getting
#: that wrong is silent rather than loud — the naive translation runs cleanly
#: and installs the wrong package.  Measured 2026-09-18 against the real
#: indexes, with ``jaato-sdk`` 0.22.0 on PyPI and 0.23.0rc4 on TestPyPI:
#:
#:   uv pip install -U --prerelease allow \
#:       --index-url https://test.pypi.org/simple/ \
#:       --extra-index-url https://pypi.org/simple/ jaato-sdk
#:   -> jaato-sdk==0.22.0            the PyPI STABLE, not the candidate
#:
#: Two differences produce that, and each needs its own flag:
#:
#:   * ``--pre`` is ``--prerelease allow``; uv has no ``--pre``.
#:   * uv gives ``--extra-index-url`` priority OVER ``--index-url`` (pip's
#:     precedence is the reverse) and defaults to ``--index-strategy
#:     first-index``, so the first index holding the name wins outright.
#:     ``--index-strategy unsafe-best-match`` restores pip's rule — consider
#:     every index, take the best version — and is what makes the two
#:     commands resolve the same thing.  Its name is uv's own and it is
#:     accurate: reaching across indexes for a best version is how a
#:     dependency-confusion substitution gets in, which is a property of
#:     ``--extra-index-url`` in BOTH tools rather than something uv adds.
#:
#: Verified equal: both commands resolve ``jaato-sdk==0.23.0rc4`` and the
#: same seven packages.  Spelling the flags out here rather than deriving
#: them from the pip string is deliberate — they are not a transformation of
#: it, and a helper that pretended otherwise would re-introduce exactly the
#: wrong-package failure above.
CHANNELS: Tuple[Channel, ...] = (
    Channel(name="pypi",
            label="production release",
            base_url="https://pypi.org",
            allow_prereleases=False,
            install_hint="pip install -U {dist}",
            uv_install_hint="uv pip install -U {dist}"),
    Channel(name="testpypi",
            label="release candidate",
            base_url="https://test.pypi.org",
            allow_prereleases=True,
            install_hint=("pip install -U --pre --index-url "
                          "https://test.pypi.org/simple/ "
                          "--extra-index-url https://pypi.org/simple/ {dist}"),
            uv_install_hint=("uv pip install -U --prerelease allow "
                             "--index-strategy unsafe-best-match "
                             "--index-url https://test.pypi.org/simple/ "
                             "--extra-index-url https://pypi.org/simple/ "
                             "{dist}")),
)

#: Seconds a fetched answer stays good.  An index does not publish often and
#: a diagnostic that re-asks on every invocation is a diagnostic people turn
#: off, so the network cost is paid a few times a day rather than per run.
DEFAULT_MAX_AGE = 6 * 3600

#: Per-request deadline.  Short on purpose: this check is a courtesy inside a
#: preflight, and the preflight must stay usable on a machine with no route
#: to the internet.  Raise it with the caller's own argument if a corporate
#: proxy needs longer.
DEFAULT_TIMEOUT = 3.0

#: The off switch.  Read ONLY here, so ``shared`` surfaces honour it by
#: calling this module rather than by growing a second reader that could
#: disagree — and so the var does not have to be classified in
#: ``shared/env_scope.py``, whose scan covers the daemon tree and would
#: report an SDK-only entry as stale.
ENV_SWITCH = "JAATO_RELEASE_CHECK"

_OFF = {"0", "off", "no", "false", "none", "never"}

#: Where a fetched answer is remembered, under the daemon-global jaato dir.
CACHE_FILENAME = "release_check.json"


def release_check_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Is the release check switched on?

    On by default: a notification nobody enables is a notification nobody
    gets, which is the whole complaint this module answers.  It is switched
    OFF by ``JAATO_RELEASE_CHECK=off`` (also ``0``/``no``/``false``/``none``/
    ``never``), for an air-gapped host or a policy that forbids the egress.

    A value that is set and unrecognised reads as ON, because the failure
    directions are not equal: a typo that silently disables the notifier
    reproduces the state being fixed, while a typo that leaves it on costs
    one bounded request.
    """
    raw = (env if env is not None else os.environ).get(ENV_SWITCH)
    if raw is None:
        return True
    return raw.strip().lower() not in _OFF


# --------------------------------------------------------------------------
# Which packages are ours
# --------------------------------------------------------------------------

#: What makes a distribution one of jaato's own, matched against the
#: NORMALISED name so ``jaato_premium`` and ``Jaato-Premium`` both qualify.
_JAATO_PREFIX = "jaato-"


def normalize_dist_name(name: str) -> str:
    """PEP 503-ish normalisation of a distribution name, for comparison only.

    Distributions are compared and de-duplicated by this form; they are
    REPORTED under the name their own metadata spells, which is what an
    operator types into ``pip install`` and what the index is keyed by.
    """
    return (name or "").strip().lower().replace("_", "-")


def installed_distributions() -> Dict[str, str]:
    """Every installed jaato distribution, mapped to its installed version.

    MEASURED from installed metadata rather than read off a hardcoded tuple,
    for #966's reason: a distribution shipped apart from this repository —
    ``jaato-premium`` today, whatever comes next — participates in the
    release check without anyone editing a list here, and a list that has to
    be edited is a list that will be out of date exactly when a new package
    starts shipping.

    Returns:
        ``{name: version}`` under the names metadata spells.  Empty when
        metadata cannot be read at all: this is a diagnostic, and one that
        raises is worse than one that is vague.
    """
    try:
        from importlib.metadata import distributions
        installed = list(distributions())
    except Exception:      # noqa: BLE001 — metadata is best-effort here
        return {}
    found: Dict[str, str] = {}
    for dist in installed:
        # Per distribution, not around the loop: one unreadable metadata
        # directory (a half-removed .egg-info) must cost that entry and not
        # the whole answer, or a diagnostic reports "nothing is installed"
        # about an environment full of packages.
        try:
            meta = getattr(dist, "metadata", None)
            name = (meta["Name"] or "") if meta is not None else ""
            if not normalize_dist_name(name).startswith(_JAATO_PREFIX):
                continue
            # Several metadata directories can describe one distribution
            # (a stale .egg-info beside a .dist-info); first wins, and the
            # name is kept as its own metadata spells it.
            found.setdefault(name, getattr(dist, "version", "") or "")
        except Exception:  # noqa: BLE001 — see above
            continue
    return found


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

#: What a channel had to say about one distribution.
VERDICTS = ("update", "current", "ahead", "unknown")


@dataclass
class ChannelStatus:
    """One channel's answer about one distribution.

    Attributes:
        channel: The :class:`Channel` that was asked.
        latest: The highest version that channel carries, under that
            channel's pre-release rule — or ``None`` when it carries none
            this module could order.
        verdict: One of :data:`VERDICTS`.  ``"unknown"`` covers every case
            where no comparison was possible (unreachable index, absent
            project, unparseable installed version) and is deliberately NOT
            collapsed into ``"current"``: an index that did not answer has
            not told you that you are up to date.
        error: Why the answer is unknown, in one line, or ``None``.
        unparseable: Versions the channel serves that this module declined
            to order.  Carried rather than dropped so a release named in a
            spelling we do not know is visible instead of invisible.
        from_cache: Whether this answer came from the on-disk cache rather
            than from the network on this run.
        checked_at: Unix time the answer was obtained (cached or fetched).
    """

    channel: Channel
    latest: Optional[str] = None
    verdict: str = "unknown"
    error: Optional[str] = None
    unparseable: List[str] = field(default_factory=list)
    from_cache: bool = False
    checked_at: Optional[float] = None

    @property
    def has_update(self) -> bool:
        """Is there something newer here than what is installed?"""
        return self.verdict == "update"

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe view, for ``--json`` callers."""
        return {"channel": self.channel.name,
                "label": self.channel.label,
                "index": self.channel.base_url,
                "latest": self.latest,
                "verdict": self.verdict,
                "error": self.error,
                "unparseable": list(self.unparseable),
                "from_cache": self.from_cache,
                "checked_at": self.checked_at}


@dataclass
class DistStatus:
    """One distribution's installed version and every channel's answer.

    Attributes:
        name: The distribution name as its metadata spells it.
        installed: The version installed in this environment.
        channels: One :class:`ChannelStatus` per channel, in
            :data:`CHANNELS` order.
    """

    name: str
    installed: str
    channels: List[ChannelStatus] = field(default_factory=list)

    @property
    def updates(self) -> List[ChannelStatus]:
        """The channels carrying something newer than what is installed."""
        return [c for c in self.channels if c.has_update]

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe view, for ``--json`` callers."""
        return {"name": self.name,
                "installed": self.installed,
                "channels": [c.to_dict() for c in self.channels]}


@dataclass
class ReleaseReport:
    """What every channel had to say about every installed jaato package.

    This is DATA.  ``jaato-doctor`` renders it as one preflight check line
    and ``jaato-scaffold explain releases`` renders it as a block; neither
    re-decides anything, which is the framework's standing split between
    producing a result and displaying one.

    Attributes:
        enabled: Whether the check ran at all (see
            :func:`release_check_enabled`).
        distributions: One :class:`DistStatus` per installed jaato package,
            sorted by name.
        errors: Channel-level failures, already reported per distribution
            and collected here so a caller can say "the index was
            unreachable" once instead of once per package.
    """

    enabled: bool = True
    distributions: List[DistStatus] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def updates(self) -> List[Tuple[DistStatus, ChannelStatus]]:
        """Every (distribution, channel) pair carrying a newer version."""
        return [(d, c) for d in self.distributions for c in d.updates]

    @property
    def unknown(self) -> List[Tuple[DistStatus, ChannelStatus]]:
        """Every pair whose verdict could not be reached."""
        return [(d, c) for d in self.distributions
                for c in d.channels if c.verdict == "unknown"]

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe view, for ``--json`` callers."""
        return {"enabled": self.enabled,
                "distributions": [d.to_dict() for d in self.distributions],
                "errors": list(self.errors)}


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def _available_versions(payload: Mapping[str, Any]) -> Tuple[List[str], bool]:
    """Installable versions out of a PyPI JSON payload.

    A ``releases`` entry can be an EMPTY file list — a release whose files
    were deleted — and each file can be ``yanked``.  Neither is installable,
    so neither may be reported as "a newer release is available": the reader
    would run the install command and get the version they already have.

    Returns:
        ``(versions, complete)``.  ``complete`` is False when the payload
        carried no ``releases`` mapping at all and the answer had to fall
        back to the index's own ``info.version`` — which, per this module's
        docstring, is the newest STABLE version and so cannot see a release
        candidate.  Callers surface that rather than presenting a partial
        answer as a whole one.
    """
    releases = payload.get("releases")
    if not isinstance(releases, dict):
        info = payload.get("info") or {}
        single = info.get("version")
        return ([single] if isinstance(single, str) and single else []), False
    out: List[str] = []
    for version, files in releases.items():
        if not isinstance(files, list) or not files:
            continue          # a release with no files cannot be installed
        if all(isinstance(f, dict) and f.get("yanked") for f in files):
            continue          # every file withdrawn: not installable either
        out.append(version)
    return out, True


def _fetch(url: str, timeout: float,
           opener: Optional[Callable[[str, float], bytes]]) -> Mapping[str, Any]:
    """GET *url* and decode it as JSON, or raise.

    Args:
        url: The index metadata endpoint.
        timeout: Per-request deadline in seconds.
        opener: Test seam — a callable taking ``(url, timeout)`` and
            returning the raw body.  ``None`` uses :mod:`urllib.request`,
            which honours the standard proxy environment variables and
            verifies TLS.  Nothing here disables verification.
    """
    if opener is not None:
        raw = opener(url, timeout)
    else:
        request = urllib.request.Request(
            url, headers={"Accept": "application/json",
                          "User-Agent": "jaato-release-check"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("index returned a non-object JSON body")
    return payload


class _ChannelRetired(Exception):
    """Raised in place of a request to an index already found unusable.

    A sentinel rather than an early ``return`` so the one failure path keeps
    its single shape — stale-cache fallback included, which a retired channel
    still deserves.
    """


def _is_channel_wide(exc: BaseException) -> bool:
    """Does *exc* mean the INDEX is unusable, rather than this package missing?

    The distinction bounds how long the check can take.  A 404 is about one
    package and says nothing about the next, but a refused connection or a
    timeout will repeat for every package on that index — and with several
    distributions installed and two channels each, re-learning it per package
    is the difference between one deadline and eight.  So a connection-level
    failure retires its channel for the rest of the run and a fetch that got
    an HTTP answer does not.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return False                  # the server answered; this is per-package
    return isinstance(exc, (urllib.error.URLError, TimeoutError, OSError))


def _describe_failure(exc: BaseException, channel: Channel) -> str:
    """One line naming what went wrong, without a traceback or a URL dump."""
    if isinstance(exc, _ChannelRetired):
        return str(exc)
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 404:
            return f"not published on {channel.base_url}"
        return f"{channel.base_url} answered HTTP {exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return f"cannot reach {channel.base_url} ({exc.reason})"
    if isinstance(exc, (TimeoutError, OSError)):
        return f"cannot reach {channel.base_url} ({exc})"
    return f"{channel.base_url}: {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# Cache
#
# One file, keyed by channel and distribution.  It holds only what the index
# said, never anything about the environment that asked, so it is safe to
# share between the daemon's user and a workspace.
# --------------------------------------------------------------------------

def default_cache_path(home: Optional[str] = None) -> Path:
    """Where fetched answers are remembered (``~/.jaato/release_check.json``)."""
    base = Path(home) if home else Path.home()
    return base / ".jaato" / CACHE_FILENAME


def _load_cache(path: Optional[Path]) -> Dict[str, Any]:
    """Read the cache, or return empty.

    Every failure mode is the same answer — a missing, unreadable or corrupt
    cache means "nothing is remembered", never an exception out of a
    diagnostic.
    """
    if path is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:      # noqa: BLE001 — a cache is best-effort by nature
        return {}


def _store_cache(path: Optional[Path], data: Mapping[str, Any]) -> None:
    """Write the cache, best-effort.

    An unwritable HOME (a read-only container, a daemon running as another
    user) costs the cache and must never cost the check, so every failure is
    swallowed.  The write goes through a temporary file and :func:`os.replace`
    so a concurrent reader sees either the old file or the new one.
    """
    if path is None:
        return
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        os.replace(tmp, path)
    except Exception:      # noqa: BLE001 — see docstring
        # Bound before the `try` so this cleanup cannot itself raise a
        # NameError that the outer handler would then swallow, leaving a
        # partial temp file behind with nothing having reported it.
        try:
            tmp.unlink()
        except Exception:  # noqa: BLE001
            pass


def _cache_key(channel: Channel, dist: str) -> str:
    return f"{channel.name}:{normalize_dist_name(dist)}"


# --------------------------------------------------------------------------
# The check
# --------------------------------------------------------------------------

def _verdict(installed: str, latest: Optional[str]) -> Tuple[str, Optional[str]]:
    """Compare *installed* against *latest* and say which way it went.

    Returns:
        ``(verdict, error)``.  Both versions must parse for a verdict to
        exist; an installed version this module cannot order yields
        ``"unknown"`` with the reason, never a guess in either direction.
    """
    if latest is None:
        return "unknown", "the index lists no version this check can order"
    installed_key = parse_version(installed)
    if installed_key is None:
        return "unknown", (f"installed version {installed!r} is not a PEP 440 "
                           "version, so nothing can be compared to it")
    latest_key = parse_version(latest)
    if latest_key is None:                          # pragma: no cover - newest()
        return "unknown", f"index version {latest!r} is not a PEP 440 version"
    if latest_key > installed_key:
        return "update", None
    if latest_key < installed_key:
        return "ahead", None
    return "current", None


def _cached_versions(entry: Optional[Dict[str, Any]]) -> List[str]:
    """The version strings a cache entry holds, ignoring anything malformed.

    A cache is best-effort by nature: a hand-edited or half-written file must
    read as "fewer versions remembered", never raise out of a diagnostic.
    """
    if not entry:
        return []
    return [v for v in (entry.get("versions") or []) if isinstance(v, str)]


def _entry_is_fresh(entry: Optional[Dict[str, Any]], *,
                    max_age: float, now: float, refresh: bool) -> bool:
    """May this cache entry answer without asking the index again?"""
    if entry is None or refresh:
        return False
    stamped = entry.get("fetched_at")
    if not isinstance(stamped, (int, float)):
        return False
    return (now - float(stamped)) < max_age


def _status_from_stale_cache(channel: Channel, installed: str,
                             entry: Dict[str, Any], exc: BaseException,
                             now: float) -> ChannelStatus:
    """Answer from an expired cache entry, saying that is what it is.

    Stale evidence beats none — as long as it is LABELLED stale, or a reader
    takes an old answer for a live one and the freshness the cache exists to
    manage becomes invisible.
    """
    latest, unparseable = newest(_cached_versions(entry),
                                 allow_prereleases=channel.allow_prereleases)
    verdict, why = _verdict(installed, latest)
    stamped = float(entry.get("fetched_at") or now)
    age_minutes = int((now - stamped) // 60)
    return ChannelStatus(
        channel=channel, latest=latest, verdict=verdict,
        error=(f"{_describe_failure(exc, channel)}; showing a cached answer "
               f"from {age_minutes} min ago" + (f" ({why})" if why else "")),
        unparseable=unparseable, from_cache=True, checked_at=stamped)


def _status_from_versions(channel: Channel, installed: str,
                          versions: List[str], *, complete: bool,
                          from_cache: bool,
                          fetched_at: float) -> ChannelStatus:
    """Turn one channel's version listing into a verdict about *installed*."""
    latest, unparseable = newest(versions,
                                 allow_prereleases=channel.allow_prereleases)
    verdict, error = _verdict(installed, latest)
    if not complete and error is None and channel.allow_prereleases:
        # The fallback answer is the index's newest STABLE version, which on
        # this channel is exactly the wrong one; better to say so than to
        # present it as the candidate listing.
        error = ("the index served no release listing, so this is its own "
                 "'latest' — which excludes release candidates")
    return ChannelStatus(channel=channel, latest=latest, verdict=verdict,
                         error=error, unparseable=unparseable,
                         from_cache=from_cache, checked_at=fetched_at)


def _channel_status(channel: Channel, dist: str, installed: str, *,
                    timeout: float, opener, cache: Dict[str, Any],
                    max_age: float, now: float, refresh: bool,
                    unreachable: Dict[str, str]) -> ChannelStatus:
    """Ask one channel about one distribution, through the cache.

    Args:
        unreachable: Channels already found unusable on this run, mapped to
            why.  Read before any request and written by a channel-wide
            failure, so one dead index costs one deadline rather than one per
            installed package.  A cached answer is still served from it.
    """
    key = _cache_key(channel, dist)
    entry = cache.get(key) if isinstance(cache.get(key), dict) else None

    if _entry_is_fresh(entry, max_age=max_age, now=now, refresh=refresh):
        return _status_from_versions(
            channel, installed, _cached_versions(entry),
            complete=bool(entry.get("complete", True)), from_cache=True,
            fetched_at=float(entry["fetched_at"]))

    try:
        if channel.name in unreachable:
            raise _ChannelRetired(unreachable[channel.name])
        payload = _fetch(channel.metadata_url(dist), timeout, opener)
        versions, complete = _available_versions(payload)
    except Exception as exc:          # noqa: BLE001 — every failure is a verdict
        if _is_channel_wide(exc):
            unreachable.setdefault(channel.name,
                                   _describe_failure(exc, channel))
        if entry is not None:
            return _status_from_stale_cache(channel, installed, entry, exc, now)
        return ChannelStatus(channel=channel, verdict="unknown",
                             error=_describe_failure(exc, channel),
                             checked_at=None)

    cache[key] = {"versions": versions, "complete": complete,
                  "fetched_at": now}
    return _status_from_versions(channel, installed, versions,
                                 complete=complete, from_cache=False,
                                 fetched_at=now)


def _distribution_status(name: str, installed: str, *, channels, timeout,
                         opener, cache, max_age, now, refresh,
                         unreachable) -> DistStatus:
    """Every channel's answer about one distribution, in :data:`CHANNELS` order."""
    status = DistStatus(name=name, installed=installed)
    for channel in channels:
        status.channels.append(_channel_status(
            channel, name, installed, timeout=timeout, opener=opener,
            cache=cache, max_age=max_age, now=now, refresh=refresh,
            unreachable=unreachable))
    return status


def _unknown_reasons(report: "ReleaseReport") -> List[str]:
    """Why any channel declined to answer, de-duplicated and in first-seen order.

    One unreachable index produces one reason per installed package, and a
    report repeating "cannot reach https://pypi.org" four times buries the
    one fact it carries.
    """
    reasons: List[str] = []
    for _, status in report.unknown:
        if status.error and status.error not in reasons:
            reasons.append(status.error)
    return reasons


def check_releases(distributions: Optional[Mapping[str, str]] = None, *,
                   channels: Iterable[Channel] = CHANNELS,
                   timeout: float = DEFAULT_TIMEOUT,
                   max_age: float = DEFAULT_MAX_AGE,
                   cache_path: Optional[Path] = None,
                   use_cache: bool = True,
                   refresh: bool = False,
                   env: Optional[Mapping[str, str]] = None,
                   now: Optional[float] = None,
                   opener: Optional[Callable[[str, float], bytes]] = None,
                   ) -> ReleaseReport:
    """Ask every channel whether it carries a newer build of our packages.

    Args:
        distributions: ``{name: installed_version}``.  Defaults to
            :func:`installed_distributions`.
        channels: Which channels to ask.  Defaults to :data:`CHANNELS`.
        timeout: Per-request deadline in seconds.
        max_age: How long a cached answer stays good, in seconds.
        cache_path: Override the cache file; ``None`` uses
            :func:`default_cache_path`.
        use_cache: ``False`` neither reads nor writes the cache — for tests
            and for a caller that must not touch the filesystem.
        refresh: Ignore cached answers and re-ask.  The cache is still
            written, so one forced refresh benefits later runs.
        env: Environment to read the off switch from (tests).
        now: Unix time to reason from (tests), so freshness is a stated
            instant rather than a bet on a clock.
        opener: Test seam for the HTTP fetch; see :func:`_fetch`.

    Returns:
        A :class:`ReleaseReport`.  It NEVER raises: a diagnostic that can
        fail the thing it is diagnosing is worse than a vague one, so every
        failure becomes a verdict of ``"unknown"`` carrying its reason.
    """
    if not release_check_enabled(env):
        return ReleaseReport(enabled=False)

    dists = dict(distributions if distributions is not None
                 else installed_distributions())
    report = ReleaseReport(enabled=True)
    if not dists:
        report.errors.append("no jaato distributions are installed here")
        return report

    now = time.time() if now is None else now
    path = (cache_path if cache_path is not None else default_cache_path()) \
        if use_cache else None
    cache = _load_cache(path)
    before = json.dumps(cache, sort_keys=True) if path is not None else None

    unreachable: Dict[str, str] = {}      # channel name -> why, this run only
    for name in sorted(dists, key=normalize_dist_name):
        report.distributions.append(_distribution_status(
            name, dists[name] or "", channels=channels, timeout=timeout,
            opener=opener, cache=cache, max_age=max_age, now=now,
            refresh=refresh, unreachable=unreachable))

    report.errors.extend(_unknown_reasons(report))
    if path is not None and json.dumps(cache, sort_keys=True) != before:
        _store_cache(path, cache)
    return report
