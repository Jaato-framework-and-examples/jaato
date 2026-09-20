"""Every dependent must declare the jaato-sdk floor this repo actually ships.

WHY THIS EXISTS.  `jaato-server`, `jaato-tui` and `jaato-eval` each declared
a bare ``"jaato-sdk"`` — no constraint at all — so a resolver was free to
pair any SDK with any server, and was behaving correctly when it did.  The
failure it produces is the expensive kind: nothing fails at install, and the
mismatch appears later, at provider import, as an error naming neither
package.  Measured in #1055, a fresh `jaato-server` 0.14.0 resolved
`jaato-sdk` 0.19.0 and every session died with

    Authentication failed: Model provider 'openrouter' failed to load:
    cannot import name 'reported_cache_count' from
    'jaato_sdk.plugins.model_provider.types'

`reported_cache_count` arrived in SDK 0.21.0, and jaato-server imports it
unconditionally from **seven** provider modules — so this was never one
provider being unlucky.  The same shape recurred in this session under a
different symbol (`TRAIT_UNTRUSTED_SCHEMA`), where the registry reported it
as ``Plugin 'mcp' skipped``, which reads as a missing optional dependency
rather than a version mismatch.

**It is invisible in a working tree**, which is what makes a check the right
answer rather than vigilance: in development all four distributions are
installed editable from one checkout and always move together, so the defect
can only appear on a second machine installing from PyPI — exactly where it
is hardest to diagnose.

WHAT THIS ASSERTS.  For each dependent: its ``jaato-sdk`` requirement
declares a ``>=`` floor, and that floor is **the version of `jaato-sdk` in
this repository**.

WHY THE SHIPPED VERSION, AND NOT THE OLDEST THAT WOULD WORK.  The oldest is
the more precise answer and nothing can compute it: it is the newest SDK
symbol any of ~1700 files imports, and deriving it means mapping every such
symbol to the release that introduced it, then redoing that whenever a
provider adds an import.  #1055 proposed exactly that as a discipline
("raise it in the same commit"), and a discipline nobody checks is how the
bare ``"jaato-sdk"`` survived four releases.

So this takes the same trade #1076 took one layer up — **declare what is
tested** — and the asymmetry is what makes it safe:

* a floor that is too HIGH refuses an install that might have worked, at
  install time, with a message naming both packages and their versions;
* a floor that is too LOW accepts an install that does not work, and defers
  the failure to an ImportError inside a provider.

The first is an inconvenience an adopter can read; the second is the bug.
This check can therefore only err toward the readable side.

A corollary worth stating, because it constrains releases: the four
distributions are cut from one repository and versioned together, so the
declared floor is always a version that exists (or is published in the same
cut).  A dependent released against an unpublished SDK would be a release
process error, not something this check can catch.

WHAT THIS DOES NOT ASSERT.  No upper bound.  Nothing here stops a future SDK
from breaking an older server; that is the reverse direction, it has no
declared edge to hang a constraint on, and #1078 is where it lives — the SDK
detects the server by importing the literal string ``"shared"``, which no
constraint either package could declare would protect.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[3]

#: The distribution whose version is the floor, and the ones that must
#: declare it.  jaato-sdk is absent from the second list deliberately: it
#: declares no dependency on the server BY DESIGN (#1078), so there is no
#: edge here to constrain.
SDK_DIST = "jaato-sdk"
DEPENDENTS = ("jaato-server", "jaato-tui", "jaato-eval")

#: ``jaato-sdk>=0.23.0``, ``jaato-sdk[extra] >= 0.23.0``, ``jaato-sdk``.
_REQ_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*(?P<spec>.*)$"
)
_FLOOR_RE = re.compile(r">=\s*(?P<version>[0-9][0-9A-Za-z.\-+!]*)")


def _pyproject(dist: str) -> dict:
    path = ROOT / dist / "pyproject.toml"
    assert path.exists(), f"missing {path}"
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _normalise(name: str) -> str:
    """PEP 503 normalisation — ``jaato_sdk`` and ``jaato-sdk`` are one name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _shipped_sdk_version() -> str:
    version = _pyproject(SDK_DIST).get("project", {}).get("version")
    assert version, f"{SDK_DIST}/pyproject.toml declares no version"
    return str(version)


def _sdk_requirement(dist: str) -> Optional[str]:
    """The dependent's own ``jaato-sdk`` requirement string, if it has one.

    Only ``project.dependencies`` is read.  An optional-dependency group
    naming the SDK would be a different claim ("if you want this extra")
    and is not what an install of the package itself resolves.
    """
    for raw in _pyproject(dist).get("project", {}).get("dependencies", []):
        requirement = str(raw).split(";")[0].strip()  # drop any env marker
        match = _REQ_RE.match(requirement)
        if match and _normalise(match.group("name")) == _normalise(SDK_DIST):
            return requirement
    return None


def _declared_floor(requirement: str) -> Optional[str]:
    match = _FLOOR_RE.search(requirement)
    return match.group("version") if match else None


def _floors() -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """dist -> (its raw requirement, the floor it declares)."""
    out: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for dist in DEPENDENTS:
        requirement = _sdk_requirement(dist)
        out[dist] = (
            requirement,
            _declared_floor(requirement) if requirement else None,
        )
    return out


def test_every_dependent_declares_the_sdk_at_all():
    """A dependent that stopped importing the SDK should stop declaring it."""
    missing = [d for d, (req, _) in _floors().items() if req is None]
    assert not missing, (
        "these distributions declare no jaato-sdk dependency at all: "
        + ", ".join(missing)
        + ". If one genuinely no longer needs the SDK, drop it from "
        "DEPENDENTS here in the same change — an unchecked dependent is "
        "how the bare requirement survived four releases."
    )


@pytest.mark.parametrize("dist", DEPENDENTS)
def test_the_sdk_requirement_carries_a_floor(dist: str):
    """#1055: a bare ``"jaato-sdk"`` lets a resolver pair any two versions."""
    requirement, floor = _floors()[dist]
    assert requirement is not None, f"{dist} declares no jaato-sdk dependency"
    assert floor is not None, (
        f"{dist}/pyproject.toml declares {requirement!r} — no >= floor, so "
        f"pip may pair it with any jaato-sdk ever published. Nothing fails "
        f"at install; the failure appears at provider import, as an "
        f"ImportError naming neither package (#1055)."
    )


@pytest.mark.parametrize("dist", DEPENDENTS)
def test_the_floor_is_the_sdk_this_repo_ships(dist: str):
    """The floor is a fact about what was tested together, not a guess."""
    shipped = _shipped_sdk_version()
    requirement, floor = _floors()[dist]
    assert requirement is not None, f"{dist} declares no jaato-sdk dependency"
    assert floor == shipped, (
        f"{dist}/pyproject.toml declares {requirement!r}, but this repo "
        f"ships jaato-sdk {shipped}. The four distributions are cut from "
        f"one repository and only ever tested against each other, so the "
        f"floor is that version — raise it in the same commit that bumps "
        f"the SDK. A floor that is too high refuses an install readably; "
        f"one that is too low accepts an install that breaks later inside "
        f"a provider (#1055)."
    )


def test_the_sdk_does_not_declare_a_dependency_on_the_server():
    """The reverse edge is deliberately absent — see #1078, not a gap here.

    ``jaato.session(mode='ipc')`` is SDK-only by design, so the SDK must not
    require the server.  Asserting it keeps someone from "fixing" #1078 by
    adding the edge, which would make the server a hard dependency of every
    SDK install and still not protect the hardcoded module name.
    """
    deps = _pyproject(SDK_DIST).get("project", {}).get("dependencies", [])
    server_deps = [
        str(d)
        for d in deps
        if _normalise(str(d).split(";")[0].strip().split("[")[0]
                      .split(">")[0].split("<")[0].split("=")[0].strip())
        == "jaato-server"
    ]
    assert not server_deps, (
        "jaato-sdk now declares a dependency on jaato-server: "
        f"{server_deps}. That edge is deliberately absent — the SDK is "
        "usable without the server, and adding the edge would not fix "
        "#1078 (the SDK detects the server by importing the literal "
        "module name 'shared', which no version constraint can protect)."
    )


# ---------------------------------------------------------------------------
# Reversions -- the meta-suite
# (test_every_guard_detects_its_own_reversion) discovers this list by
# name and asserts each one makes the NAMED test fail.  A guard that
# cannot notice its own reversion is not evidence.
# ---------------------------------------------------------------------------
from shared.tests.reversion import (  # noqa: E402
    Reversion,
)

REVERSIONS = [
    Reversion(
        target="jaato-server/pyproject.toml",
        find='    "jaato-sdk>=0.23.0",  # SDK protocol: base plugin',
        replace='    "jaato-sdk",  # SDK protocol: base plugin',
        because=(
            "a bare jaato-sdk requirement is #1055 itself: the resolver "
            "may pair any SDK with any server, nothing fails at install, "
            "and the mismatch surfaces as an ImportError inside a "
            "provider naming neither package"
        ),
        test="test_the_sdk_requirement_carries_a_floor",
    ),
    Reversion(
        target="jaato-tui/pyproject.toml",
        find='"jaato-sdk>=0.23.0",  # SDK protocol: IPC client',
        replace='"jaato-sdk>=0.19.0",  # SDK protocol: IPC client',
        because=(
            "a floor left behind at an older SDK is the same defect "
            "wearing the fix as a disguise -- it looks constrained and "
            "still admits a pairing this repo has never tested"
        ),
        test="test_the_floor_is_the_sdk_this_repo_ships",
    ),
]
