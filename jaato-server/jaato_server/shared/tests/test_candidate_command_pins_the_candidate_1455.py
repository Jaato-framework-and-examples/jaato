"""The release-candidate command must not pull a pre-release of every dependency.

``jaato-doctor`` and ``jaato-scaffold explain releases`` print a command
that installs a release candidate from TestPyPI.  It used to allow
pre-releases GLOBALLY (``pip --pre``, ``uv --prerelease allow``) and name
the package unversioned, so any fresh resolve took the newest pre-release of
every dependency too: jaato-sdk's ``pydantic>=2.0,<3`` admitted
``2.14.0b2``, and that is what got installed (#1455, measured 2026-09-30).

The fix pins the candidate (``"dist==version"``).  A specifier naming a
pre-release admits pre-releases for that requirement only (PEP 440), so pip
needs no ``--pre`` and uv's ``--prerelease if-necessary-or-explicit`` admits
the pinned candidate and nothing else.  Both forms were resolved against the
real indexes before replacing the old ones; the measurement is in the comment
above ``release_channels.CHANNELS``.

Asserted on the rendered commands rather than by resolving, because a guard
that reached a package index would go red on an offline host for a reason
nobody wrote.  Lives in ``shared/tests`` because the meta-suite walks only
that directory and ``server/tests``; its subject is in the SDK.
"""

from jaato_sdk import release_channels as rc
from jaato_server.shared.tests.reversion import Reversion

REVERSIONS = [
    Reversion(
        target="jaato-sdk/jaato_sdk/release_channels.py",
        find='            install_hint=("pip install -U --index-url "',
        replace='            install_hint=("pip install -U --pre --index-url "',
        test="test_no_candidate_command_allows_prereleases_globally",
        because=("pip's global --pre admits a pre-release of every "
                 "dependency and installed pydantic 2.14.0b2"),
    ),
    Reversion(
        target="jaato-sdk/jaato_sdk/release_channels.py",
        find='                             "--prerelease if-necessary-or-explicit "',
        replace='                             "--prerelease allow "',
        test="test_no_candidate_command_allows_prereleases_globally",
        because=("uv's --prerelease allow admits a pre-release of every "
                 "dependency and installed pydantic 2.14.0b2"),
    ),
    Reversion(
        target="jaato-sdk/jaato_sdk/release_channels.py",
        find="        return f'\"{dist}=={version}\"'",
        replace="        return dist",
        test="test_every_candidate_command_pins_the_candidate",
        because=("an unpinned candidate requirement: without a global flag "
                 "it resolves the PyPI stable, and with one it resolves "
                 "every dependency's pre-release"),
    ),
]

_PACKAGES = [("jaato-sdk", "0.30.0rc1"), ("jaato-server", "1.3.0rc1")]
_GLOBAL_PRERELEASE_FLAGS = ("--pre", "--prerelease allow",
                            "--prerelease=allow")


def _candidate_commands():
    candidate = {c.name: c for c in rc.CHANNELS}["testpypi"]
    return dict(candidate.install_commands(_PACKAGES))


def test_no_candidate_command_allows_prereleases_globally():
    """Neither installer may be told to take any dependency's pre-release."""
    commands = _candidate_commands()
    assert set(commands) == {"pip", "uv"}
    for installer, command in commands.items():
        tokens = command.split()
        joined = " ".join(tokens)
        for flag in _GLOBAL_PRERELEASE_FLAGS:
            assert flag not in tokens and f" {flag} " not in f" {joined} ", (
                f"{installer} candidate command carries the global "
                f"pre-release flag {flag!r}, which installed pydantic "
                f"2.14.0b2 (#1455): {command!r}"
            )


def test_every_candidate_command_pins_the_candidate():
    """The pin is what admits the candidate once the global flag is gone."""
    for installer, command in _candidate_commands().items():
        for dist, version in _PACKAGES:
            assert f'"{dist}=={version}"' in command, (
                f"{installer} candidate command does not pin {dist}=={version}: "
                f"{command!r}"
            )
