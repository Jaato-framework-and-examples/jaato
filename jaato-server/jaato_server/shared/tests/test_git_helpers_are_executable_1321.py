"""git over HTTPS works under confinement: git's helper directory is executable (#1321).

``git`` does not speak HTTPS itself.  clone, fetch, pull and push against an
https remote exec ``git-remote-https`` from git's exec path --
``/usr/lib/git-core`` on Debian/Ubuntu, ``/usr/libexec/git-core`` on
Fedora/RHEL.  The template granted ``ix`` on the in-PATH directories only and
``/usr/lib/** rm`` (read and map, which cannot exec), so every https git
operation in a confined session died with::

    fatal: cannot exec 'remote-https': Permission denied

It mattered once #1228 / #1319 made a bound GitHub token reach ``gh`` and
``git``: ``gh`` worked and ``git push`` could not.

Pinned here, per profile body (the rendered profile is one string holding the
base body, ``tool_hat`` and ``child``):

* the base body, ``tool_hat`` and a NON-scoped ``//child`` grant ``ix`` on
  both git exec paths;
* a SCOPED ``//child`` (``apparmor_fragments`` declared) does not -- v18's
  fragment-only exec authority is unchanged;
* the grant is git's helper directory and nothing wider: no ``ix`` on
  ``/usr/lib/**``;
* the rendered profile still compiles.

No kernel is involved: this container has no enforcing AppArmor, so what is
checked is that the framework ASKS for the grant, not that the kernel then
honours it.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from jaato_server.server.apparmor import AppArmorManager
from jaato_server.shared.tests.reversion import Reversion

_AA = "jaato-server/jaato_server/server/apparmor.py"

REVERSIONS = [
    Reversion(
        target=_AA,
        find="""  /bin/**              ix,
  # git's own helpers (v36, #1321): git-remote-https & co. live on git's
  # exec path, not on PATH, and ``/usr/lib/**`` below is ``rm`` only.
  /usr/lib/git-core/*      ix,
  /usr/libexec/git-core/*  ix,
""",
        replace="""  /bin/**              ix,
""",
        test="test_the_base_body_can_exec_git_helpers",
        because="the base body cannot exec git-remote-https",
    ),
    Reversion(
        target=_AA,
        find="""    /bin/**              ix,
    /usr/lib/git-core/*      ix,
    /usr/libexec/git-core/*  ix,
""",
        replace="""    /bin/**              ix,
""",
        test="test_tool_hat_can_exec_git_helpers",
        because="tool_hat cannot exec git-remote-https",
    ),
    Reversion(
        target=_AA,
        find='''                "    /bin/**              ix,\\n"
                "    /usr/lib/git-core/*      ix,\\n"
                "    /usr/libexec/git-core/*  ix,"
''',
        replace='''                "    /bin/**              ix,"
''',
        test="test_an_unscoped_child_can_exec_git_helpers",
        because="cli's //child cannot exec git-remote-https",
    ),
]

_GIT_GRANTS = (
    re.compile(r"^\s*/usr/lib/git-core/\*\s+ix,\s*$", re.M),
    re.compile(r"^\s*/usr/libexec/git-core/\*\s+ix,\s*$", re.M),
)


@pytest.fixture
def manager(tmp_path):
    return AppArmorManager(
        workspace_root=str(tmp_path / "workspaces"),
        venv_path="/usr/local/venv",
        profile_dir=str(tmp_path / "profiles"),
    )


def _bodies(profile: str):
    """Split a rendered profile into (base, tool_hat, child) bodies.

    ``_render_profile`` returns one string: the base body's own rules come
    first, then ``profile tool_hat {``, then ``profile child {``.  An anchor
    that is missing fails here, saying the guard is stale rather than
    letting a later assertion pass on an empty body.
    """
    hat = profile.find("profile tool_hat {")
    child = profile.find("profile child {")
    assert 0 < hat < child, (
        "could not find the tool_hat / child sub-profiles in the rendered "
        "profile -- this guard is stale, not satisfied"
    )
    return profile[:hat], profile[hat:child], profile[child:]


def _render(manager, tmp_path, requested_fragments=None):
    return manager._render_profile(
        "s1", str(tmp_path / "workspaces" / "ws"),
        requested_fragments=requested_fragments,
    )


def _has_git_grants(body: str) -> bool:
    return all(p.search(body) for p in _GIT_GRANTS)


def test_the_base_body_can_exec_git_helpers(manager, tmp_path):
    base, _, _ = _bodies(_render(manager, tmp_path))
    assert _has_git_grants(base)


def test_tool_hat_can_exec_git_helpers(manager, tmp_path):
    _, hat, _ = _bodies(_render(manager, tmp_path))
    assert _has_git_grants(hat)


def test_an_unscoped_child_can_exec_git_helpers(manager, tmp_path):
    # requested_fragments=None: a session that never opted into per-stage
    # exec scoping -- the plain WS/IPC session cli runs in.
    _, _, child = _bodies(_render(manager, tmp_path))
    assert _has_git_grants(child)


def test_a_scoped_child_keeps_fragment_only_exec(manager, tmp_path):
    # A list, even empty, opts into v18: fragments are the SOLE exec
    # authority in //child, so no broad grant -- git's included -- appears.
    _, _, child = _bodies(_render(manager, tmp_path, requested_fragments=[]))
    assert not any(p.search(child) for p in _GIT_GRANTS)
    assert not re.search(r"^\s*/usr/bin/\*\*\s+ix,", child, re.M)


def test_the_grant_is_not_widened_to_usr_lib(manager, tmp_path):
    profile = _render(manager, tmp_path)
    assert not re.search(r"^\s*/usr/lib/\*\*\s+[a-z]*x[a-z]*,", profile, re.M), (
        "an exec grant on /usr/lib/** reaches every package-private binary "
        "on the host; only git's helper directory is meant to be executable"
    )


def test_the_rendered_profile_compiles(manager, tmp_path):
    parser = shutil.which("apparmor_parser")
    if not parser:
        pytest.skip("apparmor_parser not installed")
    for fragments in (None, []):
        prof = tmp_path / f"candidate-{fragments is None}.aa"
        prof.write_text(_render(manager, tmp_path, requested_fragments=fragments))
        res = subprocess.run([parser, "-Q", "-K", str(prof)],
                             capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
