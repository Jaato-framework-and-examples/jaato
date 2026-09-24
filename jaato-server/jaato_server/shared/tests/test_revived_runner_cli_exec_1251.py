"""A confined runner's cli must be able to exec /bin/sh (#1251).

A WS session ran fine — 86 consecutive ``cli_based_tool`` commands, bare
``gh`` and shell pipelines included — then the user detached and
reattached.  The reattach cold-revived the session (unload grace elapsed,
#1106), spawning a fresh runner, and from then on EVERY cli command failed
with a raw exec denial::

    [Errno 13] Permission denied: '/bin/sh'      # any shell command
    [Errno 13] Permission denied: '/usr/bin/gh'  # a bare gh, no shell

That is ``str(PermissionError(13, …))`` from the ``cli`` plugin's
``subprocess.Popen`` — an EACCES on ``exec()``, the AppArmor exec-denial
signature.

**Root cause.**  A ``cli`` subprocess transitions into the per-session
``//child`` AppArmor sub-profile (``change_profile ->
jaato-ws-{id}//child``, composed in its ``preexec_fn``).  Template v18
stripped the broad in-PATH exec grants (``/usr/bin/** ix``,
``/usr/local/bin/** ix``, ``/bin/** ix``) from ``//child`` and made
``apparmor_fragments`` the SOLE source of exec authority there — correct
for a cascade stage scoped to (say) java/mvn, which must not improvise
``curl``.  But v18 assumed EVERY confined session declares fragments.  A
plain WS/IPC session declares none (``requested_fragments is None``), so
its ``//child`` had ZERO exec grants and every cli subprocess was
exec-denied.  It surfaced on the FIRST genuinely-confined runner of such a
session — the pre-detach runner having spawned unconfined (#1100), so the
revive was the first time confinement actually bit.

**Fix (template v34).**  ``//child`` grants the broad in-PATH execs again,
but ONLY for a session that did not opt into per-stage exec scoping
(``requested_fragments is None``).  A session that DECLARED
``apparmor_fragments`` (a list, incl. ``[]``) keeps v18's fragment-sole
authority verbatim, so the curl-fallback escape stays closed.

**This file needs no kernel** — the same limit every confinement test in
this tree runs under (``is_available`` / ``apparmor_parser`` stubbed,
because CI has no AppArmor LSM).  The property here is a string one: the
rendered ``//child`` body of a non-scoping session must contain the exec
grant its cli needs.  It would have failed the day v18 landed had this
class been provisioned confined then.
"""

from __future__ import annotations

import pytest

from jaato_server.server.apparmor import AppArmorManager
from jaato_server.shared.tests.reversion import Reversion


REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/server/apparmor.py",
        find="            broad_system_exec=(requested_fragments is None),",
        replace="            broad_system_exec=False,",
        test=(
            "TestRevivedRunnerCliExec::"
            "test_non_scoping_child_grants_broad_ix"
        ),
        because=(
            "the //child sub-profile of a non-scoping session (a plain "
            "WS/IPC session, no apparmor_fragments) loses the broad "
            "/bin/** ix grants again, so every cli subprocess is "
            "EACCES-denied on exec() of /bin/sh and cli is dead under "
            "confinement (#1251)"
        ),
    ),
]

_BROAD_IX = (
    "/usr/bin/**          ix,",
    "/usr/local/bin/**    ix,",
    "/bin/**              ix,",
)


def _child_body(rendered: str) -> str:
    """The rendered ``//child`` sub-profile body — the last sub-profile
    in the template, so slicing from its marker to the end is correct."""
    start = rendered.find("profile child")
    assert start > 0, "//child sub-profile marker missing from rendered profile"
    return rendered[start:]


class TestRevivedRunnerCliExec:
    @pytest.fixture
    def manager(self) -> AppArmorManager:
        # _render_profile is a pure string builder; it does not consult
        # is_available(), so no kernel or stubbing is needed here.
        return AppArmorManager("/workspace")

    def test_non_scoping_child_grants_broad_ix(self, manager):
        """A session that declared no apparmor_fragments
        (``requested_fragments is None``) gets the broad in-PATH execs in
        //child, so its cli subprocesses can exec /bin/sh, gh, git."""
        child = _child_body(manager._render_profile("plain_ws", "/workspace"))
        for rule in _BROAD_IX:
            assert rule in child, (
                f"non-scoping //child must grant {rule!r} (#1251); without "
                f"it every cli command is EACCES-denied on exec"
            )

    def test_scoped_child_keeps_v18_fragment_sole_authority(self, manager):
        """A session that DECLARED apparmor_fragments keeps v18's
        guarantee: no broad ix in //child, so a stage scoped to a fragment
        cannot improvise an unlisted binary (the v83 curl-fallback escape).
        Both an explicit list and the empty list are scoped."""
        for req in (["host_validator"], []):
            child = _child_body(
                manager._render_profile(
                    "scoped", "/workspace", requested_fragments=req,
                )
            )
            for rule in _BROAD_IX:
                assert rule not in child, (
                    f"scoped //child (apparmor_fragments={req!r}) must NOT "
                    f"grant {rule!r}: apparmor_fragments is the sole source "
                    f"of exec authority there (v18)"
                )

    def test_child_library_mapping_is_unconditional(self, manager):
        """``/usr/lib`` / ``/lib`` mmap (``rm``) is present in //child
        whether scoped or not — an exec'd binary must map its shared
        libraries whatever authorised the exec."""
        for req in (None, ["host_validator"], []):
            child = _child_body(
                manager._render_profile(
                    "libs", "/workspace", requested_fragments=req,
                )
            )
            for rule in ("/usr/lib/**          rm,", "/lib/**              rm,"):
                assert rule in child, (
                    f"//child must keep {rule!r} (requested_fragments={req!r})"
                )

    def test_broad_ix_grant_does_not_reopen_the_escape_vector(self, manager):
        """The #1251 grant is exec-only.  A non-scoping //child must still
        carry no ``change_profile`` rule (the dropped escape vector stays
        dropped — only its DROP comment names it)."""
        child = _child_body(manager._render_profile("plain_ws", "/workspace"))
        real_rules = [
            line.strip()
            for line in child.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert not any(r.startswith("change_profile") for r in real_rules), (
            "//child must not grant a change_profile rule; #1251 grants "
            "exec, not profile transitions"
        )
