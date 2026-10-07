"""One label reading for both LSMs (SELinux backend design §3.3).

The SELinux backend is selected where AppArmor is not, and the callers that
ask "which boundary is this task in" and "is the kernel enforcing it" must
get both answers from one place.  These tests pin the SELinux half, and
that the AppArmor half is the existing parser unchanged.  No SELinux kernel
is involved: the host-enforcing and domain-permissive readers are injected.
"""

import pytest

from jaato_server.shared import lsm_label
from jaato_server.shared.lsm_label import (
    BACKEND_APPARMOR,
    BACKEND_NONE,
    BACKEND_SELINUX,
    parse_lsm_label,
    parse_selinux_context,
)
from jaato_server.shared.tests.reversion import Reversion

_LSM = "jaato-server/jaato_server/shared/lsm_label.py"

RUNNER = "system_u:system_r:jaato_runner_t:s0:c12,c345"

REVERSIONS = [
    Reversion(
        target=_LSM,
        find="    if host is True and permissive is False:\n",
        replace="    if host is True:\n",
        test="test_a_permissive_domain_is_not_enforced",
        because="a domain made permissive would read as enforced, so a "
                "session record would claim a boundary the kernel logs and "
                "allows",
    ),
    Reversion(
        target=_LSM,
        find="    if context is None or context.type not in JAATO_SELINUX_DOMAINS:\n",
        replace="    if context is None:\n",
        test="test_a_non_jaato_domain_is_not_a_boundary",
        because="every task on an SELinux host has a domain, so an "
                "unconfined_t runner would read as confined",
    ),
]


def _read(raw, host=True, permissive=False):
    return parse_lsm_label(
        raw, BACKEND_SELINUX,
        host_enforcing=lambda: host,
        domain_permissive=lambda _ctx: permissive,
    )


def test_context_keeps_the_colons_of_its_level():
    ctx = parse_selinux_context("system_u:system_r:jaato_runner_t:s0-s0:c0.c1023")
    assert ctx.type == "jaato_runner_t"
    assert ctx.level == "s0-s0:c0.c1023"


def test_context_without_a_level_and_garbage():
    assert parse_selinux_context("u:r:t").identity == "t"
    assert parse_selinux_context("unconfined") is None
    assert parse_selinux_context("") is None
    assert parse_selinux_context(None) is None


def test_an_enforced_runner_domain():
    label = _read(RUNNER + "\x00\n")
    assert label.confined and label.enforced
    assert label.identity == "jaato_runner_t:s0:c12,c345"
    assert label.mode == "enforcing"


def test_a_permissive_domain_is_not_enforced():
    label = _read(RUNNER, permissive=True)
    assert label.confined and not label.enforced
    assert label.mode == "permissive"


def test_a_permissive_host_is_not_enforced():
    label = _read(RUNNER, host=False)
    assert not label.enforced
    assert label.mode == "permissive"


@pytest.mark.parametrize("host, permissive", [(None, False), (True, None)])
def test_an_unreadable_answer_is_not_enforced(host, permissive):
    label = _read(RUNNER, host=host, permissive=permissive)
    assert label.confined and not label.enforced
    assert label.mode is None


def test_a_non_jaato_domain_is_not_a_boundary():
    label = _read("unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023")
    assert not label.confined and not label.enforced


def test_two_workspaces_differ_only_by_level():
    a = _read("system_u:system_r:jaato_runner_t:s0:c1,c2")
    b = _read("system_u:system_r:jaato_runner_t:s0:c3,c4")
    assert a.identity != b.identity


def test_apparmor_labels_go_through_the_existing_parser():
    label = parse_lsm_label("jaato-ws-a (complain)", BACKEND_APPARMOR)
    assert label.identity == "jaato-ws-a" and not label.enforced
    assert parse_lsm_label("jaato-ws-a (enforce)", BACKEND_APPARMOR).enforced


def test_the_backend_is_named_not_guessed():
    # An SELinux-shaped string read as "none" claims nothing.
    label = parse_lsm_label(RUNNER, BACKEND_NONE)
    assert not label.confined and not label.enforced


def test_sandbox_modes():
    assert lsm_label.sandbox_mode_for_selinux(permissive=False) == "selinux"
    assert lsm_label.sandbox_mode_for_selinux(permissive=True) == "selinux-permissive"
    assert lsm_label.sandbox_mode_is_kernel("selinux-permissive")
    assert not lsm_label.sandbox_mode_is_kernel_enforced("selinux-permissive")
    assert lsm_label.sandbox_mode_is_kernel_enforced("apparmor")
    assert not lsm_label.sandbox_mode_is_kernel("soft")


def test_active_lsm_prefers_the_kernels_own_list(tmp_path):
    lsm = tmp_path / "lsm"
    lsm.write_text("lockdown,capability,yama,selinux,bpf")
    assert lsm_label.active_lsm_backend(
        lsm_list_path=str(lsm), apparmor_path=str(tmp_path / "nope"),
    ) == BACKEND_SELINUX
    lsm.write_text("lockdown,capability,landlock,yama,apparmor")
    assert lsm_label.active_lsm_backend(lsm_list_path=str(lsm)) == BACKEND_APPARMOR
    lsm.write_text("lockdown,capability")
    assert lsm_label.active_lsm_backend(lsm_list_path=str(lsm)) == BACKEND_NONE
