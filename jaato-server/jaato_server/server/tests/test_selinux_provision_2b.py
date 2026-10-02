"""SELinux provisioning (selinux-backend.md §5, §6; phase 2b).

No kernel: libselinux is a fake that records every label it is asked to
set, so what is pinned is what the daemon ASKS the kernel to do.  The 2a
kernel runs (probe_policy.py) are what show the kernel then enforces it.

Pinned:

* a workspace gets one level, stable across calls and distinct from every
  other workspace's, recorded outside the workspace;
* the walk labels each entry with its type (workspace or managed, authored,
  claims, private ``/tmp``) at that level, and never follows a symlink out
  of the tree;
* the authored directories exist before the walk, so none can later be
  created with the writable type;
* a root that must not be relabelled is refused;
* a labelled workspace is not walked again, and a failed walk records no
  stamp;
* the handle names the daemon's own user and role, both domains, the level.
"""

import os
import stat
from pathlib import Path

import pytest

from jaato_server.server.confinement import Boundary
from jaato_server.server.confinement.selinux import (
    RUNNER_PROBE_CONTEXT,
    SELinuxBackend,
    policy_marker_context,
)
from jaato_server.server.confinement.selinux_levels import LevelTable
from jaato_server.shared.tests.reversion import Reversion

_LABELS = "jaato-server/jaato_server/server/confinement/selinux_labels.py"
_LEVELS = "jaato-server/jaato_server/server/confinement/selinux_levels.py"
_SELINUX = "jaato-server/jaato_server/server/confinement/selinux.py"

REVERSIONS = [
    Reversion(
        target=_LABELS,
        find="                st = entry.stat(follow_symlinks=False)\n",
        replace="                st = entry.stat()\n",
        test="test_a_symlink_out_of_the_tree_is_not_followed",
        because="a link the agent planted would carry the workspace label "
                "onto whatever it points at",
    ),
    Reversion(
        target=_LEVELS,
        find="            while level in taken:\n",
        replace="            while False:\n",
        test="test_two_workspaces_never_share_a_level",
        because="two workspaces at one level read each other's files",
    ),
    Reversion(
        target=_LABELS,
        find="            if parts[1] == CLAIMS_DIR:\n                return CLAIMS_TYPE\n",
        replace="",
        test="test_each_entry_gets_its_type",
        because="claims would take the workspace type and a child could write them",
    ),
    Reversion(
        target=_LABELS,
        find="        os.makedirs(os.path.join(dot, name), exist_ok=True)\n",
        replace="        pass\n",
        test="test_missing_authored_dirs_are_created_before_the_walk",
        because="a runner creating .jaato/agents later would give it the "
                "writable workspace type",
    ),
    Reversion(
        target=_LABELS,
        find='    forbidden = {"/", "/usr", "/etc", "/home", "/root", "/var", "/tmp", home}\n',
        replace="    forbidden = set()\n",
        test="test_a_home_directory_is_refused",
        because="the walk would relabel a whole home",
    ),
    Reversion(
        target=_SELINUX,
        find="                and kernel.link_context(plan.workspace) == want):\n            return\n",
        replace="                and kernel.link_context(plan.workspace) == want):\n            pass\n",
        test="test_a_labelled_workspace_is_not_walked_again",
        because="every session would pay a full relabel walk",
    ),
    Reversion(
        target=_SELINUX,
        find="        count = selinux_labels.apply(plan, kernel.set_file_context)\n",
        replace="        count = 0\n        table.record_stamp(plan.workspace, LabelStamp(\n"
                "            REQUIRED_POLICY_VERSION, plan.managed, plan.private_tmp_dir, 0))\n"
                "        selinux_labels.apply(plan, kernel.set_file_context)\n",
        test="test_a_failed_walk_records_no_stamp",
        because="a half-labelled tree would read as labelled and never be walked again",
    ),
]

_OWN = "unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023"
_POLICY = {RUNNER_PROBE_CONTEXT, policy_marker_context(1),
           "unconfined_u:unconfined_r:jaato_runner_t:s0"}


class _Kernel:
    """libselinux as the backend uses it, recording every label set."""

    def __init__(self, fail_on=None):
        self.labels = {}
        self.calls = 0
        self.fail_on = fail_on

    def context_valid(self, context):
        return context in _POLICY

    def mls_enabled(self):
        return True

    def current_context(self):
        return _OWN

    def file_context(self, path):
        return "system_u:object_r:bin_t:s0"

    def allowed(self, source, target, tclass, perm):
        return True

    def set_file_context(self, path, context):
        self.calls += 1
        if self.fail_on and path.endswith(self.fail_on):
            raise OSError(1, "lsetfilecon: Operation not permitted", path)
        self.labels[path] = context

    def link_context(self, path):
        return self.labels.get(path)


def _backend(kernel, tmp_path, **kw):
    return SELinuxBackend(
        kernel_factory=lambda: kernel, host_enforcing=lambda: True,
        system=lambda: "Linux", mount_present=lambda: True,
        interpreter=lambda: "/usr/bin/python3",
        levels=LevelTable(tmp_path / "state" / "selinux_levels.json"),
        domain_permissive=lambda ctx: False, **kw)


def _workspace(tmp_path, name="ws"):
    ws = tmp_path / name
    (ws / ".jaato" / "agents").mkdir(parents=True)
    (ws / ".jaato" / "agents" / "a.md").write_text("persona")
    (ws / ".jaato" / "references-claims").mkdir()
    (ws / ".jaato" / "sessions").mkdir()
    (ws / "src").mkdir()
    (ws / "src" / "m.py").write_text("x")
    (ws / ".tmp").mkdir()
    return ws


def _type(label):
    return label.split(":")[2]


def _level(label):
    return label.split(":", 3)[3]


def test_two_workspaces_never_share_a_level(tmp_path, monkeypatch):
    table = LevelTable(tmp_path / "t.json")
    from jaato_server.server.confinement import selinux_levels
    # Every seed maps to one pair until the probe moves past it.
    monkeypatch.setattr(selinux_levels, "pair_for",
                        lambda seed: (1, 2) if seed % 2 == 0 else (3, 4))
    monkeypatch.setattr(selinux_levels.hashlib, "sha256",
                        lambda b: type("H", (), {"digest": lambda self: (0).to_bytes(32, "big")})())
    a = table.level_for("/w/a")
    b = table.level_for("/w/b")
    assert a != b
    assert table.level_for("/w/a") == a


def test_the_level_table_is_private_and_outside_the_workspace(tmp_path):
    kernel = _Kernel()
    ws = _workspace(tmp_path)
    backend = _backend(kernel, tmp_path)
    backend.provision("s1", Boundary(workspace_path=str(ws)))
    path = tmp_path / "state" / "selinux_levels.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert ws not in path.parents


def test_each_entry_gets_its_type(tmp_path):
    kernel = _Kernel()
    ws = _workspace(tmp_path)
    handle = _backend(kernel, tmp_path).provision("s1", Boundary(
        workspace_path=str(ws), managed=True, private_tmp_dir=str(ws / ".tmp")))
    assert handle is not None
    got = {os.path.relpath(p, ws): _type(l) for p, l in kernel.labels.items()}
    assert got["."] == "jaato_managed_ws_t"
    assert got["src/m.py"] == "jaato_managed_ws_t"
    assert got[".jaato/agents"] == "jaato_authored_t"
    assert got[".jaato/agents/a.md"] == "jaato_authored_t"
    assert got[".jaato/references-claims"] == "jaato_claims_t"
    assert got[".jaato/sessions"] == "jaato_managed_ws_t"
    assert got[".tmp"] == "jaato_tmp_t"
    level = _level(handle.label)
    assert {_level(l) for l in kernel.labels.values()} == {level}


def test_a_user_checkout_is_not_executable(tmp_path):
    kernel = _Kernel()
    ws = _workspace(tmp_path)
    _backend(kernel, tmp_path).provision("s1", Boundary(workspace_path=str(ws)))
    assert _type(kernel.labels[str(ws / "src" / "m.py")]) == "jaato_workspace_t"


def test_a_symlink_out_of_the_tree_is_not_followed(tmp_path):
    kernel = _Kernel()
    ws = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("s")
    (ws / "link").symlink_to(outside)
    _backend(kernel, tmp_path).provision("s1", Boundary(workspace_path=str(ws)))
    assert str(ws / "link") in kernel.labels        # the link itself
    assert str(ws / "link" / "secret") not in kernel.labels
    assert str(outside / "secret") not in kernel.labels


def test_missing_authored_dirs_are_created_before_the_walk(tmp_path):
    kernel = _Kernel()
    ws = tmp_path / "bare"
    ws.mkdir()
    _backend(kernel, tmp_path).provision("s1", Boundary(workspace_path=str(ws)))
    for name in ("profiles", "instructions", "scripts", "templates"):
        assert _type(kernel.labels[str(ws / ".jaato" / name)]) == "jaato_authored_t"


def test_a_home_directory_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    kernel = _Kernel()
    assert _backend(kernel, tmp_path).provision(
        "s1", Boundary(workspace_path=str(tmp_path))) is None
    assert kernel.labels == {}


def test_a_labelled_workspace_is_not_walked_again(tmp_path):
    kernel = _Kernel()
    ws = _workspace(tmp_path)
    backend = _backend(kernel, tmp_path)
    backend.provision("s1", Boundary(workspace_path=str(ws)))
    first = kernel.calls
    backend.provision("s2", Boundary(workspace_path=str(ws)))
    assert kernel.calls == first
    # A different boundary on the same tree (now managed) walks again.
    backend.provision("s3", Boundary(workspace_path=str(ws), managed=True))
    assert kernel.calls > first


def test_a_failed_walk_records_no_stamp(tmp_path):
    ws = _workspace(tmp_path)
    failing = _Kernel(fail_on="m.py")
    backend = _backend(failing, tmp_path)
    assert backend.provision("s1", Boundary(workspace_path=str(ws))) is None
    table = LevelTable(tmp_path / "state" / "selinux_levels.json")
    assert table.stamp(os.path.realpath(ws)) is None


def test_the_handle_names_the_domains_at_the_level(tmp_path):
    kernel = _Kernel()
    ws = _workspace(tmp_path)
    backend = _backend(kernel, tmp_path)
    handle = backend.provision("s1", Boundary(workspace_path=str(ws)))
    level = _level(handle.label)
    assert handle.backend == "selinux"
    assert handle.label == f"unconfined_u:unconfined_r:jaato_runner_t:{level}"
    assert handle.child_label == f"unconfined_u:unconfined_r:jaato_child_t:{level}"
    assert handle.complain is False
    assert handle.confinement_id == backend.confinement_id_for_boundary(
        Boundary(workspace_path=str(ws)))
    assert handle.grants["level"] == level


def test_the_session_tmpdir_is_made_and_labelled_at_the_level(tmp_path):
    kernel = _Kernel()
    ws = _workspace(tmp_path)
    backend = _backend(kernel, tmp_path)
    handle = backend.provision("s1", Boundary(workspace_path=str(ws)))
    tmpdir = tmp_path / "tmp" / "jaato-x" / "s1"
    backend.prepare_session_tmpdir(handle, str(tmpdir))
    assert tmpdir.is_dir()
    level = _level(handle.label)
    for entry in (tmpdir.parent, tmpdir):
        assert kernel.labels[str(entry)] == f"system_u:object_r:jaato_tmp_t:{level}"
