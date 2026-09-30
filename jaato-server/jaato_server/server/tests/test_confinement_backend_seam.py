"""The confinement seam (SELinux backend design §3, phase 1).

What is pinned here, none of it needing a kernel:

* selection: AppArmor when available, else SELinux, else none; an explicit
  choice is honoured; an unknown value and a required-but-missing backend
  refuse;
* the SELinux backend's readiness checks run in order and name the first
  that fails, and the backend never claims availability while it cannot
  provision;
* the AppArmor adapter hands the manager exactly what its callers do today.
"""

from types import SimpleNamespace

import pytest

from jaato_server.server.confinement import Boundary, select_backend
from jaato_server.server.confinement.apparmor import AppArmorBackend
from jaato_server.server.confinement.selinux import (
    RUNNER_PROBE_CONTEXT,
    SELinuxBackend,
    policy_marker_context,
)
from jaato_server.shared.tests.reversion import Reversion

_SELECTION = "jaato-server/jaato_server/server/confinement/selection.py"
_SELINUX = "jaato-server/jaato_server/server/confinement/selinux.py"

REVERSIONS = [
    Reversion(
        target=_SELECTION,
        find="    if requested not in _CHOICES:\n",
        replace="    if False:\n",
        test="test_an_unknown_value_is_refused",
        because="a typo in JAATO_CONFINEMENT would silently change which "
                "backend (or none) confines every session",
    ),
    Reversion(
        target=_SELINUX,
        find='        """``False`` until provisioning exists, whatever the host says."""\n'
             "        return False\n",
        replace='        """``False`` until provisioning exists, whatever the host says."""\n'
                "        return self.host_readiness().ready\n",
        test="test_a_ready_host_is_still_not_available",
        because="the backend would be selected on a ready host and then "
                "provision nothing, so sessions would run unconfined while "
                "the daemon reported SELinux confinement",
    ),
]


class _Backend:
    def __init__(self, name, available, reason=None):
        self.name = name
        self._available = available
        self.unavailable_reason = None if available else reason

    def is_available(self):
        return self._available


def _select(env, apparmor=None, selinux=None):
    return select_backend(
        apparmor=lambda: apparmor, selinux=lambda: selinux, environ=env,
    )


def test_auto_prefers_apparmor():
    aa, se = _Backend("apparmor", True), _Backend("selinux", True)
    choice = _select({}, aa, se)
    assert choice.backend is aa and not choice.refuse


def test_auto_falls_back_to_selinux():
    choice = _select(
        {}, _Backend("apparmor", False, "no securityfs"), _Backend("selinux", True),
    )
    assert choice.name == "selinux"
    assert choice.reasons == {"apparmor": "no securityfs"}


def test_neither_available_says_why_and_does_not_refuse():
    choice = _select(
        {}, _Backend("apparmor", False, "a"), _Backend("selinux", False, "b"),
    )
    assert choice.backend is None and not choice.refuse
    assert "apparmor: a" in choice.describe() and "selinux: b" in choice.describe()


def test_required_and_missing_refuses():
    choice = _select(
        {"JAATO_REQUIRE_CONFINEMENT": "1"}, None, _Backend("selinux", False, "b"),
    )
    assert choice.refuse
    assert choice.reasons["apparmor"] == "not installed"


def test_explicit_choice_is_honoured_and_only_that_one_is_probed():
    probed = []

    def aa():
        probed.append("apparmor")
        return _Backend("apparmor", True)

    choice = select_backend(
        apparmor=aa, selinux=lambda: _Backend("selinux", True),
        environ={"JAATO_CONFINEMENT": "selinux"},
    )
    assert choice.name == "selinux" and probed == []


def test_none_selects_nothing():
    choice = _select({"JAATO_CONFINEMENT": "none"}, _Backend("apparmor", True))
    assert choice.backend is None and not choice.refuse


def test_an_unknown_value_is_refused():
    choice = _select({"JAATO_CONFINEMENT": "apparmr"}, _Backend("apparmor", True))
    assert choice.refuse and choice.backend is None
    assert "apparmr" in choice.describe()


def test_a_probe_that_raises_is_an_unavailable_backend():
    def boom():
        raise RuntimeError("kaboom")

    choice = select_backend(apparmor=boom, selinux=lambda: None, environ={})
    assert "kaboom" in choice.reasons["apparmor"]


# ---------------------------------------------------------------- SELinux


class _FakeKernel:
    def __init__(self, contexts=(), mls=True, own="system_u:system_r:unconfined_service_t:s0",
                 transition=True):
        self.contexts = set(contexts)
        self.mls = mls
        self.own = own
        self.transition = transition

    def context_valid(self, context):
        return context in self.contexts

    def mls_enabled(self):
        return self.mls

    def current_context(self):
        return self.own

    def may_transition(self, source, target):
        return self.transition


_POLICY = (RUNNER_PROBE_CONTEXT, policy_marker_context(1))


def _selinux(kernel=None, enforcing=True, system="Linux", mounted=True):
    return SELinuxBackend(
        kernel_factory=lambda: kernel,
        host_enforcing=lambda: enforcing,
        system=lambda: system,
        mount_present=lambda: mounted,
    )


@pytest.mark.parametrize("backend, reason", [
    (_selinux(system="Darwin"), "not running on Linux"),
    (_selinux(mounted=False), "not mounted"),
    (_selinux(kernel=None), "libselinux"),
    (_selinux(_FakeKernel(_POLICY), enforcing=None), "could not be read"),
    (_selinux(_FakeKernel()), "not loaded"),
    (_selinux(_FakeKernel([RUNNER_PROBE_CONTEXT])), "older than version 1"),
    (_selinux(_FakeKernel(_POLICY, mls=False)), "MLS/MCS"),
    (_selinux(_FakeKernel(_POLICY, own=None)), "own context"),
    (_selinux(_FakeKernel(_POLICY, transition=None)), "may not transition"),
])
def test_readiness_names_the_first_failing_check(backend, reason):
    readiness = backend.host_readiness()
    assert not readiness.ready
    assert reason in readiness.reason
    assert reason in backend.unavailable_reason


def test_a_ready_host_is_still_not_available():
    backend = _selinux(_FakeKernel(_POLICY), enforcing=False)
    readiness = backend.host_readiness()
    assert readiness.ready and readiness.enforcing is False
    assert backend.is_available() is False
    assert "cannot provision" in backend.unavailable_reason
    assert backend.provision("s1", Boundary(workspace_path="/w")) is None


# --------------------------------------------------------------- AppArmor


class _Manager:
    def __init__(self, ok=True):
        self.ok = ok
        self.calls = []

    def is_available(self):
        return True

    unavailable_reason = None

    def confinement_id_for_boundary(self, workspace_path, **kwargs):
        self.calls.append(("id", workspace_path, kwargs))
        return "ws-abc123"

    def provision_profile(self, session_id, workspace_path, **kwargs):
        self.calls.append(("provision", session_id, workspace_path, kwargs))
        return self.ok

    def get_profile_name(self, session_id):
        return "jaato-ws-ws-abc123"

    def teardown_profile_by_confinement_id(self, cid):
        self.calls.append(("teardown", cid))


def test_the_adapter_passes_what_callers_pass_today():
    manager = _Manager()
    handle = AppArmorBackend(manager).provision(
        "s1", Boundary(workspace_path="/w", config_root="/w/.jaato"),
    )
    assert handle.label == "jaato-ws-ws-abc123"
    assert handle.child_label == "jaato-ws-ws-abc123//child"
    assert handle.confinement_id == "ws-abc123"
    _, sid, ws, kwargs = manager.calls[1]
    assert (sid, ws) == ("s1", "/w")
    assert kwargs["confinement_id"] == "ws-abc123"
    assert kwargs["config_root"] == "/w/.jaato"
    # Not passed unless set, as ``private_tmp_kwargs`` does (#1381).
    assert "private_tmp_dir" not in kwargs


def test_the_adapter_forwards_private_tmp_when_set():
    manager = _Manager()
    AppArmorBackend(manager).provision(
        "s1", Boundary(workspace_path="/w", private_tmp_dir="/w/.tmp"),
    )
    assert manager.calls[1][3]["private_tmp_dir"] == "/w/.tmp"


def test_a_failed_provision_is_no_handle_and_release_tears_down():
    backend = AppArmorBackend(_Manager(ok=False))
    assert backend.provision("s1", Boundary(workspace_path="/w")) is None
    manager = _Manager()
    handle = AppArmorBackend(manager).provision("s1", Boundary(workspace_path="/w"))
    AppArmorBackend(manager).release(handle)
    assert manager.calls[-1] == ("teardown", "ws-abc123")


def test_both_backends_satisfy_the_protocol():
    from jaato_server.server.confinement import ConfinementBackend

    assert isinstance(AppArmorBackend(_Manager()), ConfinementBackend)
    assert isinstance(_selinux(), ConfinementBackend)
