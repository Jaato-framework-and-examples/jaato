"""Phase 1b of the SELinux backend design: the call sites use the seam.

The daemon provisions through ``AppArmorBackend.provision(Boundary)``, the
envelope names the backend (v8, ``SessionInitEnvelope.confinement``), and the
runner reads it through ``server/runner/lsm_confine.py``.  A refactor, so
what is pinned here is that nothing a session gets changed on the way:

* ``apparmor_fragments: []`` still means "compose none" (a scoped
  ``//child``), not "compose all" -- the phase-1a adapter had folded the
  two, which would have widened the most locked-down stage;
* the diagnostics record still knows which plugin contributed a rule (#1326);
* an IPC session in complain mode is still recorded as such (#1014);
* the envelope carries a descriptor that agrees with ``profile_name``, and
  the runner refuses one it cannot enter, BEFORE the empty-``profile_name``
  no-op could read an SELinux boundary as "unconfined" (since 2b the runner
  can enter SELinux, so the refusal is for a runner NOT in the domain; see
  ``test_selinux_runner_2b.py``).
"""

from typing import Any, List, Optional
from unittest.mock import patch

import pytest

from jaato_server.server.apparmor import PluginRules
from jaato_server.server.confinement import (
    Boundary, fragment_field, plugin_rule_fields,
)
from jaato_server.server.confinement.apparmor import (
    AppArmorBackend, child_label_for, envelope_descriptor,
)
from jaato_server.server.runner import lsm_confine
from jaato_server.server.runner.session import BootstrapError, _maybe_self_confine
from jaato_server.shared.session_envelope import (
    SESSION_ENVELOPE_VERSION, SessionInitEnvelope,
)
from jaato_server.shared.tests.reversion import Reversion

_ADAPTER = "jaato-server/jaato_server/server/confinement/apparmor.py"
_RUNNER = "jaato-server/jaato_server/server/runner/session.py"
_RUNNER_SPAWN = "jaato-server/jaato_server/server/runner_spawn.py"

REVERSIONS = [
    Reversion(
        target=_ADAPTER,
        find="            grants=_recorded_grants(profile),\n",
        replace="            grants=_recorded_grants(confinement_id),\n",
        test="test_the_handle_carries_the_recorded_grants",
        because="the record is keyed by profile name, so a lookup by the bare "
                "id returns nothing and the handle never carries the grants",
    ),
    Reversion(
        target=_ADAPTER,
        find="                None if boundary.requested_fragments is None\n"
             "                else list(boundary.requested_fragments)\n",
        replace="                list(boundary.requested_fragments or ()) or None\n"
                "                if True else None\n",
        test="test_empty_fragments_stay_empty",
        because="a stage that declared apparmor_fragments: [] would get an "
                "unscoped //child that may exec anything on PATH",
    ),
    Reversion(
        target=_ADAPTER,
        find="    if not boundary.plugin_rule_owners:\n",
        replace="    if True:\n",
        test="test_plugin_attribution_reaches_the_manager",
        because="the diagnostics panel would show every plugin rule as "
                "(unattributed) (#1326)",
    ),
    Reversion(
        target=_RUNNER,
        find="    confinement = lsm_confine.resolve(envelope)\n"
             "    backend = confinement.backend if confinement else \"\"\n",
        replace="    confinement = None\n"
                "    backend = confinement.backend if confinement else \"\"\n",
        test="test_an_selinux_boundary_is_refused_not_run_unconfined",
        because="an envelope naming an SELinux boundary carries an empty "
                "profile_name, which the runner would read as an operator "
                "opt-out and serve unconfined",
    ),
    Reversion(
        target=_RUNNER_SPAWN,
        find="        confinement=_envelope_descriptor_of(profile_name, confinement),\n",
        replace="        confinement=None,\n",
        test="test_the_envelope_builder_names_the_backend",
        because="the envelope would stop naming the backend, so a later "
                "SELinux runner could not tell which LSM to enter",
    ),
]


class _Manager:
    def __init__(self, complain: Any = False) -> None:
        self.provisioned: List[dict] = []
        self._complain = complain

    def confinement_id_for_boundary(self, workspace_path: str, **kw: Any) -> str:
        return "ws-1"

    def provision_profile(self, session_id: str, workspace_path: str, **kw: Any) -> bool:
        self.provisioned.append(kw)
        return True

    def get_profile_name(self, session_id: str) -> str:
        return "jaato-ws-ws-1"

    def profile_is_complain_mode(self, session_id: str) -> Any:
        return self._complain


# ------------------------------------------------------------ the adapter


def test_empty_fragments_stay_empty() -> None:
    manager = _Manager()
    backend = AppArmorBackend(manager)
    backend.provision("s", Boundary(workspace_path="/w",
                                    requested_fragments=fragment_field([])))
    backend.provision("s", Boundary(workspace_path="/w",
                                    requested_fragments=fragment_field(None)))
    backend.provision("s", Boundary(workspace_path="/w",
                                    requested_fragments=fragment_field(["a"])))
    assert [c["requested_fragments"] for c in manager.provisioned] == [
        [], None, ["a"],
    ]


def test_plugin_attribution_reaches_the_manager() -> None:
    manager = _Manager()
    rules = PluginRules(["r1", "r2"], {"lsp": ["r1"], "gc": ["r2"]})
    AppArmorBackend(manager).provision(
        "s", Boundary(workspace_path="/w", **plugin_rule_fields(rules)))
    passed = manager.provisioned[0]["plugin_rules"]
    assert list(passed) == ["r1", "r2"]
    assert passed.by_plugin == {"gc": ["r2"], "lsp": ["r1"]}


def test_unattributed_and_absent_rules_are_passed_as_before() -> None:
    manager = _Manager()
    AppArmorBackend(manager).provision(
        "s", Boundary(workspace_path="/w", **plugin_rule_fields(["r"])))
    AppArmorBackend(manager).provision(
        "s", Boundary(workspace_path="/w", **plugin_rule_fields(None)))
    first, second = (c["plugin_rules"] for c in manager.provisioned)
    assert first == ["r"] and not hasattr(first, "by_plugin")
    assert second is None


@pytest.mark.parametrize("answer, expected", [(True, True), (False, False)])
def test_the_handle_carries_complain_mode(answer: bool, expected: bool) -> None:
    handle = AppArmorBackend(_Manager(complain=answer)).provision(
        "s", Boundary(workspace_path="/w"))
    assert handle.complain is expected


def test_a_sub_profile_has_no_child_of_its_own() -> None:
    assert child_label_for("jaato-ws-x") == "jaato-ws-x//child"
    assert child_label_for("jaato-ws-x//sub") == "jaato-ws-x//sub"


# ------------------------------------------------------------ the envelope


def _envelope(profile_name: str = "", **kw: Any) -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="s", workspace_path="/tmp/ws", profile_name=profile_name,
        provider_name="echo", model_name="m", **kw,
    )


def test_the_envelope_round_trips_the_descriptor_at_v8() -> None:
    assert SESSION_ENVELOPE_VERSION >= 8
    env = _envelope("jaato-ws-x", confinement=envelope_descriptor("jaato-ws-x"))
    back = SessionInitEnvelope.from_dict(env.to_dict())
    assert back.confinement == {
        "backend": "apparmor", "label": "jaato-ws-x",
        "child_label": "jaato-ws-x//child",
    }
    assert envelope_descriptor("") is None


def test_the_envelope_builder_names_the_backend() -> None:
    from jaato_server.server import runner_spawn

    src = open(runner_spawn.__file__).read()
    # The builder is large and needs a whole server to run; what matters
    # is that the kwarg is written from ``profile_name`` and not dropped.
    assert "confinement=_envelope_descriptor_of(profile_name, confinement)," in src
    assert runner_spawn._envelope_descriptor_of("jaato-ws-x", None)["label"] == "jaato-ws-x"
    assert runner_spawn._envelope_descriptor_of("", None) is None
    from jaato_server.server.confinement import ConfinementHandle

    handle = ConfinementHandle(
        backend="selinux", label="u:r:jaato_runner_t:s0:c1,c2",
        confinement_id="ws-abc", child_label="u:r:jaato_child_t:s0:c1,c2")
    assert runner_spawn._envelope_descriptor_of("", handle) == {
        "backend": "selinux", "label": "u:r:jaato_runner_t:s0:c1,c2",
        "child_label": "u:r:jaato_child_t:s0:c1,c2", "confinement_id": "ws-abc"}


# ------------------------------------------------------------ the runner


def test_resolve_reads_a_legacy_envelope_as_apparmor() -> None:
    conf = lsm_confine.resolve(_envelope("jaato-ws-x"))
    assert (conf.backend, conf.label, conf.child_label) == (
        "apparmor", "jaato-ws-x", "jaato-ws-x//child")
    assert lsm_confine.resolve(_envelope("")) is None


@pytest.mark.parametrize("descriptor, words", [
    ({"backend": "selinux", "label": "system_u:system_r:jaato_runner_t:s0:c1,c2",
      "child_label": "system_u:system_r:jaato_child_t:s0:c1,c2"}, "one LSM"),
    ({"backend": "apparmor", "label": "jaato-ws-other"}, "disagrees"),
    ("apparmor", "not a mapping"),
    ({"label": "jaato-ws-x"}, "(unnamed)"),
])
def test_resolve_refuses_what_it_cannot_enter(descriptor: Any, words: str) -> None:
    with pytest.raises(BootstrapError, match=words.replace("(", r"\(").replace(")", r"\)")):
        lsm_confine.resolve(_envelope("jaato-ws-x", confinement=descriptor))


def test_an_selinux_boundary_is_refused_not_run_unconfined() -> None:
    envelope = _envelope("", confinement={
        "backend": "selinux",
        "label": "system_u:system_r:jaato_runner_t:s0:c1,c2",
        "child_label": "system_u:system_r:jaato_child_t:s0:c1,c2",
    })
    with patch("jaato_server.server.runner.bootstrap.confine_to_profile") as confine:
        with pytest.raises(BootstrapError):
            _maybe_self_confine(envelope)
    confine.assert_not_called()


def test_the_runner_enters_the_label_through_the_dispatcher() -> None:
    calls: List[Optional[str]] = []
    with patch("jaato_server.server.runner.bootstrap.confine_to_profile",
               side_effect=calls.append), \
         patch("jaato_server.server.runner.bootstrap.current_confinement") as cur:
        cur.return_value.raw = "unconfined"
        with patch("jaato_server.server.runner.session._retire_and_verify_threads"), \
             patch("jaato_server.server.runner.session._announce_unenforced_profile",
                   create=True):
            try:
                _maybe_self_confine(_envelope(
                    "jaato-ws-x", confinement=envelope_descriptor("jaato-ws-x")))
            except BootstrapError:
                pass  # a post-transition readback may object; the call is the point
    assert calls == ["jaato-ws-x"]


def test_the_handle_carries_the_recorded_grants() -> None:
    from jaato_server.server import apparmor

    manager = _Manager()
    with patch.object(apparmor, "recorded_grants",
                      side_effect=lambda key: {"key": key}):
        handle = AppArmorBackend(manager).provision("s1", Boundary(workspace_path="/w"))
    assert handle.grants == {"key": handle.label}
