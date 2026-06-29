"""Contract tests for the in-process facade (Shape 1, PR1 + PR1.5).

Proves the existing ``jaato_sdk`` facade (``Session.ask``) rides UNCHANGED on
``InProcessClient`` for the ex01 ``ask`` round-trip — the AGENT_OUTPUT +
TURN_COMPLETED seams (PR1) — and that the real-provider wiring is in place:
``connect`` resolves credential secret URIs + ``verify_auth``s, and
``create_session`` ``configure_tools``s the embedded session (PR1.5).

A fake embedded runtime is used so no provider / credentials / network are
needed (CI-safe). The real-provider "it actually answers" proof
(facade-over-in-process == direct-in-process == IPC-facade) lives in the
dual-mode example suite.
"""

import asyncio

from jaato.in_process import (
    InProcessClient,
    InProcessEventEmitter,
    _bundle_inline_profile,
    session,
)
from jaato_sdk.events import AgentOutputEvent, EventType, TurnCompletedEvent


class _FakeEmbedded:
    """Stand-in for ``jaato.JaatoClient`` — records the real-path lifecycle
    calls and streams two model chunks via ``send_message(on_output=...)``."""

    def __init__(self) -> None:
        self.connected_with = None
        self.verify_auth_plugin_configs = "<unset>"
        self.configure_tools_session_kwargs = "<unset>"
        self.configure_tools_permission_plugin = "<unset>"
        self.closed = False

    def connect(self, project=None, location=None, model=None) -> None:
        self.connected_with = (project, location, model)

    def verify_auth(self, allow_interactive=False, on_message=None, plugin_configs=None):
        self.verify_auth_plugin_configs = plugin_configs
        return True

    def configure_tools(
        self, registry, permission_plugin=None, ledger=None,
        session_kwargs=None, skip_model_test=False,
    ) -> None:
        self.configure_tools_session_kwargs = session_kwargs
        self.configure_tools_permission_plugin = permission_plugin

    def send_message(self, prompt, on_output=None, **_kwargs) -> str:
        if on_output is not None:
            on_output("model", "Hello ", "write")
            on_output("model", "world", "append")
        return "Hello world"

    def close_session(self) -> None:
        self.closed = True


def _factory_for(holder):
    """Build an ``embedded_factory(provider) -> client`` that stashes the
    constructed fake in ``holder`` for post-run assertions."""
    def _make(provider):
        client = _FakeEmbedded()
        holder["client"] = client
        holder["provider"] = provider
        return client
    return _make


class TestInProcessEventEmitter:
    def test_subscribe_emit_dispatch(self):
        emitter = InProcessEventEmitter()
        seen = []
        emitter.subscribe(EventType.AGENT_OUTPUT, seen.append)
        emitter.emit(AgentOutputEvent(source="model", text="x"))
        emitter.emit(TurnCompletedEvent())  # different type — not delivered
        assert len(seen) == 1
        assert seen[0].text == "x"

    def test_unsubscribe_stops_delivery(self):
        emitter = InProcessEventEmitter()
        seen = []
        unsub = emitter.subscribe(EventType.AGENT_OUTPUT, seen.append)
        unsub()
        emitter.emit(AgentOutputEvent(source="model", text="x"))
        assert seen == []

    def test_subscribe_once_fires_exactly_once(self):
        emitter = InProcessEventEmitter()
        seen = []
        emitter.subscribe_once(EventType.TURN_COMPLETED, seen.append)
        emitter.emit(TurnCompletedEvent())
        emitter.emit(TurnCompletedEvent())
        assert len(seen) == 1


class TestFacadeRidesOnInProcessClient:
    def test_ask_round_trip(self):
        """The facade ``ask`` collects the streamed AGENT_OUTPUT and returns
        on TURN_COMPLETED — the unchanged facade driving the embedded client."""

        async def _run():
            holder = {}
            counters = {"output": 0, "turn": 0}
            async with InProcessClient.session(
                model="fake-model",
                provider="fakeprov",
                embedded_factory=_factory_for(holder),
            ) as s:
                s.client.subscribe(
                    EventType.AGENT_OUTPUT,
                    lambda ev: counters.__setitem__("output", counters["output"] + 1),
                )
                s.client.subscribe_once(
                    EventType.TURN_COMPLETED,
                    lambda ev: counters.__setitem__("turn", counters["turn"] + 1),
                )
                answer = await s.ask("Who are you?")

            assert answer == "Hello world"
            assert counters["output"] == 2  # two model chunks
            assert counters["turn"] == 1    # one terminal
            assert holder["provider"] == "fakeprov"  # provider threaded through

        asyncio.run(_run())

    def test_source_filter_excludes_non_model_output(self):
        """``ask`` defaults to ``sources=("model",)`` — non-model chunks are
        not collected into the answer."""

        class _MixedEmbedded(_FakeEmbedded):
            def send_message(self, prompt, on_output=None, **_kwargs):
                if on_output is not None:
                    on_output("system", "[setup]", "write")  # filtered out
                    on_output("model", "answer", "write")
                return "answer"

        async def _run():
            async with InProcessClient.session(
                model="fake-model",
                embedded_factory=lambda provider: _MixedEmbedded(),
            ) as s:
                answer = await s.ask("hi")
            assert answer == "answer"

        asyncio.run(_run())


class TestRealPathWiring:
    """PR1.5: connect resolves creds + verify_auths; create_session
    configure_tools the embedded session with the resolved plugin_configs."""

    def test_verify_auth_and_configure_tools_wired(self):
        async def _run():
            holder = {}
            plugin_configs = {"openrouter": {"api_key": "sk-or-plain"}}
            async with InProcessClient.session(
                model="m",
                provider="openrouter",
                plugin_configs=plugin_configs,
                embedded_factory=_factory_for(holder),
            ) as s:
                await s.ask("hi")

            fake = holder["client"]
            # verify_auth was called with the (resolved) plugin_configs.
            assert fake.verify_auth_plugin_configs == {
                "openrouter": {"api_key": "sk-or-plain"}
            }
            # configure_tools received the resolved plugin_configs in session_kwargs.
            assert fake.configure_tools_session_kwargs == {
                "plugin_configs": {"openrouter": {"api_key": "sk-or-plain"}},
                "plugins": [],
                "suppress_base_instructions": False,
            }
            # A plain key passes through resolve_secret_uri unchanged; pass://
            # resolution requires jaato-premium and is exercised by the example
            # suite, not CI.

        asyncio.run(_run())

    def test_failed_auth_raises(self):
        async def _run():
            class _NoAuth(_FakeEmbedded):
                def verify_auth(self, allow_interactive=False, on_message=None,
                                plugin_configs=None):
                    return False

            raised = False
            try:
                async with InProcessClient.session(
                    model="m", provider="p",
                    embedded_factory=lambda provider: _NoAuth(),
                ):
                    pass
            except RuntimeError:
                raised = True
            assert raised

        asyncio.run(_run())


class TestTransportAgnosticEntry:
    """The ``jaato.session(mode=...)`` entry + inline-profile parity, so one
    example runs both modes with the same kwargs (Daniel's target shape)."""

    def test_bundle_inline_profile_from_separate_kwargs(self):
        bundled = _bundle_inline_profile(
            {"model": "m", "provider": "p", "plugins": [],
             "plugin_configs": {"p": {"api_key": "k"}}, "agent": "pirate"}
        )
        # spec keys bundled into a `profile` dict; non-spec kwargs pass through.
        assert bundled["profile"] == {
            "model": "m", "provider": "p", "plugins": [],
            "plugin_configs": {"p": {"api_key": "k"}},
        }
        assert bundled["agent"] == "pirate"
        assert "model" not in bundled  # consumed into profile

    def test_bundle_keeps_explicit_profile(self):
        bundled = _bundle_inline_profile({"profile": {"model": "x"}, "agent": "a"})
        assert bundled == {"profile": {"model": "x"}, "agent": "a"}

    def test_unknown_mode_raises(self):
        try:
            session(mode="bogus")
        except ValueError as e:
            assert "bogus" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError for unknown mode")

    def test_session_in_process_routes_and_answers(self):
        """``jaato.session(mode='in_process', ...)`` routes to InProcessClient,
        bundles the separate kwargs into the inline profile, and answers."""

        async def _run():
            holder = {}
            async with session(
                mode="in_process",
                model="m",
                provider="openrouter",
                plugins=[],
                plugin_configs={"openrouter": {"api_key": "sk-or-plain"}},
                embedded_factory=_factory_for(holder),
            ) as s:
                answer = await s.ask("hi")

            assert answer == "Hello world"
            # provider + plugin_configs survived the bundle -> expand round-trip.
            assert holder["provider"] == "openrouter"
            assert holder["client"].configure_tools_session_kwargs == {
                "plugin_configs": {"openrouter": {"api_key": "sk-or-plain"}},
                "plugins": [],
                "suppress_base_instructions": False,
            }

        asyncio.run(_run())

    def test_inline_profile_dict_expands_in_in_process_client(self):
        """``InProcessClient.session(profile={...})`` expands the inline spec
        into model/provider/plugin_configs (parity with IPCClient's spec)."""

        async def _run():
            holder = {}
            async with InProcessClient.session(
                profile={
                    "model": "m", "provider": "openrouter", "plugins": [],
                    "plugin_configs": {"openrouter": {"api_key": "sk-or-plain"}},
                },
                embedded_factory=_factory_for(holder),
            ) as s:
                answer = await s.ask("hi")

            assert answer == "Hello world"
            assert holder["provider"] == "openrouter"

        asyncio.run(_run())


class TestEnvFileBothModes:
    """``env_file`` is a both-modes kwarg (only ``socket_path`` is IPC-only):
    the in-process client loads the ``.env`` before the embedded runtime reads
    the process env."""

    def test_env_file_loaded_into_process_env(self, tmp_path):
        import os

        env_path = tmp_path / "in_process.env"
        env_path.write_text("JAATO_INPROC_TEST_VAR=loaded123\n")
        os.environ.pop("JAATO_INPROC_TEST_VAR", None)

        async def _run():
            async with InProcessClient.session(
                model="m", env_file=str(env_path),
                embedded_factory=lambda provider: _FakeEmbedded(),
            ) as s:
                await s.ask("hi")

        try:
            asyncio.run(_run())
            assert os.environ.get("JAATO_INPROC_TEST_VAR") == "loaded123"
        finally:
            os.environ.pop("JAATO_INPROC_TEST_VAR", None)

    def test_missing_env_file_is_graceful(self, tmp_path):
        async def _run():
            async with InProcessClient.session(
                model="m", env_file=str(tmp_path / "does_not_exist.env"),
                embedded_factory=lambda provider: _FakeEmbedded(),
            ) as s:
                return await s.ask("hi")

        assert asyncio.run(_run()) == "Hello world"

    def test_env_file_none_skips_load(self):
        async def _run():
            async with InProcessClient.session(
                model="m", env_file=None,
                embedded_factory=lambda provider: _FakeEmbedded(),
            ) as s:
                return await s.ask("hi")

        assert asyncio.run(_run()) == "Hello world"


class TestPluginLoading:
    """_build_registry replicates the daemon's per-session registry setup for
    PLUGIN PARITY: discover(tier='runner') + expose_all with NO requested-plugin
    filter, so the FULL runner-tier set initializes (same as a daemon session),
    regardless of the model-gate ``plugins`` list. Model tool-visibility is gated
    separately at create_session(plugins=...). These exercise the real build
    (~seconds); they use a tmp workspace to stay hermetic (no cwd config reads)."""

    def test_build_registry_inits_full_runner_tier_and_wires_permission(self, tmp_path):
        import asyncio as _asyncio

        from jaato._in_process_permission import InProcessChannel

        # plugins=["todo"] is only the MODEL gate; _build_registry still inits
        # the whole runner-tier set (plugin parity), not just todo.
        client = InProcessClient(
            model="m", plugins=["todo"], workspace_path=str(tmp_path)
        )
        client._loop = _asyncio.new_event_loop()
        try:
            registry = client._build_registry()
            exposed = registry.list_exposed()
            # Full runner-tier initialized (NOT just the model-gate plugin).
            for name in ("cli", "todo", "permission", "introspection"):
                assert name in exposed, name
            # Permission channel wired + returned (for create_session gating).
            returned = client._wire_permission_channel(registry)
            perm = registry.get_plugin("permission")
            assert perm is not None
            assert isinstance(perm._channel, InProcessChannel)
            assert returned is perm
        finally:
            client._loop.close()

    def test_permission_plugin_passed_to_configure_tools(self, tmp_path):
        """create_session must PASS the loaded permission plugin to
        configure_tools so the session gates — the policy + channel are inert
        otherwise (the ex07 un-gated bug). Uses registry_factory to inject a
        permission-bearing registry without the full runner-tier init."""

        from shared import PluginRegistry

        def _reg():
            r = PluginRegistry(model_name="m")
            r.discover()
            r.set_workspace_path(str(tmp_path))
            r.set_config_root(str(tmp_path))
            r.expose_all(
                {"permission": {"policy": {"defaultPolicy": "ask"}}},
                requested_plugins=[],
            )
            return r

        async def _run():
            holder = {}
            async with InProcessClient.session(
                model="m",
                embedded_factory=_factory_for(holder),
                registry_factory=_reg,
            ) as s:
                await s.ask("hi")
            perm = holder["client"].configure_tools_permission_plugin
            assert perm is not None  # the session gates with it (was None = bug)

        asyncio.run(_run())


class TestProfiles:
    """PR3: profile resolution — inline dict + named (disk) — threads
    system_instructions (ex03 persona) and completion_payload_schema (ex04
    byte-exact) to configure_tools, the InProcessClient daemon-replication of
    the daemon's profile setup."""

    def test_inline_profile_threads_instructions_and_schema(self):
        async def _run():
            holder = {}
            async with InProcessClient.session(
                profile={
                    "model": "m", "provider": "p", "plugins": [],
                    "system_instructions": "You are a pirate.",
                    "completion_payload_schema": {"type": "object"},
                },
                embedded_factory=_factory_for(holder),
            ) as s:
                await s.ask("hi")
            sk = holder["client"].configure_tools_session_kwargs
            assert sk["system_instructions"] == "You are a pirate."
            assert sk["completion_payload_schema"] == {"type": "object"}

        asyncio.run(_run())

    def test_no_instructions_omits_keys(self):
        """Unset instructions/schema are OMITTED (so create_session applies its
        own defaults), not passed as None."""

        async def _run():
            holder = {}
            async with InProcessClient.session(
                model="m", embedded_factory=_factory_for(holder),
            ) as s:
                await s.ask("hi")
            sk = holder["client"].configure_tools_session_kwargs
            assert "system_instructions" not in sk
            assert "completion_payload_schema" not in sk

        asyncio.run(_run())

    def test_named_profile_resolved_from_disk(self, tmp_path):
        from jaato.in_process import _resolve_named_profile

        profiles = tmp_path / "profiles"
        profiles.mkdir()
        (profiles / "pirate.yaml").write_text(
            "name: pirate\n"
            "model: some-model\n"
            "provider: openrouter\n"
            "plugins: []\n"
            "system_instructions: 'Arr, ye be a pirate.'\n"
        )
        spec = _resolve_named_profile("pirate", str(tmp_path))
        assert spec["model"] == "some-model"
        assert spec["provider"] == "openrouter"
        assert spec["system_instructions"] == "Arr, ye be a pirate."

    def test_named_profile_needs_config_root(self):
        from jaato.in_process import _resolve_named_profile

        try:
            _resolve_named_profile("pirate", None)
        except ValueError as e:
            assert "config_root" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError without config_root")

    def test_named_profile_not_found(self, tmp_path):
        from jaato.in_process import _resolve_named_profile

        (tmp_path / "profiles").mkdir()
        try:
            _resolve_named_profile("ghost", str(tmp_path))
        except ValueError as e:
            assert "ghost" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError for missing profile")


class TestAgentCompleted:
    """The AGENT_COMPLETED bridge: the embedded runtime's on_agent_completed
    (the completion-gate payload) is emitted onto the facade emitter so
    s.complete() captures it (ex04 byte-exact). The missing half of PR3."""

    def test_complete_captures_agent_completed_payload(self):
        class _CompletingEmbedded(_FakeEmbedded):
            def __init__(self):
                super().__init__()
                self._ui_hooks = None

            def set_ui_hooks(self, hooks):
                self._ui_hooks = hooks

            def send_message(self, prompt, on_output=None, **_kwargs):
                # Completion-gated: no streamed text; signal the typed payload
                # through the lifecycle hook (as the real runtime does).
                if self._ui_hooks is not None:
                    self._ui_hooks.on_agent_completed(
                        agent_id="main", completed_at=None, success=True,
                        payload={"name": "Alice", "age": 30},
                    )
                return ""

        async def _run():
            async with InProcessClient.session(
                profile={
                    "model": "m", "provider": "p", "plugins": [],
                    "completion_payload_schema": {"type": "object"},
                },
                embedded_factory=lambda provider: _CompletingEmbedded(),
            ) as s:
                payload = await s.complete("Extract: Alice is 30")
            assert payload == {"name": "Alice", "age": 30}

        asyncio.run(_run())


class TestAgentPersona:
    """ex03: agent=<name> resolves .jaato/agents/<name>.md and threads it as
    system_instructions (the daemon-free analog of --agent). Reuses the shared
    resolve_agent loader (lifted from SessionManager)."""

    def _write_pirate(self, tmp_path):
        agents = tmp_path / "agents"
        agents.mkdir()
        (agents / "pirate.md").write_text(
            "---\ndescription: pirate\n---\n"
            "You are Captain {{name:Redbeard}}, a fearsome pirate. Arr!\n"
        )

    def test_agent_persona_threaded_as_system_instructions(self, tmp_path):
        self._write_pirate(tmp_path)

        async def _run():
            holder = {}
            async with InProcessClient.session(
                model="m", agent="pirate", config_root=str(tmp_path),
                embedded_factory=_factory_for(holder),
            ) as s:
                await s.ask("hi")
            si = holder["client"].configure_tools_session_kwargs.get(
                "system_instructions"
            )
            assert si is not None
            assert "Captain Redbeard" in si  # default param substituted

        asyncio.run(_run())

    def test_agent_params_substituted(self, tmp_path):
        self._write_pirate(tmp_path)

        async def _run():
            holder = {}
            async with InProcessClient.session(
                model="m", agent="pirate", agent_params={"name": "Hook"},
                config_root=str(tmp_path), embedded_factory=_factory_for(holder),
            ) as s:
                await s.ask("hi")
            si = holder["client"].configure_tools_session_kwargs["system_instructions"]
            assert "Captain Hook" in si

        asyncio.run(_run())

    def test_unknown_agent_raises(self, tmp_path):
        (tmp_path / "agents").mkdir()

        async def _run():
            try:
                async with InProcessClient.session(
                    model="m", agent="ghost", config_root=str(tmp_path),
                    embedded_factory=lambda provider: _FakeEmbedded(),
                ):
                    pass
            except ValueError as e:
                assert "ghost" in str(e)
            else:  # pragma: no cover
                raise AssertionError("expected ValueError for unknown agent")

        asyncio.run(_run())


class TestSuppressBaseInstructions:
    """The profile knob suppress_base_instructions (include_base = not it) is
    threaded so an embedded session honors the same base-composition choice the
    daemon does. Default False (compose persona ON TOP of base)."""

    def test_default_false_passed(self):
        async def _run():
            holder = {}
            async with InProcessClient.session(
                model="m", embedded_factory=_factory_for(holder),
            ) as s:
                await s.ask("hi")
            sk = holder["client"].configure_tools_session_kwargs
            assert sk["suppress_base_instructions"] is False

        asyncio.run(_run())

    def test_inline_profile_true_threaded(self):
        async def _run():
            holder = {}
            async with InProcessClient.session(
                profile={
                    "model": "m", "provider": "p", "plugins": [],
                    "suppress_base_instructions": True,
                },
                embedded_factory=_factory_for(holder),
            ) as s:
                await s.ask("hi")
            sk = holder["client"].configure_tools_session_kwargs
            assert sk["suppress_base_instructions"] is True

        asyncio.run(_run())

    def test_named_profile_extracts_knob(self, tmp_path):
        from jaato.in_process import _resolve_named_profile

        profiles = tmp_path / "profiles"
        profiles.mkdir()
        (profiles / "lean.yaml").write_text(
            "name: lean\nmodel: m\nsuppress_base_instructions: true\nplugins: []\n"
        )
        spec = _resolve_named_profile("lean", str(tmp_path))
        assert spec["suppress_base_instructions"] is True
