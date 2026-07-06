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
import os as _os

import pytest

from jaato import session, _bundle_inline_profile
from jaato_embedded import InProcessClient, InProcessEventEmitter
from jaato_sdk.events import AgentOutputEvent, EventType, TurnCompletedEvent


@pytest.fixture(autouse=True)
def _restore_os_environ():
    """InProcessClient.connect sets workspace/config-root os.environ keys with no
    reset (mirroring the daemon's one-session-per-process runner — see #451).
    Snapshot + restore around each test so these tests don't leak workspace env
    into others — notably the subagent suite's find_workspace_root / cwd /
    profile-discovery tests, which read os.environ['workspaceRoot']."""
    snapshot = dict(_os.environ)
    yield
    _os.environ.clear()
    _os.environ.update(snapshot)


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
    """Build an ``embedded_factory(provider, workspace_path, config_root) ->
    client`` that stashes the constructed fake + the isolation args in
    ``holder`` for post-run assertions."""
    def _make(provider, workspace_path=None, config_root=None):
        client = _FakeEmbedded()
        holder["client"] = client
        holder["provider"] = provider
        holder["workspace_path"] = workspace_path
        holder["config_root"] = config_root
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
                embedded_factory=lambda provider, *a, **k:_MixedEmbedded(),
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
            # No ``plugins`` was passed -> None (INHERIT / drag-all), preserved
            # NOT collapsed to [] — full IPC parity (None=inherit, []=override-none).
            assert fake.configure_tools_session_kwargs == {
                "plugin_configs": {"openrouter": {"api_key": "sk-or-plain"}},
                "plugins": None,
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
                    embedded_factory=lambda provider, *a, **k:_NoAuth(),
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
                embedded_factory=lambda provider, *a, **k:_FakeEmbedded(),
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
                embedded_factory=lambda provider, *a, **k:_FakeEmbedded(),
            ) as s:
                return await s.ask("hi")

        assert asyncio.run(_run()) == "Hello world"

    def test_env_file_none_skips_load(self):
        async def _run():
            async with InProcessClient.session(
                model="m", env_file=None,
                embedded_factory=lambda provider, *a, **k:_FakeEmbedded(),
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

        from jaato_embedded.permission import InProcessChannel

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
        from jaato_embedded.client import _resolve_named_profile

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
        from jaato_embedded.client import _resolve_named_profile

        try:
            _resolve_named_profile("pirate", None)
        except ValueError as e:
            assert "config_root" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError without config_root")

    def test_named_profile_not_found(self, tmp_path):
        from jaato_embedded.client import _resolve_named_profile

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
                embedded_factory=lambda provider, *a, **k:_CompletingEmbedded(),
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
                    embedded_factory=lambda provider, *a, **k:_FakeEmbedded(),
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
        from jaato_embedded.client import _resolve_named_profile

        profiles = tmp_path / "profiles"
        profiles.mkdir()
        (profiles / "lean.yaml").write_text(
            "name: lean\nmodel: m\nsuppress_base_instructions: true\nplugins: []\n"
        )
        spec = _resolve_named_profile("lean", str(tmp_path))
        assert spec["suppress_base_instructions"] is True


class TestStream:
    """PR4: the facade's stream() rides UNCHANGED on InProcessClient — it
    subscribes AGENT_OUTPUT + the terminal and drives send_message, exactly the
    contract ask() uses. Yields each model chunk; stops at TURN_COMPLETED."""

    class _Streaming(_FakeEmbedded):
        def send_message(self, prompt, on_output=None, **_kwargs):
            for c in ("The ", "quick ", "brown ", "fox"):
                if on_output is not None:
                    on_output("model", c, "append")
            return "The quick brown fox"

    def test_stream_yields_chunks_in_order(self):
        async def _run():
            chunks = []
            async with InProcessClient.session(
                model="m", embedded_factory=lambda p, *a, **k:self._Streaming(),
            ) as s:
                async for chunk in s.stream("go"):
                    chunks.append(chunk)
            assert chunks == ["The ", "quick ", "brown ", "fox"]
            assert "".join(chunks) == "The quick brown fox"

        asyncio.run(_run())

    def test_stream_filters_non_model_source(self):
        class _Mixed(_FakeEmbedded):
            def send_message(self, prompt, on_output=None, **_kwargs):
                if on_output is not None:
                    on_output("system", "[setup]", "append")  # filtered out
                    on_output("model", "answer", "append")
                return "answer"

        async def _run():
            chunks = []
            async with InProcessClient.session(
                model="m", embedded_factory=lambda p, *a, **k:_Mixed(),
            ) as s:
                async for chunk in s.stream("go"):
                    chunks.append(chunk)
            assert chunks == ["answer"]  # the system chunk is excluded

        asyncio.run(_run())

    def test_stream_empty_turn_terminates(self):
        """A turn with no output still terminates on TURN_COMPLETED — the
        iterator ends rather than hanging."""

        class _Silent(_FakeEmbedded):
            def send_message(self, prompt, on_output=None, **_kwargs):
                return ""  # no chunks

        async def _run():
            chunks = []
            async with InProcessClient.session(
                model="m", embedded_factory=lambda p, *a, **k:_Silent(),
            ) as s:
                async for chunk in s.stream("go"):
                    chunks.append(chunk)
            assert chunks == []

        asyncio.run(_run())


class TestMultiTurnContinuity:
    """PR4: multiple turns ride on ONE embedded session (conversation memory
    lives in the embedded JaatoSession; here we prove the facade keeps driving
    the same session across asks rather than re-creating one)."""

    def test_two_asks_drive_the_same_session(self):
        class _Counting(_FakeEmbedded):
            def __init__(self):
                super().__init__()
                self.sends = 0

            def send_message(self, prompt, on_output=None, **_kwargs):
                self.sends += 1
                if on_output is not None:
                    on_output("model", f"turn{self.sends}", "write")
                return f"turn{self.sends}"

        async def _run():
            holder = {}

            def _make(provider, *a, **k):
                c = _Counting()
                holder["c"] = c
                return c

            async with InProcessClient.session(
                model="m", embedded_factory=_make,
            ) as s:
                a1 = await s.ask("first")
                a2 = await s.ask("second")
            assert a1 == "turn1"
            assert a2 == "turn2"
            # Both turns went to the SAME embedded session (not a fresh one).
            assert holder["c"].sends == 2

        asyncio.run(_run())


class TestConfigRootIsolation:
    """The embedded runtime must be PINNED to <workspace>/.jaato — not left to
    inherit the operator's home-tier ~/.jaato. Daniel caught an embedded
    gemini session self-describing as 'Claude Code Agent' (read from the
    operator's ~/.jaato prompt library): the daemon pins config_root; the
    embedded path must pass it to the JaatoClient (the registry's set_config_root
    does NOT cover prompt-library / instruction discovery)."""

    def test_embedded_factory_gets_pinned_config_root(self):
        async def _run():
            holder = {}
            async with InProcessClient.session(
                model="m", workspace_path="/tmp/ws-x",
                embedded_factory=_factory_for(holder),
            ) as s:
                await s.ask("hi")
            # config_root defaults to <workspace>/.jaato and is PASSED to the
            # embedded runtime (None would inherit ~/.jaato — the contamination).
            assert holder["config_root"] == "/tmp/ws-x/.jaato"
            assert holder["workspace_path"] == "/tmp/ws-x"

        asyncio.run(_run())

    def test_explicit_config_root_passed_through(self):
        async def _run():
            holder = {}
            async with InProcessClient.session(
                model="m", workspace_path="/tmp/ws",
                config_root="/etc/jaato-conf",
                embedded_factory=_factory_for(holder),
            ) as s:
                await s.ask("hi")
            assert holder["config_root"] == "/etc/jaato-conf"

        asyncio.run(_run())


class TestPluginsIPCParity:
    """Full IPC parity for the plugins gate: None (missing) is INHERIT/drag-all,
    [] is an explicit override to NONE, [list] is exactly those. None must be
    PRESERVED through to create_session (NOT collapsed to []) — else a base
    profile that omits plugins would wrongly drag nothing instead of all,
    diverging from the daemon/runner's create_session(plugins=...) semantics."""

    def _session_plugins(self, **session_kw):
        holder = {}

        async def _run():
            async with InProcessClient.session(
                model="m", embedded_factory=_factory_for(holder), **session_kw,
            ) as s:
                await s.ask("hi")

        asyncio.run(_run())
        return holder["client"].configure_tools_session_kwargs["plugins"]

    def test_missing_plugins_is_none_inherit(self):
        # No plugins kwarg -> None (inherit / drag-all), NOT [].
        assert self._session_plugins() is None

    def test_explicit_empty_is_override_none(self):
        assert self._session_plugins(plugins=[]) == []

    def test_explicit_list_passes_through(self):
        assert self._session_plugins(plugins=["cli"]) == ["cli"]

    def test_preload_and_scope_modifiers_parsed(self):
        # (preload) / ([tools]) modifiers are parsed like the daemon
        # (parse_plugin_list): clean names -> plugins, force-eager set ->
        # preloaded_plugins. Pre-fix the raw "cli(preload)" matched no plugin
        # (dropped) — an IPC-parity gap.
        holder = {}

        async def _run():
            async with InProcessClient.session(
                model="m", embedded_factory=_factory_for(holder),
                plugins=["cli(preload)", "web_search"],
            ) as s:
                await s.ask("hi")

        asyncio.run(_run())
        sk = holder["client"].configure_tools_session_kwargs
        assert sk["plugins"] == ["cli", "web_search"]   # clean names
        assert sk["preloaded_plugins"] == {"cli"}       # force-eager set


class _FakeExecutor:
    def __init__(self) -> None:
        self._map = {}

    def register(self, name, fn) -> None:
        self._map[name] = fn


class _FakeSession:
    """Minimal stand-in exposing the two attributes the in-process client-tool
    registration touches: ``_tools`` (the wire schema list) and ``_executor``."""
    def __init__(self) -> None:
        self._tools = []
        self._executor = _FakeExecutor()


class _FakeEmbeddedWithSession(_FakeEmbedded):
    def __init__(self) -> None:
        super().__init__()
        self._session = _FakeSession()

    def get_session(self):
        return self._session


class TestClientToolsForwarding:
    """Host/client tools forwarded through the facade — the ex05 gap. The
    _SessionContext pops ``client_tools``, registers them BEFORE create_session
    (IPC ordering), and create_session appends each schema to the session wire +
    its handler to the executor so the model SEES and CALLS them in-process."""

    def test_client_tools_land_on_session_wire_and_executor(self):
        holder = {}

        def _factory(provider, *a, **k):
            c = _FakeEmbeddedWithSession()
            holder["client"] = c
            return c

        tools = [{
            "name": "host_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            "handler": lambda args: {"temp": "24C", "echo": args.get("city")},
            "auto_approve": True,
        }]

        async def _run():
            async with InProcessClient.session(
                model="m", embedded_factory=_factory, client_tools=tools,
            ) as s:
                await s.ask("hi")

        asyncio.run(_run())
        sess = holder["client"]._session
        assert "host_weather" in [t.name for t in sess._tools]   # model SEES it
        assert "host_weather" in sess._executor._map             # model can CALL it
        # the registered handler is the one we supplied (runs in-process)
        assert sess._executor._map["host_weather"]({"city": "Paris"}) == {
            "temp": "24C", "echo": "Paris"}

    def test_no_client_tools_is_noop(self):
        # Without client_tools the embedded fake (no get_session) is never
        # touched — registration is gated on self._client_tools.
        holder = {}
        async def _run():
            async with InProcessClient.session(
                model="m", embedded_factory=_factory_for(holder),
            ) as s:
                await s.ask("hi")
        asyncio.run(_run())
        assert holder["client"].closed


class TestSessionContextForSpawn:
    """The embedded path must populate the session-context workspace_root
    (ContextVar + os.environ) like the daemon's runner — else the subagent
    spawn path's get_workspace_root() returns None and spawn_subagent fails.
    The ContextVar covers context-inheriting threads; os.environ covers the
    subagent ThreadPoolExecutor, which does not inherit it."""

    def test_workspace_root_populated_main_and_pool(self):
        import os as _os
        from concurrent.futures import ThreadPoolExecutor
        from shared.session_context import get_workspace_root
        holder = {}

        async def _run():
            async with InProcessClient.session(
                model="m", workspace_path="/tmp/ws-spawn",
                embedded_factory=_factory_for(holder),
            ) as s:
                # ContextVar set (inheriting context)
                assert get_workspace_root() == "/tmp/ws-spawn"
                # subagent pool: no ContextVar inheritance -> os.environ fallback
                # (mirrors jaato_runtime.py's `get_workspace_root() or os.environ`)
                resolve = lambda: get_workspace_root() or _os.environ.get("workspaceRoot")
                assert ThreadPoolExecutor(1).submit(resolve).result() == "/tmp/ws-spawn"

        asyncio.run(_run())


class TestSubagentAutoContinue:
    """Part B: an idle lead resumes when a subagent injects its result. The
    session's continuation callback -> _schedule_lead_continuation runs a
    follow-up lead turn (send_message with the drained text) — the embedded
    analog of the daemon's run-loop resume. The full wiring
    (set_continuation_callback on the real session) is verified against a real
    embedded session; here we assert the continuation MECHANISM + coalescing."""

    def test_continuation_runs_a_lead_turn_with_drained_text(self):
        seen = []

        class _Recording(_FakeEmbeddedWithSession):
            def send_message(self, prompt, on_output=None, **k):
                seen.append(prompt)
                return "blurb"

        holder = {}

        def _factory(provider, *a, **k):
            c = _Recording()
            holder["client"] = c
            return c

        async def _run():
            async with InProcessClient.session(
                model="m", embedded_factory=_factory,
            ) as s:
                client = s.client
                client._schedule_lead_continuation("[SUBAGENT COMPLETED] result=42")
                for _ in range(200):
                    await asyncio.sleep(0.01)
                    if not client._continuation_busy:
                        break

        asyncio.run(_run())
        assert seen == ["[SUBAGENT COMPLETED] result=42"]

    def test_overlapping_continuations_coalesce(self):
        seen = []

        class _Recording(_FakeEmbeddedWithSession):
            def send_message(self, prompt, on_output=None, **k):
                seen.append(prompt)
                return "ok"

        holder = {}

        def _factory(provider, *a, **k):
            c = _Recording()
            holder["client"] = c
            return c

        async def _run():
            async with InProcessClient.session(
                model="m", embedded_factory=_factory,
            ) as s:
                client = s.client
                # two arrivals before the first turn settles -> the 2nd coalesces
                # into _continuation_pending (only the latest), then runs after.
                client._schedule_lead_continuation("first")
                client._schedule_lead_continuation("second")
                assert client._continuation_pending == "second"
                for _ in range(200):
                    await asyncio.sleep(0.01)
                    if not client._continuation_busy and client._continuation_pending is None:
                        break

        asyncio.run(_run())
        assert seen == ["first", "second"]


class TestSessionTerminatedEmission:
    """#5D: on_session_quiescent (fired by JaatoSession after signal_completion +
    turn wrap-up, jaato_session.py:5204) must emit SESSION_TERMINATED to the
    facade — the event a delegation example awaits. _InProcessUIHooks.__getattr__
    no-ops every on_*, so this needs to be an EXPLICIT method or the terminal
    event is silently swallowed."""

    def test_on_session_quiescent_emits_session_terminated(self):
        from jaato_embedded.client import _InProcessUIHooks
        from jaato_sdk.events import SessionTerminatedEvent

        seen = []
        hooks = _InProcessUIHooks(seen.append)
        hooks.on_session_quiescent(agent_id="main", reason="natural")
        assert len(seen) == 1
        assert isinstance(seen[0], SessionTerminatedEvent)
        assert seen[0].agent_id == "main"
        assert seen[0].reason == "natural"


class TestOutstandingSubagentGate:
    """#5C gate hardening: the completion nudge must NOT fire while a subagent is
    still working. The original gate keyed on session.is_running, which flickers
    (False pre-spawn + between turns) -> the lead nudged + signaled prematurely.
    The fix keys on terminal injects (_on_lead_inject parses
    [SUBAGENT ... event=COMPLETED|ERROR|CANCELLED]) vs _active_sessions
    membership -> a stable 'still outstanding' signal."""

    def test_gate_keys_on_terminal_injects_not_is_running(self):
        from jaato_embedded import InProcessClient

        c = InProcessClient(model="m")
        c._schedule_lead_continuation = lambda text: None  # decouple from the loop

        class _Sub:
            _active_sessions = {"subagent_1": {}}

        class _Reg:
            def get_plugin(self, n):
                return _Sub() if n == "subagent" else None

        c._registry = _Reg()
        # spawned, no terminal inject -> outstanding (gate closed, no premature nudge)
        assert c._has_outstanding_subagents() is True
        # a mid-run STATUS inject (IDLE) is NOT terminal -> still outstanding
        c._on_lead_inject("[SUBAGENT agent_id=subagent_1 event=IDLE]\nstill working")
        assert c._has_outstanding_subagents() is True
        # the terminal COMPLETED inject clears it -> gate opens
        c._on_lead_inject("[SUBAGENT agent_id=subagent_1 event=COMPLETED]\nFound 3 facts")
        assert "subagent_1" in c._completed_subagent_ids
        assert c._has_outstanding_subagents() is False


class TestFacadeRecoveryMode:
    """`recovery=True` (daemon modes only) swaps the client for its auto-reconnect
    variant — IPCRecoveryClient (ipc) / WSRecoveryClient (ws); in_process+recovery
    is an error (no daemon to reconnect to)."""

    def test_recovery_routing_picks_the_reconnect_client(self):
        import jaato as ip
        from jaato_sdk import (
            WSRecoveryClient,
            IPCRecoveryClient,
            WSClient,
            IPCClient,
        )

        assert isinstance(
            ip.session(mode="ws", recovery=True, url="ws://x:1")._client,
            WSRecoveryClient,
        )
        assert isinstance(ip.session(mode="ws", url="ws://x:1")._client, WSClient)
        assert isinstance(
            ip.session(mode="ipc", recovery=True)._client, IPCRecoveryClient
        )
        assert isinstance(ip.session(mode="ipc")._client, IPCClient)

    def test_in_process_recovery_is_an_error(self):
        import jaato as ip

        with pytest.raises(ValueError, match="daemon transport"):
            ip.session(mode="in_process", recovery=True)
