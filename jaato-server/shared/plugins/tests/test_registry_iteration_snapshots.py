"""A shared registry is read while another thread mutates it (issue #938).

Subagents share the parent's :class:`~shared.plugins.registry.PluginRegistry`
— that is the cascade-sharing design — so ``spawn_subagent`` calls
``expose_tool`` (``self._exposed.add(...)``) on the spawning thread while the
parent's model thread is part-way through a read that walks the same set.

The casualty reported in #938 was :meth:`PluginRegistry.get_plugin_for_tool`,
reached from ``JaatoSession._apply_tool_scopes`` on *every* provider call::

    RuntimeError: Set changed size during iteration

which escaped the model loop and terminated the parent's turn, reaching the
caller as an opaque ``RunnerCallError``.

These tests pin the invariant that fixes it: **every read path iterates a
snapshot, never the live container**.  Three layers:

1. :class:`TestSpawnDuringScan` — deterministic reproduction.  A plugin
   mutates ``_exposed`` from inside the loop body, which is precisely what a
   concurrent spawn does at a GIL yield point.  Fails with
   ``RuntimeError`` before the fix.
2. :class:`TestConcurrentSpawnStress` — the real shape, with real threads.
   Cannot produce a false failure: with the snapshot in place there is no
   interleaving that raises.
3. :class:`TestNoLiveContainerIteration` — an AST guard over ``registry.py``
   so a future read path cannot reintroduce the shape silently.
"""

import ast
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

from shared.plugins import registry as registry_module
from shared.plugins.registry import PluginRegistry


class _StubPlugin:
    """Minimal ToolPlugin stand-in the registry's read paths can walk.

    Only the surface the read paths touch is implemented: ``name`` and
    ``get_executors()``.  ``on_get_executors`` is the injection point the
    deterministic test uses to mutate the registry from *inside* the scan
    loop — the same position a GIL switch hands to a spawning thread.
    """

    def __init__(
        self,
        name: str,
        executors: Dict[str, Callable[..., Any]] = None,
        on_get_executors: Callable[[], None] = None,
    ) -> None:
        self.name = name
        self._executors = executors or {}
        self._on_get_executors = on_get_executors

    def get_executors(self) -> Dict[str, Callable[..., Any]]:
        if self._on_get_executors is not None:
            self._on_get_executors()
        return dict(self._executors)

    def get_tool_schemas(self) -> List[Any]:
        return []


def _register(registry: PluginRegistry, plugin: _StubPlugin) -> None:
    """Put ``plugin`` straight into the registry's exposed state.

    Bypasses ``expose_tool`` deliberately: these tests are about the read
    paths' iteration, not about the exposure lifecycle, and the real
    lifecycle would drag in discovery, config augmentation and
    ``initialize()``.
    """
    registry._plugins[plugin.name] = plugin
    registry._exposed.add(plugin.name)


class TestSpawnDuringScan:
    """The registry is mutated from inside a read path's loop body."""

    def test_get_plugin_for_tool_survives_exposure_during_scan(self):
        """A plugin exposed mid-scan must not kill the lookup.

        This is #938's traceback reduced to one thread: the mutation
        happens at the exact point a GIL switch would hand control to a
        spawning thread.  Before the fix this raises
        ``RuntimeError: Set changed size during iteration``.
        """
        registry = PluginRegistry()
        spawned = {"count": 0}

        def spawn_a_subagents_plugin() -> None:
            # What expose_tool() does to _exposed, in the window the
            # parent's scan is open.
            spawned["count"] += 1
            child = _StubPlugin(f"child_{spawned['count']}")
            registry._plugins[child.name] = child
            registry._exposed.add(child.name)

        _register(
            registry,
            _StubPlugin("memory", on_get_executors=spawn_a_subagents_plugin),
        )
        _register(
            registry,
            _StubPlugin("subagent", {"spawn_subagent": lambda **kw: None}),
        )

        # A miss first: set iteration order is hash-seed dependent, and a
        # lookup that finds its tool early returns before ever reaching the
        # mutating plugin.  Asking for a tool nobody provides forces the
        # whole scan, so the mutation lands inside it on every run.
        assert registry.get_plugin_for_tool("no_such_tool") is None
        assert spawned["count"] >= 1, "the mutation never ran; test is inert"

        registry.invalidate_tool_cache()
        found = registry.get_plugin_for_tool("spawn_subagent")
        assert found is not None
        assert found.name == "subagent"

    def test_get_plugin_for_tool_survives_unexposure_during_scan(self):
        """A plugin *removed* mid-scan must not raise either.

        The snapshot keeps iterating a name the live set no longer holds,
        so the lookup that follows must tolerate its absence.  The
        ``try``/``except`` already wrapping the loop body is what absorbs
        it; this pins that it still does.
        """
        registry = PluginRegistry()

        def unexpose_everything_else() -> None:
            registry._exposed.discard("doomed")
            registry._plugins.pop("doomed", None)

        _register(
            registry,
            _StubPlugin("memory", on_get_executors=unexpose_everything_else),
        )
        _register(registry, _StubPlugin("doomed", {"gone_tool": lambda: None}))
        _register(registry, _StubPlugin("cli", {"run_command": lambda: None}))

        # Whichever order the snapshot yields, the scan completes.
        assert registry.get_plugin_for_tool("run_command") is not None
        assert registry.get_plugin_for_tool("nonexistent_tool") is None

    def test_exposed_read_paths_survive_mutation_during_scan(self):
        """Every exposed-walking read path, not just the one that surfaced.

        #938 notes the shape "is worth checking wherever ``_exposed`` is
        iterated — this is the instance that surfaced, not necessarily the
        only one".  These are the read paths a live turn reaches.
        """
        readers = [
            "get_exposed_tool_schemas",
            "get_exposed_executors",
            "get_all_tool_names",
            "get_tool_status",
            "get_enabled_tool_schemas",
            "get_enabled_executors",
            "get_core_tool_schemas",
            "get_streaming_tools",
            "get_auto_approved_tools",
            "get_exposed_user_commands",
        ]
        for reader in readers:
            registry = PluginRegistry()
            counter = {"n": 0}

            def spawn() -> None:
                counter["n"] += 1
                name = f"child_{counter['n']}"
                registry._plugins[name] = _StubPlugin(name)
                registry._exposed.add(name)

            _register(registry, _StubPlugin("memory", on_get_executors=spawn))
            _register(registry, _StubPlugin("cli", {"run_command": lambda: None}))

            method = getattr(registry, reader, None)
            if method is None:  # pragma: no cover - guards a rename
                pytest.fail(f"PluginRegistry has no read path {reader!r}")
            # Must not raise.  The result itself is not asserted: what a
            # reader sees of an in-flight mutation is deliberately
            # unspecified (see the class docstring on registry.py).
            method()


class TestConcurrentSpawnStress:
    """The real shape: a spawning thread against a scanning thread.

    This test cannot fail spuriously — with the snapshot in place no
    interleaving raises.  It needs help to be *sensitive*, though: at the
    default 5 ms switch interval a scan over a handful of plugins finishes
    inside one time slice, so the interpreter rarely hands the spawning
    thread the window.  ``sys.setswitchinterval`` is turned down for the
    duration so the interleaving that #938 hit in production (where the
    scan walks real plugins doing real work) is reachable in a unit test.
    """

    def test_scan_and_spawn_concurrently(self):
        registry = PluginRegistry()
        _register(registry, _StubPlugin("subagent", {"spawn_subagent": lambda: None}))
        for i in range(8):
            _register(registry, _StubPlugin(f"plugin_{i}", {f"tool_{i}": lambda: None}))

        stop = threading.Event()
        errors: List[BaseException] = []

        def spawn_loop() -> None:
            i = 0
            try:
                while not stop.is_set():
                    name = f"child_{i % 32}"
                    registry._plugins[name] = _StubPlugin(name, {f"child_tool_{i}": lambda: None})
                    registry._exposed.add(name)
                    registry._exposed.discard(f"child_{(i + 16) % 32}")
                    i += 1
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        previous_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        spawner = threading.Thread(target=spawn_loop, daemon=True)
        spawner.start()
        try:
            for i in range(400):
                # A cache MISS every iteration — the path #938 lands on.
                registry.invalidate_tool_cache()
                registry.get_plugin_for_tool(f"never_exists_{i}")
                registry.get_exposed_executors()
        finally:
            stop.set()
            spawner.join(timeout=5.0)
            sys.setswitchinterval(previous_interval)

        assert not errors, f"mutating thread raised: {errors[0]!r}"


class TestNoLiveContainerIteration:
    """AST guard: registry read paths may not iterate a live container.

    The fix for #938 is an invariant, not a one-line patch, so it needs a
    check that survives the next read path someone adds.  A ``for`` whose
    iterable is a bare ``self._x`` (or ``self._x.items()`` / ``.keys()`` /
    ``.values()``) is the shape that raised; wrapping it in ``list()`` /
    ``sorted()`` / ``set()``, or building a fresh set with ``|``, is not.
    """

    _SNAPSHOTTING_CALLS = {"list", "sorted", "set", "tuple", "frozenset", "dict"}
    _VIEWS = {"items", "keys", "values"}

    @classmethod
    def _live_container_loops(cls, tree: ast.AST) -> List[int]:
        offenders: List[int] = []

        def is_live(node: ast.AST) -> bool:
            # self._x.items() / .keys() / .values()
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr in cls._VIEWS
                    and cls._is_private_self_attr(func.value)
                ):
                    return True
                return False
            # self._x
            return cls._is_private_self_attr(node)

        for node in ast.walk(tree):
            iters = []
            if isinstance(node, (ast.For, ast.AsyncFor)):
                iters.append(node.iter)
            elif isinstance(
                node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
            ):
                iters.extend(gen.iter for gen in node.generators)
            for it in iters:
                if is_live(it):
                    offenders.append(it.lineno)
        return sorted(offenders)

    @staticmethod
    def _is_private_self_attr(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("_")
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        )

    def test_registry_iterates_only_snapshots(self):
        source_path = Path(registry_module.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))

        offenders = self._live_container_loops(tree)

        assert not offenders, (
            "registry.py iterates a live registry container at "
            f"{source_path.name}:{offenders} — a concurrent subagent spawn "
            "mutates the same object and the loop dies with "
            "'changed size during iteration' (issue #938).  Iterate a "
            "snapshot instead: `for name in list(self._exposed):`."
        )

    def test_guard_detects_the_shape_it_exists_to_catch(self):
        """The guard must actually fail on the pre-fix source."""
        tree = ast.parse(
            "class R:\n"
            "    def read(self):\n"
            "        for name in self._exposed:\n"
            "            pass\n"
        )
        assert self._live_container_loops(tree) == [3]

    def test_guard_accepts_a_snapshot(self):
        for snippet in (
            "for name in list(self._exposed): pass",
            "for name in sorted(self._exposed): pass",
            "for name in self._exposed | self._enrichment_only: pass",
            "for k, v in list(self._plugins.items()): pass",
        ):
            tree = ast.parse(f"class R:\n    def read(self):\n        {snippet}\n")
            assert self._live_container_loops(tree) == [], snippet
