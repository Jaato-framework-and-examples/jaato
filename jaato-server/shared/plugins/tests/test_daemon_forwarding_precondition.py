"""A plugin that forwards to the daemon MUST be able to find the registry.

``DaemonForwardingMixin`` decides runner-side vs daemon-side by looking for
``runner_rpc_client`` on ``self._plugin_registry``.  That attribute is set by
the ``set_plugin_registry`` lifecycle hook — which ``PluginRegistry`` calls
only ``if hasattr(plugin, 'set_plugin_registry')`` (registry.py:1117).

So a plugin that uses the mixin and never defines the hook gets
``_plugin_registry`` unset, ``getattr(..., None)`` returns ``None``, and the
mixin reads that as "no runner client attached, therefore I AM the daemon".
Every call runs in-process on the runner instead of forwarding — silently,
with no error anywhere.

That is exactly what happened to ``list_siblings``: the runner-side instance
answered every call and hit its own "no session manager attached" guard, on the
driver-created cascade path the feature exists for.  A MISSING HOOK WAS
INDISTINGUISHABLE FROM BEING DAEMON-SIDE — the same absent-vs-empty collapse as
``_injection_queue`` (#589) and the phantom entry-point group (#595).

Found by the cascade-coordination example, from one ToolResult line.
"""
import inspect

import pytest

from shared.plugins.daemon_forwarding import DaemonForwardingMixin


def _forwarding_plugin_classes():
    """Every plugin class in the tree that mixes in daemon forwarding."""
    import importlib
    import pkgutil
    import shared.plugins as pkg

    found = []
    for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        if ".tests" in mod.name or mod.name.endswith("_test"):
            continue
        try:
            m = importlib.import_module(mod.name)
        except Exception:
            continue                      # optional deps — not our concern here
        for _, obj in vars(m).items():
            if (inspect.isclass(obj)
                    and issubclass(obj, DaemonForwardingMixin)
                    and obj is not DaemonForwardingMixin
                    and obj.__module__ == mod.name):
                found.append(obj)
    return found


def test_the_mixin_is_actually_used_somewhere():
    """Guard the guard: if nothing mixes it in, the test below is vacuous."""
    assert _forwarding_plugin_classes(), (
        "no plugin uses DaemonForwardingMixin — this test would pass by "
        "iterating an empty list")


@pytest.mark.parametrize("cls", _forwarding_plugin_classes(),
                         ids=lambda c: c.__name__)
def test_a_forwarding_plugin_defines_set_plugin_registry(cls):
    assert hasattr(cls, "set_plugin_registry"), (
        f"{cls.__name__} mixes in DaemonForwardingMixin but defines no "
        f"set_plugin_registry hook. PluginRegistry calls that hook only when "
        f"it exists, so _plugin_registry stays unset, the mixin cannot find "
        f"runner_rpc_client, and EVERY forwarded call silently runs in-process "
        f"on the runner instead."
    )


@pytest.mark.parametrize("cls", _forwarding_plugin_classes(),
                         ids=lambda c: c.__name__)
def test_the_hook_actually_stores_it_where_the_mixin_looks(cls):
    """Defining the hook is not enough — it must store the attribute the
    mixin reads.  Storing it under any other name reproduces the bug while
    passing the check above."""
    from types import SimpleNamespace

    plugin = cls.__new__(cls)            # skip __init__; only the hook matters
    plugin.set_plugin_registry(SimpleNamespace(runner_rpc_client="SENTINEL"))
    assert plugin._runner_rpc_client_handle() == "SENTINEL", (
        f"{cls.__name__}.set_plugin_registry does not store the registry as "
        f"_plugin_registry, so the mixin still cannot find the runner client"
    )
