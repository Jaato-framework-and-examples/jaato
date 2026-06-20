"""Regression: memory plugin must not self-disable when a store tier is denied.

Under AppArmor confinement the global ``~/.jaato/memories`` and the daemon-cwd
relative workspace template are *correctly* denied at global registry init.
Before the fix, ``load_curated()`` raised ``PermissionError`` out of
``initialize()`` -> the registry disabled the plugin ("Plugin will not be
exposed"), so ``update_memory`` hit a non-exposed plugin and returned null even
though the per-session workspace store (wired by ``set_workspace_path()``)
existed.  See the 2026-06-20 LORA cascade bug report.
"""

from shared.plugins.memory.plugin import MemoryPlugin
from shared.plugins.memory import storage as storage_mod


def test_initialize_survives_denied_store_loads(monkeypatch):
    def deny_load(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(storage_mod.MemoryStorage, "load_curated", deny_load)

    plugin = MemoryPlugin()
    # Must NOT raise — the plugin stays exposed; both tiers degrade to empty.
    plugin.initialize({})
    assert plugin._indexer is not None
    assert plugin._global_indexer is not None


def test_set_workspace_path_loads_real_store_after_denied_init(monkeypatch, tmp_path):
    """A denied global-init must still allow set_workspace_path() to wire the
    accessible workspace-tier store."""
    real_load = storage_mod.MemoryStorage.load_curated
    state = {"deny": True}

    def maybe_deny(self):
        if state["deny"]:
            raise PermissionError(13, "Permission denied")
        return real_load(self)

    monkeypatch.setattr(storage_mod.MemoryStorage, "load_curated", maybe_deny)

    plugin = MemoryPlugin()
    plugin.initialize({})            # both init loads denied -> degrade, no raise
    state["deny"] = False            # the workspace tier is accessible
    plugin.set_workspace_path(str(tmp_path))   # must wire the real store
    assert plugin._storage is not None
