"""Fix B: registry.set_workspace_path / set_config_root broadcast to ALL
registered plugins, not just exposed ones — closing the #344-class pool-slot
propagation gap where a registered-but-unexposed runner-tier plugin (e.g. a
pool-slot notebook plugin initialized at template time) silently kept its
init-time (launch-dir) default instead of the per-session workspace."""

from shared.plugins.registry import PluginRegistry


class _WsPlugin:
    def __init__(self):
        self.ws = None
        self.cfg = None

    def set_workspace_path(self, path):
        self.ws = path

    def set_config_root(self, path):
        self.cfg = path


def _bare_registry():
    # Skip __init__ (directory discovery) — exercise only the broadcast.
    reg = PluginRegistry.__new__(PluginRegistry)
    reg._plugins = {}
    reg._exposed = set()
    reg._workspace_path = None
    reg._config_root = None
    return reg


def test_set_workspace_path_reaches_unexposed_plugin():
    reg = _bare_registry()
    p = _WsPlugin()
    reg._plugins["notebookish"] = p          # registered but NOT in _exposed
    reg.set_workspace_path("/session/ws")
    assert p.ws == "/session/ws"             # B: unexposed plugin still notified


def test_set_config_root_reaches_unexposed_plugin():
    reg = _bare_registry()
    p = _WsPlugin()
    reg._plugins["notebookish"] = p
    reg.set_config_root("/session/cfg")
    assert p.cfg == "/session/cfg"
