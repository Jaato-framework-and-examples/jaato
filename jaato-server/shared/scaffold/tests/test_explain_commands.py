"""scaffold explain surfaces the user-facing TUI commands (jaato #715).

`get_user_commands` is the operator's whole runtime control surface (auth,
permissions, plans, memory, profiles, sandbox, budgets) — invoked directly in
the TUI, not via the model's function calling.  Before this, `explain`
interrogated 14 scopes and none of them mentioned a single command; the way
out of a wedged permission-prompt session (`permissions allow *`) was found by
grepping plugin source, which is exactly what scaffold exists to replace.
"""
from shared.scaffold import explain, introspect


def _plugins_exposing_commands():
    """Registry names of every plugin whose instance exposes get_user_commands
    with at least one command — the ground truth the guard compares against."""
    from shared.plugins.registry import PluginRegistry

    reg = PluginRegistry()
    try:
        reg.discover()
    except Exception:
        return {}
    out = {}
    for name in reg.list_available():
        plugin = reg.get_plugin(name)
        try:
            cmds = list(plugin.get_user_commands() or [])
        except Exception:
            continue
        if cmds:
            out[name] = {getattr(c, "name", "?") for c in cmds}
    return out


def test_commands_scope_lists_every_command_exposing_plugin():
    # The guard: a plugin that ships a command must appear in `explain
    # commands`.  The failure mode this prevents is silent — a new plugin
    # ships a command, nothing breaks, and the command is simply
    # undiscoverable — which is how the original gap survived.
    exposing = _plugins_exposing_commands()
    data, text = explain.commands()
    for name, cmds in exposing.items():
        assert name in data, (
            f"plugin {name!r} exposes get_user_commands() but is missing "
            f"from `explain commands`")
        assert f"[{name}]" in text
        for c in cmds:
            assert c in data.get(name, []) or any(
                x["name"] == c for x in data.get(name, [])), (
                f"command {c!r} of plugin {name!r} missing from "
                f"`explain commands`")


def test_commands_scope_is_flat_and_grouped():
    _data, text = explain.commands()
    assert "user commands" in text
    # grouped by owning plugin, so a reader who knows the verb but not the
    # owner can find it
    assert "[permission]" in text
    assert "permissions" in text


def test_plugin_page_names_its_commands():
    _data, text = explain.plugin("permission")
    assert "permissions" in text
    # subcommands are where the actionable surface lives
    assert "allow" in text and "deny" in text


def test_overview_lists_the_commands_scope():
    _data, text = explain.overview()
    assert "explain commands" in text


def test_introspect_collects_subcommands():
    PL = introspect.plugins()
    pi = PL.get("permission")
    assert pi is not None
    names = [c.name for c in pi.commands]
    assert "permissions" in names
    perms = next(c for c in pi.commands if c.name == "permissions")
    # the permission plugin exposes CommandCompletion entries for its
    # subcommands — the collection must carry them
    for sub in ("allow", "deny", "suspend", "resume", "clear"):
        assert sub in perms.subcommands
