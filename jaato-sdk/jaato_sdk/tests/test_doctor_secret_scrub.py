"""jaato-doctor preflight: an unscrubbed subprocess surface is WARNed (#863)."""
import sys
from types import SimpleNamespace

from jaato_sdk import doctor


def _diag(code, profile, where):
    return SimpleNamespace(code=code, profile=profile, where=where)


def _patch_validator(monkeypatch, diags):
    mod = SimpleNamespace(validate_workspace=lambda ws, config_root=None: diags)
    monkeypatch.setitem(sys.modules, "shared.scaffold.validate", mod)


def test_warns_naming_each_leaky_profile(monkeypatch):
    _patch_validator(monkeypatch, [
        _diag("secret_scrub_disabled", "dev-desktop", "scrub_secret_env"),
        _diag("invalid_scrub_secret_env", "typo", "plugin_configs.mcp.scrub_secret_env"),
        _diag("unknown_knob", "other", "plugin_configs.cli.x"),
    ])
    [c] = doctor.check_secret_scrub("/ws", None)
    assert c.status == doctor.WARN
    assert "dev-desktop (scrub_secret_env)" in c.detail
    assert "typo (plugin_configs.mcp.scrub_secret_env)" in c.detail
    assert "other" not in c.detail
    assert "!GH_TOKEN" in c.detail


def test_passes_when_every_surface_is_scrubbed(monkeypatch):
    _patch_validator(monkeypatch, [_diag("unknown_knob", "p", "x")])
    [c] = doctor.check_secret_scrub("/ws", None)
    assert c.status == doctor.PASS


def test_client_only_install_warns_instead_of_passing_on_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "shared.scaffold.validate", None)
    [c] = doctor.check_secret_scrub("/ws", None)
    assert c.status == doctor.WARN and "not importable" in c.detail


def test_validator_failure_is_a_warn_not_a_crash(monkeypatch):
    def boom(ws, config_root=None):
        raise RuntimeError("no such workspace")
    monkeypatch.setitem(sys.modules, "shared.scaffold.validate",
                        SimpleNamespace(validate_workspace=boom))
    [c] = doctor.check_secret_scrub("/ws", None)
    assert c.status == doctor.WARN and "no such workspace" in c.detail


def test_config_root_is_forwarded(monkeypatch):
    seen = {}
    def rec(ws, config_root=None):
        seen["args"] = (ws, config_root)
        return []
    monkeypatch.setitem(sys.modules, "shared.scaffold.validate",
                        SimpleNamespace(validate_workspace=rec))
    doctor.check_secret_scrub("/ws", "/elsewhere/.jaato")
    assert seen["args"] == ("/ws", "/elsewhere/.jaato")
