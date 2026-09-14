"""``new client --profile`` — the form a real client uses.

Until this flag existed the only way to generate a client was an inline
``{model, provider}`` spec, whatever the workspace already had in it: the
template's comment said "swap for profile=" and the generator always wrote
the spec, and the only message an author saw when they supplied no flags was
``missing required --provider / --model`` — the inline answer, pointing away
from the one form that can carry plugins, a persona, GC, ceilings and a
completion schema.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from shared.scaffold import build


class _Args:
    """The subset of the ``new`` namespace the binding path reads."""

    def __init__(self, workspace, **kw):
        self.workspace = str(workspace)
        self.archetype = "client"
        self.provider = self.model = self.profile = self.set = None
        self.force = self.dry_run = False
        self.transport = "ipc"
        self.__dict__.update(kw)


@pytest.fixture()
def ws(tmp_path):
    """A workspace declaring one unscoped profile and one inside a set."""
    root = tmp_path / "ws"
    profiles = root / ".jaato" / "profiles"
    (profiles / "anthropic_sonnet").mkdir(parents=True)
    (profiles / "solo.yaml").write_text(textwrap.dedent("""
        name: solo
        description: unscoped
        plugins: []
        provider: echo
        model: echo-1
    """), encoding="utf-8")
    (profiles / "anthropic_sonnet" / "scoped.yaml").write_text(textwrap.dedent("""
        name: scoped
        description: inside a set
        plugins: []
        provider: echo
        model: echo-1
    """), encoding="utf-8")
    return root


def test_unscoped_profiles_are_enumerated(ws):
    assert build.workspace_profile_names(ws) == ["solo"]


def test_a_set_scoped_profile_needs_its_set(ws):
    assert build.workspace_profile_names(ws, "anthropic_sonnet") == [
        "scoped", "solo"]


def test_the_selector_comes_from_the_workspace_env(ws):
    # JAATO_PROFILE_SET in the workspace .env is what the generated client
    # runs under, so it is what --profile must resolve against.
    (ws / ".env").write_text("JAATO_PROFILE_SET=anthropic_sonnet\n",
                             encoding="utf-8")
    assert "scoped" in (build.workspace_profile_names(ws) or [])


def test_could_not_look_is_not_declares_none():
    # `[]` is a measured absence and can refuse a --profile; `None` is "I
    # could not look" and must not be read as one.
    assert build.workspace_profile_names(None) is None
    assert build.workspace_profile_names("/nonexistent/workspace") == []


def test_a_known_profile_resolves(ws):
    assert build._check_named_profile(_Args(ws), "client", "solo") is None


def test_an_unknown_profile_is_refused(ws, capsys):
    assert build._check_named_profile(_Args(ws), "client", "nope") == 2
    assert "no profile 'nope'" in capsys.readouterr().out


def test_a_set_scoped_profile_names_its_set(ws, capsys):
    # Found, but only under a set this workspace does not select.  Saying
    # "does not exist" about a file the author can see would send them
    # looking for a typo they did not make.
    assert build._check_named_profile(_Args(ws), "client", "scoped") == 2
    out = capsys.readouterr().out
    assert "anthropic_sonnet" in out and "--set" in out


def test_two_bindings_are_refused(ws, capsys):
    code, _, _ = build._resolve_client_binding(
        _Args(ws, profile="solo", provider="echo", model="m"), "client", "ipc")
    assert code == 2
    assert "two bindings" in capsys.readouterr().out


def test_in_process_has_no_profile_to_resolve(ws, capsys):
    code, _, _ = build._resolve_client_binding(
        _Args(ws, profile="solo", transport="in_process"), "client", "in_process")
    assert code == 2
    assert "in_process" in capsys.readouterr().out


def test_no_flags_offers_profile_when_the_workspace_has_them(ws, capsys):
    code, _, _ = build._resolve_client_binding(_Args(ws), "client", "ipc")
    assert code == 2
    out = capsys.readouterr().out
    assert "--profile" in out and "solo" in out


def test_no_flags_on_an_empty_workspace_is_unchanged(tmp_path, capsys):
    code, _, _ = build._resolve_client_binding(
        _Args(tmp_path), "client", "ipc")
    assert code == 2
    out = capsys.readouterr().out
    assert "missing required --provider / --model" in out
    assert "--profile" not in out


def test_the_emitted_body_names_the_profile():
    subs = build._binding_substitutions("client", None, None, "solo")
    assert subs["__SESSION_BINDING__"] == '"solo"'
    # A profile carries the binding; a MODEL/PROVIDER constant beside it
    # would be a second one that decides nothing.
    assert subs["__MODEL_CONSTANTS__"] == ""


def test_an_unbound_generate_still_emits_the_inline_spec():
    subs = build._binding_substitutions("client", "echo", "echo-1")
    assert subs["__SESSION_BINDING__"] == '{"model": MODEL, "provider": PROVIDER}'


def test_a_profile_client_never_rewrites_the_workspace_env(tmp_path):
    # That file is where JAATO_PROFILE_SET lives, and this archetype's
    # template carries a provider/model pair the profile supersedes.
    envf = tmp_path / ".env"
    envf.write_text("JAATO_PROFILE_SET=anthropic_sonnet\n", encoding="utf-8")
    assert build._should_write_client_env(
        _Args(tmp_path, profile="solo", force=True), envf) is False
    assert build._should_write_client_env(
        _Args(tmp_path, force=True), envf) is True
    assert build._should_write_client_env(
        _Args(tmp_path), Path(tmp_path / "absent.env")) is True
