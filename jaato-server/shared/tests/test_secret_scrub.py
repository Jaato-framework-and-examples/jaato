"""Secrets-broker env scrub: the primitive + its cli/run_command chokepoint.

Covers feature #10's scrub half — declared secret env vars are removed from the
environment handed to a model-driven subprocess, without touching the runner's
own os.environ — and the #863 grammar layered on it: ``default`` / ``none`` /
``!EXEMPT`` entries, absence meaning the framework set, and the fail-closed
resolver the plugins go through.
"""

import logging

import pytest

from shared.secret_scrub import (
    DEFAULT_SECRET_ENV_PATTERNS,
    SCRUB_SURFACES,
    is_scrub_disabled,
    matches_secret,
    normalize_scrub_patterns,
    resolve_scrub_patterns,
    scrub_env,
)
from shared.subprocess_runner import run_command


# ---- primitive -----------------------------------------------------------

def test_matches_secret_case_insensitive_globs():
    pats = ["*_API_KEY", "*_TOKEN", "ANTHROPIC_AUTH_TOKEN"]
    assert matches_secret("OPENAI_API_KEY", pats)
    assert matches_secret("openai_api_key", pats)      # case-insensitive
    assert matches_secret("GITHUB_TOKEN", pats)
    assert matches_secret("ANTHROPIC_AUTH_TOKEN", pats)
    assert not matches_secret("PATH", pats)
    assert not matches_secret("HOME", pats)


def test_scrub_env_removes_only_matches_and_copies():
    env = {"PATH": "/bin", "GITHUB_TOKEN": "ghp_x", "OPENAI_API_KEY": "sk-x"}
    out = scrub_env(env, ["*_TOKEN", "*_API_KEY"])
    assert out == {"PATH": "/bin"}
    # input not mutated
    assert "GITHUB_TOKEN" in env


def test_scrub_env_empty_patterns_is_noop_copy():
    env = {"GITHUB_TOKEN": "x"}
    out = scrub_env(env, [])
    assert out == env and out is not env


# ---- exemptions (#863): the developer-desktop answer ----------------------

def test_exemption_wins_over_a_matching_scrub_glob():
    pats = ["*_TOKEN", "!GH_TOKEN"]
    assert matches_secret("GITHUB_TOKEN", pats)
    assert not matches_secret("GH_TOKEN", pats)
    assert not matches_secret("gh_token", pats)          # case-insensitive too


def test_exemption_may_be_a_glob():
    pats = ["default", "!AWS_*"]
    pats = normalize_scrub_patterns(pats)
    assert not matches_secret("AWS_SECRET_ACCESS_KEY", pats)
    assert not matches_secret("AWS_SESSION_TOKEN", pats)
    assert matches_secret("OPENAI_API_KEY", pats)


def test_scrub_env_honours_exemptions():
    env = {"GH_TOKEN": "keep", "GITHUB_TOKEN": "drop", "PATH": "/bin"}
    assert scrub_env(env, ["*_TOKEN", "!GH_TOKEN"]) == {
        "GH_TOKEN": "keep", "PATH": "/bin",
    }


def test_only_exemptions_scrubs_nothing_at_the_primitive_and_reads_as_disabled():
    # The PRIMITIVE is policy-free; the grammar layer above refuses this
    # shape (see test_ambiguous_opt_out_spellings_are_rejected).
    assert scrub_env({"X_TOKEN": "v"}, ["!GH_TOKEN"]) == {"X_TOKEN": "v"}
    assert is_scrub_disabled(["!GH_TOKEN"])
    assert is_scrub_disabled(())
    assert not is_scrub_disabled(["*_TOKEN"])


# ---- the value grammar (#863) --------------------------------------------

def test_absent_means_the_framework_set():
    # The flipped default: a caller that never mentions the knob scrubs.
    assert normalize_scrub_patterns(None) == tuple(DEFAULT_SECRET_ENV_PATTERNS)


@pytest.mark.parametrize("value", ["default", "DEFAULT", " default "])
def test_default_shorthand(value):
    assert normalize_scrub_patterns(value) == tuple(DEFAULT_SECRET_ENV_PATTERNS)


@pytest.mark.parametrize("value", ["none", "NONE", " none "])
def test_none_is_the_explicit_opt_out(value):
    assert normalize_scrub_patterns(value) == ()


@pytest.mark.parametrize("value", [[], (), "", "  ", ["!GH_TOKEN"], ["!A_*", "!B"]])
def test_ambiguous_opt_out_spellings_are_rejected(value):
    # ``none`` is the ONLY way to disable.  ``[]`` reads as "the minimal set"
    # elsewhere in this codebase (``plugins: []``), and an exemption-only
    # list is ``[default, '!X']`` with the ``default`` forgotten — either
    # would otherwise select the leaky posture the flip exists to remove.
    with pytest.raises(ValueError, match="ambiguous"):
        normalize_scrub_patterns(value)


@pytest.mark.parametrize("value", [True, False])
def test_booleans_are_outside_the_grammar(value):
    with pytest.raises(ValueError):
        normalize_scrub_patterns(value)


def test_lone_string_is_one_pattern_not_characters():
    # The YAML mistake ``scrub_secret_env: "*_TOKEN"`` must not fail OPEN.
    assert normalize_scrub_patterns("*_TOKEN") == ("*_TOKEN",)


def test_list_entry_default_expands_in_place_and_exemptions_pass_through():
    out = normalize_scrub_patterns(["MY_SECRET", "default", "!GH_TOKEN"])
    assert out[0] == "MY_SECRET"
    assert out[1:-1] == tuple(DEFAULT_SECRET_ENV_PATTERNS)
    assert out[-1] == "!GH_TOKEN"


def test_list_entries_are_coerced_to_str_and_blanks_dropped():
    assert normalize_scrub_patterns(["*_TOKEN", 123, "  "]) == ("*_TOKEN", "123")


@pytest.mark.parametrize("value", [5, 1.5, {"patterns": ["x"]}, ["ok", None], [{}]])
def test_unsupported_shapes_raise(value):
    with pytest.raises(ValueError):
        normalize_scrub_patterns(value)


def test_surfaces_are_the_three_subprocess_plugins():
    assert SCRUB_SURFACES == ("cli", "interactive_shell", "mcp")


# ---- the plugin-side resolver: fail closed, announce the opt-out ----------

def test_resolve_absent_is_the_default_and_silent(caplog):
    with caplog.at_level(logging.WARNING, logger="shared.secret_scrub"):
        out = resolve_scrub_patterns(None, surface="cli")
    assert out == tuple(DEFAULT_SECRET_ENV_PATTERNS)
    assert caplog.records == []


def test_resolve_none_is_announced_at_warning_naming_the_surface(caplog):
    with caplog.at_level(logging.WARNING, logger="shared.secret_scrub"):
        out = resolve_scrub_patterns("none", surface="mcp")
    assert out == ()
    [rec] = caplog.records
    assert rec.levelno == logging.WARNING
    assert "mcp" in rec.getMessage() and "DISABLED" in rec.getMessage()


def test_resolve_malformed_fails_closed_with_an_error(caplog):
    with caplog.at_level(logging.ERROR, logger="shared.secret_scrub"):
        out = resolve_scrub_patterns({"bad": "shape"}, surface="interactive_shell")
    assert out == tuple(DEFAULT_SECRET_ENV_PATTERNS)
    [rec] = caplog.records
    assert rec.levelno == logging.ERROR
    assert "interactive_shell" in rec.getMessage()
    assert "fail closed" in rec.getMessage()


def test_resolve_empty_list_fails_closed_not_open(caplog):
    # The one shape most likely to be typed by mistake lands on the SAFE side.
    with caplog.at_level(logging.ERROR, logger="shared.secret_scrub"):
        out = resolve_scrub_patterns([], surface="cli")
    assert out == tuple(DEFAULT_SECRET_ENV_PATTERNS)
    assert any("ambiguous" in r.getMessage() for r in caplog.records)


def test_resolve_explicit_list_is_silent(caplog):
    with caplog.at_level(logging.WARNING, logger="shared.secret_scrub"):
        out = resolve_scrub_patterns(["default", "!GH_TOKEN"], surface="cli")
    assert out[-1] == "!GH_TOKEN" and len(out) == len(DEFAULT_SECRET_ENV_PATTERNS) + 1
    assert caplog.records == []


# ---- run_command chokepoint (the cli foreground path) --------------------

def _echo_secret_cmd():
    # Prints the value or 'EMPTY' if unset — works whether or not scrubbed.
    return 'sh -c \'echo "${SECRET_X:-EMPTY}"\''


def test_run_command_scrubs_declared_secret(monkeypatch):
    monkeypatch.setenv("SECRET_X", "topsecret")
    r = run_command(_echo_secret_cmd(), scrub_env=["SECRET_X"])
    assert r.returncode == 0
    assert "topsecret" not in r.stdout
    assert "EMPTY" in r.stdout


def test_run_command_without_scrub_still_sees_secret(monkeypatch):
    # The PRIMITIVE stays policy-free: the default lives in the plugins.
    monkeypatch.setenv("SECRET_X", "topsecret")
    r = run_command(_echo_secret_cmd())
    assert "topsecret" in r.stdout


def test_run_command_glob_scrub(monkeypatch):
    monkeypatch.setenv("MYAPP_TOKEN", "tok")
    r = run_command('sh -c \'echo "${MYAPP_TOKEN:-EMPTY}"\'',
                    scrub_env=["*_TOKEN"])
    assert "tok" not in r.stdout and "EMPTY" in r.stdout


def test_run_command_exemption_survives_the_glob(monkeypatch):
    monkeypatch.setenv("MYAPP_TOKEN", "tok")
    r = run_command('sh -c \'echo "${MYAPP_TOKEN:-EMPTY}"\'',
                    scrub_env=["*_TOKEN", "!MYAPP_TOKEN"])
    assert "tok" in r.stdout
