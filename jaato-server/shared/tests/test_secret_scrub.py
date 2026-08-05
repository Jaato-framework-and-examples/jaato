"""Secrets-broker env scrub: the primitive + its cli/run_command chokepoint.

Covers feature #10's scrub half — declared secret env vars are removed from the
environment handed to a model-driven subprocess, without touching the runner's
own os.environ.
"""

from shared.secret_scrub import matches_secret, scrub_env
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
    monkeypatch.setenv("SECRET_X", "topsecret")
    r = run_command(_echo_secret_cmd())
    assert "topsecret" in r.stdout


def test_run_command_glob_scrub(monkeypatch):
    monkeypatch.setenv("MYAPP_TOKEN", "tok")
    r = run_command('sh -c \'echo "${MYAPP_TOKEN:-EMPTY}"\'',
                    scrub_env=["*_TOKEN"])
    assert "tok" not in r.stdout and "EMPTY" in r.stdout
