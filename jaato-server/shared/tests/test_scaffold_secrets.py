"""Tests for `jaato-scaffold new --secrets` — how a scaffolded profile
REFERENCES its provider credential.

Regression guard for the root cause behind public example repos shipping
profiles that only work with jaato-premium: the generator used to hardcode
`api_key: pass://jaato/<provider>/api-key`. The default is now env-var
interpolation (`${<PROVIDER_KEY_ENV>}`), which runs on a public checkout;
`pass://` (and other secret-URI schemes) are opt-in via `--secrets`.
"""

import argparse

from shared.scaffold import introspect, build


def _args(**kw):
    ns = argparse.Namespace()
    defaults = dict(archetype="profile-set", workspace=None, provider=None,
                    model=None, set=None, agents=None, force=True, json=False,
                    recoverable=False, secrets=None, secret_path=None)
    defaults.update(kw)
    for k, v in defaults.items():
        setattr(ns, k, v)
    return ns


# ------------------------------------------------------------- mode parsing

def test_resolve_secrets_mode_variants():
    assert build._resolve_secrets_mode(None) == ("env", None)     # default
    assert build._resolve_secrets_mode("env") == ("env", None)
    assert build._resolve_secrets_mode("none") == ("none", None)
    assert build._resolve_secrets_mode("pass") == ("uri", "pass")
    assert build._resolve_secrets_mode("pass://") == ("uri", "pass")
    assert build._resolve_secrets_mode("vault") == ("uri", "vault")


# ------------------------------------ env-var name from the AuthSource chain

def test_primary_key_env_var_is_read_from_provider_authsource():
    # Correct per-provider names — NOT a guessed JAATO_<PROVIDER>_API_KEY.
    a = introspect.resolve_provider("anthropic")
    assert build._primary_key_env_var(a, "anthropic") == "ANTHROPIC_API_KEY"
    d = introspect.resolve_provider("doubleword")
    assert build._primary_key_env_var(d, "doubleword") == "JAATO_DOUBLEWORD_API_KEY"


def test_primary_key_env_var_prefers_api_key_over_oauth_token():
    # anthropic's chain lists OAuth-token env vars BEFORE the api-key one;
    # the scaffold must pick the API key, not an OAuth token var.
    a = introspect.resolve_provider("anthropic")
    assert build._primary_key_env_var(a, "anthropic").endswith("API_KEY")


def test_primary_key_env_var_falls_back_to_convention_without_authsource():
    class _NoAuth:
        auth = ()
    assert build._primary_key_env_var(_NoAuth(), "acme") == "JAATO_ACME_API_KEY"


# ---------------------------------------------------------- emit: env (default)

def test_default_mode_emits_env_var_interpolation_not_pass(tmp_path):
    rc = build.run(_args(workspace=str(tmp_path), set="or_x",
                         provider="openrouter", model="m", agents="a"))
    assert rc == 0  # valid by construction
    prof = (tmp_path / ".jaato" / "profiles" / "or_x" / "a.yaml").read_text()
    assert 'api_key: "${JAATO_OPENROUTER_API_KEY}"' in prof
    assert "pass://" not in prof


def test_default_mode_surfaces_key_var_in_env_and_gitignores_it(tmp_path):
    build.run(_args(workspace=str(tmp_path), set="or_x",
                    provider="openrouter", model="m", agents="a"))
    env = (tmp_path / ".env").read_text()
    # active (uncommented) empty fill-in, not just a commented knob
    assert "\nJAATO_OPENROUTER_API_KEY=\n" in ("\n" + env + "\n")
    gi = (tmp_path / ".gitignore").read_text()
    assert ".env" in gi.split() and "!.env.example" in gi.split()


# --------------------------------------------------------------- emit: none

def test_none_mode_omits_api_key_line(tmp_path):
    rc = build.run(_args(workspace=str(tmp_path), set="n_x", provider="openrouter",
                         model="m", agents="a", secrets="none"))
    assert rc == 0
    prof = (tmp_path / ".jaato" / "profiles" / "n_x" / "a.yaml").read_text()
    assert "api_key" not in prof
    # provider still reads the env var directly → still surfaced + ignored
    assert "JAATO_OPENROUTER_API_KEY=" in (tmp_path / ".env").read_text()


# ---------------------------------------------------------------- emit: uri

def test_uri_mode_emits_secret_uri_and_no_env_key(tmp_path):
    rc = build.run(_args(workspace=str(tmp_path), set="p_x", provider="openrouter",
                         model="m", agents="a", secrets="pass"))
    assert rc == 0  # still VALID (the value is a well-formed api_key)
    prof = (tmp_path / ".jaato" / "profiles" / "p_x" / "a.yaml").read_text()
    assert "api_key: pass://jaato/openrouter/api-key" in prof
    # key lives in the secret store, not the workspace env: no ACTIVE key line
    # (the provider var may still appear as a commented `# ...=` knob).
    env_lines = [ln.strip() for ln in (tmp_path / ".env").read_text().splitlines()]
    assert not any(ln.startswith("JAATO_OPENROUTER_API_KEY=") for ln in env_lines)
    assert not (tmp_path / ".gitignore").exists()


def test_uri_mode_custom_secret_path(tmp_path):
    build.run(_args(workspace=str(tmp_path), set="c_x", provider="openrouter",
                    model="m", agents="a", secrets="pass",
                    secret_path="team/{provider}/key"))
    prof = (tmp_path / ".jaato" / "profiles" / "c_x" / "a.yaml").read_text()
    assert "api_key: pass://team/openrouter/key" in prof


# ------------------------------------------------------ persistence / inherit

def test_explicit_secrets_choice_is_recorded_and_inherited(tmp_path):
    # first set: explicit --secrets none is recorded
    build.run(_args(workspace=str(tmp_path), set="s1", provider="openrouter",
                    model="m", agents="a", secrets="none"))
    marker = tmp_path / ".jaato" / "scaffold.json"
    assert marker.exists() and '"secrets": "none"' in marker.read_text()
    # second set WITHOUT --secrets inherits the recorded mode
    build.run(_args(workspace=str(tmp_path), set="s2", provider="openrouter",
                    model="m", agents="b"))
    prof = (tmp_path / ".jaato" / "profiles" / "s2" / "b.yaml").read_text()
    assert "api_key" not in prof  # inherited 'none'


def test_default_is_not_persisted_when_secrets_omitted(tmp_path):
    build.run(_args(workspace=str(tmp_path), set="s1", provider="openrouter",
                    model="m", agents="a"))
    # no explicit choice → no marker written (default stays 'env' implicitly)
    assert not (tmp_path / ".jaato" / "scaffold.json").exists()
