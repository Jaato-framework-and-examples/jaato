"""A profile's provider binds the way its model does — at EVERY producer.

#822 reported one symptom: a profile whose ``model_tiers`` fully declared
model AND provider, and which therefore dropped both top-level keys exactly as
the profile loader's own warning advises, reached the runner with an empty
``provider_name`` and was refused.  The fix gave ``build_session_envelope`` the
same tier-aware binder ``model_name`` has had since #574.

That was one of THREE producers of this binding, and fixing the one the issue
named left the worst of the three untouched:

  * ``server/runner_spawn.py``       — the runner envelope (#822's symptom)
  * ``server/session_manager.py``    — the isolated-subagent envelope
  * ``server/core.py``               — the daemon-side JaatoRuntime

The third is the dangerous one, because it is the one that does NOT fail
loudly.  The two envelope builders hand an empty ``provider_name`` to the
runner, which refuses it by name.  ``core.py`` instead falls through to
``JAATO_PROVIDER`` from the workspace ``.env`` — so a profile whose initial
tier declares ``openai/gpt-audio-mini`` on ``openrouter`` builds a
``JaatoRuntime`` bound to whatever the ``.env`` happened to name, and connects
the tier's model to a different vendor's provider with nothing said.

It was also LIVE rather than latent: ``SessionManager._construct_and_initialize_server``
is the one construction funnel for both session CREATE and disk-RESTORE, it
builds the ``JaatoServer`` with ``profile=envelope.profile`` and
``provider=None`` (so the env supplies the fallback), and calls
``server.initialize()`` — alongside, not instead of, the runner envelope.

The comment above the ``core.py`` site already said "Use the SAME binder the
gate above used".  The model half took that lesson and the provider half did
not, which is why the guard at the bottom of this file checks the RULE across
every producer rather than the three sites someone happened to look at.
"""

from __future__ import annotations

import ast
import pathlib
from unittest.mock import patch

import pytest

from server.core import JaatoServer
from shared.plugins.subagent.config import SubagentProfile


# --------------------------------------------------------------- fixtures

def _tiers_only_profile(**overrides) -> SubagentProfile:
    """The reporter's profile: tiers declare model AND provider, and the
    top-level keys are absent."""
    base = dict(
        name="speaker",
        description="tiers declare model and provider; no top-level keys",
        model_tiers={
            "executor": {"model": "openai/gpt-audio-mini",
                         "provider": "openrouter"},
            "initial": "executor",
            "fallback": "executor",
        },
    )
    base.update(overrides)
    return SubagentProfile(**base)


def _runtime_provider(tmp_path, profile, env_provider="anthropic"):
    """Run ``JaatoServer.initialize`` with the runtime mocked and report the
    ``provider_name`` it would have connected.

    The provider connection is the only part of init this cares about, and it
    is the part that must not reach a real vendor from a unit test.
    """
    env = tmp_path / ".env"
    env.write_text(f"JAATO_PROVIDER={env_provider}\n", encoding="utf-8")
    server = JaatoServer(
        workspace_path=str(tmp_path),
        session_id="s",
        profile=profile,
        env_file=str(env),
    )
    # Pushed after construction, exactly as the real funnel does
    # (``_construct_and_initialize_server`` sets it before initialize so
    # plugins see it on their first set_config_root notification).
    server.config_root = str(tmp_path / ".jaato")
    with patch("server.core.JaatoRuntime") as runtime_cls:
        assert server.initialize() is True
    assert runtime_cls.call_args is not None, "initialize built no runtime"
    return runtime_cls.call_args.kwargs["provider_name"], server


# ------------------------------------------------- core.py (the silent one)

class TestDaemonRuntimeBinding:

    def test_the_provider_comes_from_the_initial_tier(self, tmp_path):
        """The defect, as its symptom: an OpenRouter model was handed to the
        Anthropic provider because ``.env`` named one and the tier was never
        consulted."""
        provider, _ = _runtime_provider(tmp_path, _tiers_only_profile())
        assert provider == "openrouter", (
            "the daemon-side runtime bound the .env provider while the tier "
            "supplied the model — the tier's model handed to a different "
            "vendor, silently"
        )

    def test_the_model_and_the_provider_agree(self, tmp_path):
        """The property that actually matters, stated as itself: whatever the
        two halves resolve to, they come from the SAME tier."""
        profile = _tiers_only_profile()
        provider, server = _runtime_provider(tmp_path, profile)
        assert server._model_name == "openai/gpt-audio-mini"
        assert provider == "openrouter"

    def test_a_top_level_provider_still_wins_over_the_tier(self, tmp_path):
        """Precedence is unchanged.  The binder consults the initial tier only
        when the flat key is absent, so an explicit ``provider:`` outranks it
        exactly as it did before."""
        provider, _ = _runtime_provider(
            tmp_path, _tiers_only_profile(provider="zhipuai"))
        assert provider == "zhipuai"

    def test_the_profile_still_wins_over_the_session_env(self, tmp_path):
        """...and both profile routes still outrank ``JAATO_PROVIDER``, which
        is the fallback for a profile that binds nothing."""
        provider, _ = _runtime_provider(
            tmp_path, _tiers_only_profile(), env_provider="google_genai")
        assert provider == "openrouter"

    def test_a_profile_binding_no_provider_falls_back_to_the_env(self, tmp_path):
        """A tier may legitimately omit ``provider`` — it then means "the
        session's main provider", which is what the env names.  The fix must
        not turn that into a hard bind on nothing."""
        profile = SubagentProfile(
            name="speaker", description="tier names a model only",
            model_tiers={"executor": "some-model", "initial": "executor"},
        )
        provider, _ = _runtime_provider(tmp_path, profile,
                                        env_provider="anthropic")
        assert provider == "anthropic"


# ------------------------------------------- the rule, at every producer

_SERVER = pathlib.Path(__file__).resolve().parents[1]

#: Where the framework turns a profile into a (model, provider) pair.
#:
#: Enumerated so a FOURTH producer cannot appear unnoticed: the guard below
#: fails if this list stops matching the tree, which is the whole lesson of
#: #822's follow-up — the issue named one site, the defect was in three, and
#: the one nobody looked at was the one that failed silently.
_PRODUCER_FILES = {
    _SERVER / "core.py",
    _SERVER / "runner_spawn.py",
    _SERVER / "session_manager.py",
}


def _binding_functions(path: pathlib.Path):
    """Functions in *path* that BIND a model from a profile.

    A binder ASSIGNS from ``bound_model_for_profile``; a predicate merely
    asks.  ``core._profile_binds_a_model`` is the latter — it returns
    ``... is not None`` and has no provider half to be missing — so the rule
    below applies to assignment sites only, which is the difference between
    "this code decides what to connect to" and "this code asks a question".
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assigned = {
            call
            for stmt in ast.walk(node) if isinstance(stmt, ast.Assign)
            for call in ast.walk(stmt.value)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id == "bound_model_for_profile"
        }
        if assigned:
            called = {c.func.id for c in ast.walk(node)
                      if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
            out.append((node.name, called))
    return out


def test_the_producer_list_matches_the_tree():
    """Guard the guard.  If a new file starts binding a profile's model, this
    fails until someone decides whether it needs the provider half too —
    rather than the guard quietly covering three of four sites."""
    found = {
        p for p in _SERVER.rglob("*.py")
        if "tests" not in p.parts and "bound_model_for_profile("
        in p.read_text(encoding="utf-8")
    }
    assert found == _PRODUCER_FILES, (
        f"the set of files binding a profile's model changed: "
        f"unexpected {sorted(str(p) for p in found - _PRODUCER_FILES)}, "
        f"missing {sorted(str(p) for p in _PRODUCER_FILES - found)}"
    )


@pytest.mark.parametrize("path", sorted(_PRODUCER_FILES), ids=lambda p: p.name)
def test_every_binder_binds_BOTH_halves(path):
    """The rule: a site that binds the model from a profile binds the provider
    from it too, by the same route.

    Stated as a rule rather than as three assertions about three files,
    because the failure this exists to prevent is one half of a pair being
    updated — which is exactly what happened when #822 fixed two producers and
    left ``core.py`` pairing a tier-aware model binder with a flat-key
    provider read.
    """
    binders = _binding_functions(path)
    assert binders, f"{path.name} no longer binds a model; update _PRODUCER_FILES"
    for name, called in binders:
        assert "bound_provider_for_profile" in called, (
            f"{path.name}::{name} binds the model through the tier-aware "
            "binder and the provider some other way; the two halves must "
            "come from the same profile by the same route (jaato #822)"
        )
