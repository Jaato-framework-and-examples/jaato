"""``JaatoSession.reload_provider`` -- the provider is rebuilt from the CURRENT environment.

The provider resolves its credential once, in ``initialize()``; this is the
seam ``session.reload_env`` uses to make a session look again.  Pinned:

- the old provider is forgotten (and shut down), the tier cache cleared,
  and a NEW instance is created through ``_ensure_provider`` from the
  binding the session is currently on -- eagerly, so a credential that
  does not resolve fails here;
- the answer names provider, model and the provider's ``auth_info``;
- a running turn is refused with nothing changed;
- an unconfigured session has nothing to rebuild from.
"""
from __future__ import annotations

import threading

import pytest

from shared.jaato_session import JaatoSession


class _Provider:
    def __init__(self, tag):
        self.tag = tag
        self.shut = False

    def get_auth_info(self):
        return f"API key ({self.tag})"

    def shutdown(self):
        self.shut = True

    def get_context_limit(self):
        return 100_000


class _Runtime:
    provider_name = "zhipuai"

    def __init__(self):
        self.created = []

    def create_provider(self, model, provider_name=None, skip_model_test=False,
                        plugin_configs=None, session_id=None):
        self.created.append((model, provider_name, skip_model_test, plugin_configs))
        return _Provider(f"build-{len(self.created)}")


def _session(*, provider=None, running=False, configured=True) -> JaatoSession:
    s = JaatoSession.__new__(JaatoSession)
    s._runtime = _Runtime()
    s._provider = provider
    s._provider_cache = {"zhipuai": provider} if provider else {}
    s._provider_init_lock = threading.Lock()
    s._provider_lazy_pending = None
    s._tier_provider_base = (
        {"skip_model_test": True, "plugin_configs": {"zhipuai": {"k": 1}}}
        if configured else None)
    s._model_name = "glm-5"
    s._provider_name_override = None
    s._active_provider_name = "zhipuai" if provider else None
    s._is_running = running
    s._daemon_session_id = "s1"
    s._agent_type, s._agent_name, s._agent_id = "main", None, "main"
    s._instruction_budget = None
    s._trace = lambda msg: None
    # the two post-create hooks _ensure_provider runs; irrelevant here
    s._validate_modality_tier_capabilities = lambda: None
    s._request_active_tier_output_modalities = lambda: None
    return s


def test_rebuilds_from_the_current_binding_and_names_the_source():
    old = _Provider("stale")
    s = _session(provider=old)

    info = s.reload_provider()

    assert old.shut is True
    assert s._provider is not old and s._provider.tag == "build-1"
    assert s._runtime.created == [("glm-5", "zhipuai", True, {"zhipuai": {"k": 1}})]
    assert info == {"provider": "zhipuai", "model": "glm-5", "auth_info": "API key (build-1)"}
    assert s._provider_cache == {"zhipuai": s._provider}


def test_works_before_the_first_lazy_creation():
    """A session that never took a turn has no provider yet; reload builds it."""
    s = _session(provider=None)
    info = s.reload_provider()
    assert s._provider is not None and info["provider"] == "zhipuai"


def test_a_running_turn_is_refused_with_nothing_changed():
    old = _Provider("live")
    s = _session(provider=old, running=True)
    with pytest.raises(RuntimeError, match="turn is running"):
        s.reload_provider()
    assert s._provider is old and old.shut is False


def test_an_unconfigured_session_has_nothing_to_rebuild():
    s = _session(configured=False)
    with pytest.raises(RuntimeError, match="not configured"):
        s.reload_provider()


def test_a_credential_that_does_not_resolve_fails_in_the_reload():
    s = _session(provider=_Provider("stale"))

    def _boom(*a, **k):
        raise RuntimeError("No Zhipu AI API key found")
    s._runtime.create_provider = _boom
    with pytest.raises(RuntimeError, match="No Zhipu AI API key"):
        s.reload_provider()
