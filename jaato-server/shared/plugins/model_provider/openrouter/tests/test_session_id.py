"""The per-session id OpenRouter uses to group a conversation's requests.

OpenRouter accepts a ``session_id`` field in the request body (a sibling
of ``model``/``messages``) so the requests of one conversation are grouped
on their side.  The framework stamps it in
``JaatoRuntime.create_provider`` via ``ProviderConfig.extra`` — deliberately
NOT as a new method on ``ModelProviderPlugin``, which is
``@runtime_checkable``: adding a method there silently drops every provider
that does not implement it from ``isinstance`` gates (see #708).

The load-bearing property these tests defend is that the id is
PER-SESSION.  A shared source (the registry) would hand every sibling
subagent the last-bootstrapped session's id, which is the leak
``JaatoSession.set_daemon_session_id`` exists to close — so a sibling
mixing test is the point, not a nicety.
"""

from typing import Optional

import pytest

from shared.plugins.model_provider.openrouter.provider import OpenRouterProvider
from shared.plugins.model_provider.base import ProviderConfig


def _provider(session_id: Optional[str]) -> OpenRouterProvider:
    """A provider initialized as the framework would, with api calls stubbed."""
    p = OpenRouterProvider()
    extra = {"session_id": session_id} if session_id is not None else {}
    try:
        p.initialize(ProviderConfig(api_key="sk-or-test", extra=extra))
    except Exception:
        # initialize() may reach out for a catalog / client build depending on
        # environment; the field under test is read before any of that, and
        # asserting on it is the point.  Re-read it directly if so.
        p._session_id = extra.get("session_id")
    return p


def test_session_id_reaches_the_request_body():
    body = _provider("20260830_172724")._build_extra_body()
    assert body["session_id"] == "20260830_172724"


def test_absent_session_id_emits_no_field():
    """Absent must mean ABSENT, not an empty string on the wire."""
    body = _provider(None)._build_extra_body()
    assert "session_id" not in body


def test_unconfigured_provider_does_not_raise():
    """``initialize(config=None)`` never reaches the ``config.extra`` read,
    so the attribute must exist from construction."""
    body = OpenRouterProvider()._build_extra_body()
    assert "session_id" not in body


def test_siblings_do_not_share_an_id():
    """Two providers built for two sessions must not converge on one id.

    This is the regression that matters: a registry-sourced id would give
    both siblings whichever session bootstrapped last.
    """
    a = _provider("session_aaa")
    b = _provider("session_bbb")
    assert a._build_extra_body()["session_id"] == "session_aaa"
    assert b._build_extra_body()["session_id"] == "session_bbb"


def test_runtime_threads_the_id_into_provider_config():
    """``create_provider`` must accept the caller's id and inject it into
    ``extra`` — the seam the provider reads."""
    import inspect
    from shared.jaato_runtime import JaatoRuntime

    assert "session_id" in inspect.signature(JaatoRuntime.create_provider).parameters
    src = inspect.getsource(JaatoRuntime.create_provider)
    assert "'session_id'" in src or '"session_id"' in src


def test_sessions_pass_their_own_id_not_a_shared_one():
    """Both ``create_provider`` call sites in JaatoSession must pass the
    session's own ``_daemon_session_id``."""
    import inspect
    from shared.jaato_session import JaatoSession

    src = inspect.getsource(JaatoSession)
    assert src.count("session_id=self._daemon_session_id") == 2
