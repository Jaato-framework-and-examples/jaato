"""Unit tests for ``PromptApiBridge`` reuse_page / helper self-heal logic.

These exercise the bridge's page-acquisition and teardown decisions in
isolation with a minimal fake CDP connection — no real browser, mirroring
the ``FakeBridge`` philosophy at the layer below it.
"""

from typing import Any, Dict, List, Optional

from shared.plugins.model_provider.chrome_ai.bridge import PromptApiBridge


class _FakeConn:
    """Minimal stand-in for ``cdp.CDPConnection`` recording sent methods."""

    def __init__(self, targets: Optional[List[Dict[str, Any]]] = None) -> None:
        self.is_open = True
        self._targets = targets or []
        self.sent: List[str] = []
        self.closed = False

    def send(self, method: str, params: Optional[Dict[str, Any]] = None,
             **kwargs: Any) -> Dict[str, Any]:
        self.sent.append(method)
        if method == "Target.getTargets":
            return {"targetInfos": self._targets}
        if method == "Target.createTarget":
            return {"targetId": "created-id"}
        if method == "Target.attachToTarget":
            return {"sessionId": "sess"}
        return {}

    def close(self) -> None:
        self.closed = True


def _bridge(**kw: Any) -> PromptApiBridge:
    return PromptApiBridge(cdp_url="http://localhost:9222",
                           page_url="https://example.com/", **kw)


# ----------------------------------------------------------- _find_existing_page

def test_find_existing_page_matches_exact_url():
    b = _bridge(reuse_page=True)
    b._conn = _FakeConn(targets=[
        {"type": "page", "url": "https://other.example/", "targetId": "t1"},
        {"type": "page", "url": "https://example.com/", "targetId": "t2"},
    ])
    assert b._find_existing_page() == "t2"


def test_find_existing_page_ignores_non_page_and_mismatch():
    b = _bridge(reuse_page=True)
    b._conn = _FakeConn(targets=[
        {"type": "background_page", "url": "https://example.com/", "targetId": "bg"},
        {"type": "page", "url": "https://example.com/other", "targetId": "t1"},
    ])
    assert b._find_existing_page() is None


# ---------------------------------------------------------------- close ownership

def test_close_leaves_reused_tab_open():
    """A tab we attached to (not created) is never closed on teardown."""
    b = _bridge(reuse_page=True)
    conn = _FakeConn()
    b._conn = conn
    b._target_id = "t2"
    b._owns_target = False
    b.close()
    assert "Target.closeTarget" not in conn.sent
    assert conn.closed  # the CDP link itself is still dropped


def test_close_closes_created_tab():
    """A tab we created is closed on teardown."""
    b = _bridge()
    conn = _FakeConn()
    b._conn = conn
    b._target_id = "created-id"
    b._owns_target = True
    b.close()
    assert "Target.closeTarget" in conn.sent


# --------------------------------------------------------- per-turn helper reinstall

def test_start_turn_reinstalls_helper_before_running(monkeypatch):
    """start_turn re-asserts the page helper (self-heal after navigation)."""
    b = _bridge(reuse_page=True)
    order: List[Any] = []
    monkeypatch.setattr(b, "_wait_page_ready", lambda: order.append("ready"))
    monkeypatch.setattr(b, "_ensure_helper", lambda: order.append("helper"))
    monkeypatch.setattr(
        b, "_call_helper",
        lambda fn, run_id, payload=None: order.append(("call", fn)))
    run_id = b.start_turn({"prompt": "hi"})
    assert run_id
    assert order == ["ready", "helper", ("call", "run")]
