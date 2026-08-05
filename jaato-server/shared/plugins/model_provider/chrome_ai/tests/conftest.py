"""Shared fixtures for chrome_ai provider tests.

The mocking boundary is the browser bridge (:class:`FakeBridge` stands in
for ``PromptApiBridge``) — the same philosophy as ollama mocking httpx and
claude_cli mocking subprocess: everything Python-side runs for real, only
the browser I/O is scripted.
"""

import queue
from typing import Any, Dict, List, Optional

import pytest

from jaato_sdk.plugins.model_provider.types import Message, Role

from shared.plugins.model_provider.base import ProviderConfig
from shared.plugins.model_provider.chrome_ai.provider import ChromeAIProvider


class FakeBridge:
    """Scripted stand-in for ``PromptApiBridge``.

    Configure behaviour via attributes, then hand it to a provider through
    ``bridge_factory``.  Records every interaction for assertions:
    ``ctor_kwargs`` (what the provider resolved), ``payloads`` (per-turn
    run payloads), ``connect_calls``, ``download_calls``, ``aborted``.
    """

    def __init__(self) -> None:
        # -- scripting knobs --
        self.availability_value: str = "available"
        self.quota: Optional[int] = 9216
        #: Events (dicts) delivered for each successive start_turn call.
        self.turn_scripts: List[List[Dict[str, Any]]] = [
            [{"type": "done", "text": "hello",
              "usage": {"inputUsage": 42, "inputQuota": 9216}}],
        ]
        self.alive = True
        # -- recordings --
        self.ctor_kwargs: Dict[str, Any] = {}
        self.connect_calls = 0
        self.close_calls = 0
        self.download_calls = 0
        self.warmup_calls = 0
        self.warmup_should_raise = False
        self.payloads: List[Dict[str, Any]] = []
        self.aborted: List[str] = []
        self._queues: Dict[str, "queue.Queue[Dict[str, Any]]"] = {}
        self._turn_index = 0

    # --- factory hook -----------------------------------------------------

    def factory(self, **kwargs: Any) -> "FakeBridge":
        """Use as ``bridge_factory`` — records the resolved settings."""
        self.ctor_kwargs = kwargs
        return self

    # --- bridge surface ----------------------------------------------------

    @property
    def is_alive(self) -> bool:
        return self.alive

    def connect(self) -> None:
        self.connect_calls += 1
        self.alive = True

    def close(self) -> None:
        self.close_calls += 1
        self.alive = False

    def availability(self) -> str:
        return self.availability_value

    def probe_quota(self) -> Optional[int]:
        return self.quota

    def warmup(self, timeout: float = 60.0) -> None:
        self.warmup_calls += 1
        if self.warmup_should_raise:
            from shared.plugins.model_provider.chrome_ai.errors import (
                ChromeAIConnectionError,
            )
            raise ChromeAIConnectionError("warmup boom")

    def download(self, on_progress=None, timeout: float = 1800.0) -> None:
        self.download_calls += 1
        if on_progress:
            on_progress(0.5)
            on_progress(1.0)
        # Downloading flips the model available, like the real browser.
        self.availability_value = "available"

    def start_turn(self, payload: Dict[str, Any]) -> str:
        self.payloads.append(payload)
        run_id = f"run-{len(self.payloads)}"
        q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        script = (self.turn_scripts[self._turn_index]
                  if self._turn_index < len(self.turn_scripts) else [])
        self._turn_index += 1
        for event in script:
            q.put(dict(event))
        self._queues[run_id] = q
        return run_id

    def events(self, run_id: str) -> "queue.Queue[Dict[str, Any]]":
        return self._queues.setdefault(run_id, queue.Queue())

    def abort(self, run_id: str) -> None:
        self.aborted.append(run_id)

    def finish_turn(self, run_id: str) -> None:
        self._queues.pop(run_id, None)


@pytest.fixture(autouse=True)
def _scrub_env(monkeypatch):
    """Isolate tests from any ambient JAATO_CHROME_AI_* configuration."""
    for var in ("JAATO_CHROME_AI_BINARY", "JAATO_CHROME_AI_CDP_URL",
                "JAATO_CHROME_AI_USER_DATA_DIR", "JAATO_CHROME_AI_HEADLESS",
                "JAATO_CHROME_AI_CONTEXT_LENGTH", "JAATO_CHROME_AI_MODEL",
                "JAATO_CHROME_AI_PAGE_URL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fake_bridge() -> FakeBridge:
    return FakeBridge()


@pytest.fixture
def make_provider(fake_bridge):
    """Build an initialized (not yet connected) provider on the fake bridge."""
    def _make(extra: Optional[Dict[str, Any]] = None) -> ChromeAIProvider:
        provider = ChromeAIProvider(bridge_factory=fake_bridge.factory)
        provider.initialize(ProviderConfig(extra=extra or {}))
        return provider
    return _make


@pytest.fixture
def connected_provider(make_provider):
    """A provider connected to the (available) fake model.

    Uses attach mode (``cdp_url``) so no test depends on a Chrome binary
    existing on the test host.
    """
    provider = make_provider({"cdp_url": "http://localhost:9222"})
    provider.connect("gemini-nano")
    return provider


def user_message(text: str) -> Message:
    return Message.from_text(Role.USER, text)
