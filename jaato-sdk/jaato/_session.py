"""Transport-agnostic ``session(mode=...)`` dispatcher — the public facade entry.

Lives in **jaato-sdk** so ``import jaato`` + ``jaato.session(mode="ipc"|"ws")``
work with only the SDK installed (the daemon may be on another host — a thin
client needs no server runtime). The ``in_process`` mode is the sole exception:
it runs the agent loop in your process, so it lazily imports the embedded runtime
from **jaato-server** (``jaato_embedded``) and fails loud with an install hint if
that package isn't present.

The ipc / ws branches only ever import from ``jaato_sdk`` (lazily, per branch), so
this module never pulls ``shared`` at import time. See
``docs/design/in-process-facade.md``.
"""

from __future__ import annotations

from typing import Any, Dict


def _bundle_inline_profile(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Bundle separate ``model`` / ``provider`` / ``plugins`` /
    ``plugin_configs`` kwargs into the inline-spec ``profile`` dict that both
    clients accept (unless a ``profile`` is already given). Other kwargs (agent,
    connection knobs) pass through untouched."""
    kwargs = dict(kwargs)
    if "profile" not in kwargs:
        spec = {
            key: kwargs.pop(key)
            for key in ("model", "provider", "plugins", "plugin_configs")
            if key in kwargs
        }
        if spec:
            kwargs["profile"] = spec
    return kwargs


def session(mode: str = "ipc", *, recovery: bool = False, **kwargs: Any) -> Any:
    """Transport-agnostic session entry — the facade picks the client by ``mode``.

    Three transports, one spec, one facade (``s.ask`` / ``.complete`` /
    ``.stream``):

    * ``mode="in_process"`` — the embedded ``InProcessClient`` (no daemon,
      no socket; the agent runs in your process). Requires **jaato-server**
      (``pip install jaato-server``); the ipc / ws modes need only jaato-sdk.
    * ``mode="ipc"`` — a *local* daemon over a Unix socket via ``IPCClient``.
    * ``mode="ws"`` — a *remote* daemon over ``ws://`` / ``wss://`` via
      ``WSClient`` (pass ``url=`` and optional ``token=``).

    ``recovery=True`` (daemon modes only) swaps in the **auto-reconnect** client
    — ``IPCRecoveryClient`` (``ipc``) / ``WSRecoveryClient`` (``ws``) — so the
    session survives daemon restarts (exponential backoff + session
    reattachment; pass ``on_status_change=`` for the reconnection callback).
    ``recovery=True`` with ``mode="in_process"`` is an error: the embedded
    runtime has no daemon to reconnect to.

    All accept the same session spec — pass ``model`` / ``provider`` /
    ``plugins`` / ``plugin_configs`` as separate kwargs (bundled into the
    inline-spec ``profile``) or a ``profile`` dict directly — so one example
    runs every way with ``mode`` the only variable::

        async with jaato.session(mode=m, model=..., provider=..., plugins=[],
                                 plugin_configs={...}) as s:
            print(await s.ask("Hi"))

        # remote daemon:
        async with jaato.session(mode="ws", url="wss://host:8080", token="...",
                                 profile={...}) as s:
            print(await s.ask("Hi"))

    ``env_file`` (the session ``.env``) applies to the embedded mode — the
    embedded runtime reads the same env the daemon loads from it. Knobs that
    apply to only one transport (``socket_path`` / ``auto_start`` for IPC;
    ``url`` / ``token`` for WS) are optional and ignored by the others — per-mode
    connection config, not a code clone. See
    ``docs/design/in-process-facade.md``.
    """
    spec_kwargs = _bundle_inline_profile(kwargs)
    if mode == "in_process":
        if recovery:
            raise ValueError(
                "session(recovery=True) needs a daemon transport (mode='ipc' or "
                "mode='ws'); the in-process runtime has no daemon to reconnect to"
            )
        # The embedded runtime lives in jaato-server (the agent loop runs in this
        # process). Import it lazily + fail loud so the ipc/ws client paths stay
        # sdk-only and a missing server surfaces an actionable install hint
        # instead of an opaque ModuleNotFoundError.
        try:
            from jaato_embedded import InProcessClient
        except ModuleNotFoundError as exc:
            raise ImportError(
                "session(mode='in_process') needs the embedded runtime, which "
                "ships with jaato-server. Install it (pip install jaato-server) "
                "or use mode='ipc'/'ws' to connect to a running daemon."
            ) from None
        return InProcessClient.session(**spec_kwargs)
    if mode == "ipc":
        if recovery:
            from jaato_sdk import IPCRecoveryClient
            return IPCRecoveryClient.session(**spec_kwargs)
        from jaato_sdk import IPCClient
        return IPCClient.session(**spec_kwargs)
    if mode == "ws":
        url = spec_kwargs.pop("url", None)
        if not url:
            raise ValueError("session(mode='ws') requires a url= (ws:// or wss://)")
        if recovery:
            from jaato_sdk import WSRecoveryClient
            return WSRecoveryClient.session(url, **spec_kwargs)
        from jaato_sdk import WSClient
        return WSClient.session(url, **spec_kwargs)
    raise ValueError(
        f"unknown session mode {mode!r}; expected 'ipc', 'in_process', or 'ws'"
    )
