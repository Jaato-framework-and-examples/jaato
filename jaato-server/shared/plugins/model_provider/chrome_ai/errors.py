"""Exception types for the Chrome built-in AI (Prompt API) provider.

The hierarchy mirrors the other local providers (ollama, claude_cli):
one base class so callers can catch everything provider-specific, plus
targeted subclasses for the distinct failure surfaces — locating the
browser, driving it over CDP, and the on-device model's own availability
lifecycle (Gemini Nano is a separately-downloaded Chrome component that
may be absent, still downloading, or gated on unsupported hardware).
"""


class ChromeAIError(Exception):
    """Base class for all Chrome built-in AI provider errors."""


class ChromeAIBinaryNotFoundError(ChromeAIError):
    """No Google Chrome (or Edge) binary could be located.

    Raised by ``connect()`` when neither the ``binary`` knob /
    ``JAATO_CHROME_AI_BINARY`` env var nor the well-known install
    locations yield a launchable browser, and no ``cdp_url`` was given
    to attach to an already-running one.
    """


class ChromeAIConnectionError(ChromeAIError):
    """The browser process or its DevTools (CDP) connection failed.

    Covers launch failures, the DevTools WebSocket dropping mid-turn,
    and turn-level inactivity timeouts.  Classified as *transient infra*
    by ``ChromeAIProvider.classify_error`` so the framework's retry
    layer re-enters ``complete()``, where ``_ensure_connected()``
    relaunches the browser if the process died.
    """


class ChromeAIUnavailableError(ChromeAIError):
    """The Prompt API or the on-device model is not usable in this browser.

    Raised at ``connect()`` with actionable, state-specific instructions:
    the ``LanguageModel`` global is missing (browser too old / API not
    exposed to this context), availability is ``"unavailable"``
    (unsupported hardware or server-side gating), or the model is
    ``"downloadable"``/``"downloading"`` and ``auto_download`` is off.
    """


class ChromeAIResponseError(ChromeAIError):
    """The Prompt API rejected or failed a generation request.

    Non-transient: ``complete()`` converts it to
    ``TurnResult.from_exception`` instead of raising, per the provider
    contract (only rate-limit/overload style errors should propagate to
    the retry layer, and the local Prompt API has neither).
    """
