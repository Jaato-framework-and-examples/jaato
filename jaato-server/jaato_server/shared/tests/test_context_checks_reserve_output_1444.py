"""Context checks reserve the output a request carries (#1444).

A MiniMax-M3 session (1M window) died on a provider 400 ("context window
exceeds limit (2013)") while jaato's readout said 91.9% used and 80.9k
remaining.  MiniMax counts input PLUS the output a request reserves against
the window, and every M3 request reserves ``max_completion_tokens: 131072``,
so the real input limit was about 869k.  Nothing in the framework knew:

- the pre-send guard added the output cap only for the three providers that
  exposed ``get_max_output_tokens()`` (vllm, openrouter, tensorrt_llm);
- the GC threshold and target were percentages of the whole window;
- ``get_context_usage`` (and every client readout) said
  ``remaining = window - used``.

Now every provider that puts an output cap on a request reports it through
``get_max_output_tokens()``, and ``instruction_budget.effective_input_limit``
(``context_limit - reserved_output``) is the one limit the guard, GC and the
readout measure against.

This module checks three things:

1. every provider package that writes an output cap into a request
   (``max_tokens`` / ``max_completion_tokens`` / ``max_output_tokens`` /
   ``maxTokens`` / ``maxOutputTokens``) has a provider class that implements
   ``get_max_output_tokens()`` -- an AST scan, so a provider added later is
   covered whether or not its author knew;
2. for the providers whose cap is not a plain profile knob, the method
   returns the value the request actually carries;
3. a MiniMax-M3 session at 900k tokens of a 1M window is refused before the
   send, and GC sees it above threshold.
"""

import ast
import importlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jaato_server.shared.instruction_budget import (
    InstructionBudget,
    InstructionSource,
    PayloadExceedsContextError,
    effective_input_limit,
)
from jaato_server.shared.jaato_session import JaatoSession
from jaato_server.shared.plugins.gc import GCConfig
from jaato_server.shared.plugins.model_provider.minimax.provider import (
    MiniMaxProvider,
)
from jaato_server.shared.tests.reversion import Reversion


_SESSION = "jaato-server/jaato_server/shared/jaato_session.py"
_BUDGET = "jaato-server/jaato_server/shared/instruction_budget.py"
_MINIMAX = (
    "jaato-server/jaato_server/shared/plugins/model_provider/minimax/provider.py"
)

REVERSIONS = [
    Reversion(
        target=_MINIMAX,
        find="    def get_max_output_tokens(self) -> Optional[int]:\n"
             "        \"\"\"The profile's ``api_params.max_tokens``, else the model's\n",
        replace="    def _get_max_output_tokens_unused(self) -> Optional[int]:\n"
                "        \"\"\"The profile's ``api_params.max_tokens``, else the model's\n",
        because="minimax stops reporting the 131k it reserves on M3",
        test="test_minimax_m3_reports_the_cap_it_sends",
    ),
    Reversion(
        target=_SESSION,
        find="        effective = effective_input_limit(limit, reserved)\n",
        replace="        effective = limit\n",
        because="the pre-send guard compares the prompt against the raw window",
        test="test_minimax_m3_at_900k_of_1m_is_refused_before_the_send",
    ),
    Reversion(
        target=_BUDGET,
        find="        limit = self.effective_input_limit()\n"
             "        if limit == 0:\n"
             "            return 100.0\n",
        replace="        limit = self.context_limit\n",
        because="GC's percent_used is measured against the raw window again",
        test="test_gc_sees_a_reserved_window_above_threshold",
    ),
]


_CAP_KEYS = frozenset({
    "max_tokens", "max_completion_tokens", "max_output_tokens",
    "maxTokens", "maxOutputTokens",
})

_PROVIDERS_DIR = (
    Path(__file__).resolve().parent.parent / "plugins" / "model_provider"
)


def _is_probe_value(node: ast.AST) -> bool:
    """A literal ``1``: the one-token connectivity probes auth modules send.

    A probe is not a reservation on the conversation's requests.
    """
    return isinstance(node, ast.Constant) and node.value == 1


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _assign_writes_a_cap(node: ast.Assign) -> bool:
    """``kwargs["max_tokens"] = value`` with a non-probe value."""
    return any(
        isinstance(target, ast.Subscript)
        and isinstance(target.slice, ast.Constant)
        and target.slice.value in _CAP_KEYS
        and not _is_probe_value(node.value)
        for target in node.targets
    )


def _dict_writes_a_cap(node: ast.Dict) -> bool:
    """``{"maxTokens": value}`` with a non-probe value."""
    return any(
        isinstance(key, ast.Constant) and key.value in _CAP_KEYS
        and not _is_probe_value(value)
        for key, value in zip(node.keys, node.values)
    )


def _call_writes_a_cap(node: ast.Call) -> bool:
    """``setdefault("max_completion_tokens", ...)`` or a ``max_tokens=``
    keyword.  Error constructors (``ContextLimitError(max_tokens=...)``)
    report a limit, they send none."""
    name = _call_name(node)
    if name.endswith("Error"):
        return False
    if (name == "setdefault" and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in _CAP_KEYS):
        return True
    return any(
        kw.arg in _CAP_KEYS and not _is_probe_value(kw.value)
        for kw in node.keywords
    )


_CAP_WRITERS = {
    ast.Assign: _assign_writes_a_cap,
    ast.Dict: _dict_writes_a_cap,
    ast.Call: _call_writes_a_cap,
}


def _writes_a_cap(tree: ast.AST) -> bool:
    """Whether a module puts an output cap on a request.

    The cap keys as a subscript assignment target, a dict-literal key, a
    ``setdefault`` key, or a call keyword.  Probe values (``1``) are
    skipped.
    """
    for node in ast.walk(tree):
        check = _CAP_WRITERS.get(type(node))
        if check is not None and check(node):
            return True
    return False


def _cap_writing_packages():
    """Provider packages with a module that writes an output cap."""
    found = set()
    for path in sorted(_PROVIDERS_DIR.glob("*/*.py")):
        package = path.parent.name
        if package.startswith("_") or package == "tests":
            continue
        if _writes_a_cap(ast.parse(path.read_text())):
            found.add(package)
    return sorted(found)


def _provider_classes(package: str):
    """Classes defined in ``<package>.provider`` that are model providers."""
    module = importlib.import_module(
        f"jaato_server.shared.plugins.model_provider.{package}.provider"
    )
    return [
        obj for obj in vars(module).values()
        if isinstance(obj, type)
        and obj.__module__ == module.__name__
        and callable(getattr(obj, "get_context_limit", None))
    ]


def test_the_scan_finds_the_providers_the_issue_names():
    """The detector is not vacuous: it finds the caps the issue lists."""
    packages = set(_cap_writing_packages())
    assert {"minimax", "anthropic", "bedrock", "github_models"} <= packages


@pytest.mark.parametrize("package", _cap_writing_packages())
def test_every_provider_that_sends_a_cap_reports_it(package):
    """A provider that caps output implements ``get_max_output_tokens()``."""
    try:
        classes = _provider_classes(package)
    except ImportError as exc:  # optional SDK not installed here
        pytest.skip(f"{package}: {exc}")
    assert classes, f"{package}.provider defines no provider class"
    for cls in classes:
        assert callable(getattr(cls, "get_max_output_tokens", None)), (
            f"{cls.__name__} puts an output cap on its requests but does not "
            "implement get_max_output_tokens(), so the context checks cannot "
            "reserve it (#1444)"
        )


def _minimax(model="MiniMax-M3", api_params=None):
    provider = MiniMaxProvider()
    provider._model_name = model
    provider._context_length = 1_000_000
    provider._api_params = dict(api_params or {})
    return provider


def test_minimax_m3_reports_the_cap_it_sends():
    provider = _minimax()
    kwargs = {}
    provider._apply_api_params(kwargs, None)
    assert kwargs["max_completion_tokens"] == 131_072
    assert provider.get_max_output_tokens() == 131_072


def test_minimax_profile_cap_wins_and_is_what_is_sent():
    provider = _minimax(api_params={"max_tokens": 8_192})
    kwargs = {}
    provider._apply_api_params(kwargs, None)
    assert kwargs["max_completion_tokens"] == 8_192
    assert provider.get_max_output_tokens() == 8_192


def test_the_openai_compat_base_reports_a_profile_cap_under_its_wire_name():
    from jaato_server.shared.plugins.model_provider.kimi.provider import (
        KimiProvider,
    )
    provider = KimiProvider()
    provider._model_name = "kimi-k3"
    provider._api_params = {"max_tokens": 4_096}
    kwargs = {}
    provider._apply_api_params(kwargs, None)
    assert kwargs["max_completion_tokens"] == provider.get_max_output_tokens() == 4_096


def test_the_openai_compat_base_reports_none_when_no_cap_is_sent():
    from jaato_server.shared.plugins.model_provider.nim.provider import (
        NIMProvider,
    )
    provider = NIMProvider()
    provider._model_name = "meta/llama-3.3-70b-instruct"
    kwargs = {}
    provider._apply_api_params(kwargs, None)
    assert "max_tokens" not in kwargs
    assert provider.get_max_output_tokens() is None


def test_anthropic_reports_the_cap_complete_would_send():
    anthropic = pytest.importorskip(
        "jaato_server.shared.plugins.model_provider.anthropic.provider")
    provider = anthropic.AnthropicProvider()
    provider._model_name = "claude-sonnet-4-5"
    assert provider.get_max_output_tokens() == anthropic.DEFAULT_MAX_TOKENS
    provider._max_tokens_override = 2_000
    assert provider.get_max_output_tokens() == 2_000


def _session_on(provider, conversation_tokens):
    session = JaatoSession(MagicMock(), "MiniMax-M3")
    session._provider = provider
    budget = InstructionBudget.create_default(
        session_id="s", context_limit=0)
    budget.set_entry(InstructionSource.CONVERSATION, tokens=conversation_tokens)
    session._instruction_budget = budget
    session._refresh_context_limit_from_provider()
    return session


def test_the_reservation_is_stamped_with_the_window():
    session = _session_on(_minimax(), 0)
    budget = session._instruction_budget
    assert budget.context_limit == 1_000_000
    assert budget.reserved_output == 131_072
    assert budget.effective_input_limit() == effective_input_limit(
        1_000_000, 131_072) == 868_928


def test_minimax_m3_at_900k_of_1m_is_refused_before_the_send():
    session = _session_on(_minimax(), 900_000)
    with pytest.raises(PayloadExceedsContextError) as exc_info:
        session._assert_payload_fits_context()
    assert exc_info.value.max_output_tokens == 131_072


def test_a_prompt_under_the_effective_limit_is_sent():
    session = _session_on(_minimax(), 860_000)
    session._assert_payload_fits_context()


def test_gc_sees_a_reserved_window_above_threshold():
    """750k of a 1M window is 75% raw, 86% of the ~869k a request can carry."""
    from jaato_server.shared.plugins.gc_budget.plugin import BudgetGCPlugin

    session = _session_on(_minimax(), 750_000)
    usage = session.get_context_usage()
    assert usage["reserved_output_tokens"] == 131_072
    assert usage["effective_input_limit"] == 868_928
    assert usage["tokens_remaining"] == 868_928 - 750_000
    assert usage["percent_used"] > 80.0

    plugin = BudgetGCPlugin()
    plugin.initialize({})
    should_gc, _reason = plugin.should_collect(
        usage, GCConfig(threshold_percent=80.0))
    assert should_gc
