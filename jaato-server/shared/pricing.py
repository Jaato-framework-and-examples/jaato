"""Pricing table loader and cost computation.

The daemon optionally loads ``.jaato/pricing.json`` (workspace) or
``~/.jaato/pricing.json`` (user-level) and uses it to populate
``UsageBreakdown.cost_usd`` on emitted Context/Turn events.

Schema is intentionally **Litellm-compatible** so operators can dump
the relevant subset of Litellm's model_prices_and_context_window.json
straight into ``.jaato/pricing.json`` without translation:

.. code-block:: json

    {
      "claude-sonnet-4-5": {
        "input_cost_per_token":             3e-06,
        "output_cost_per_token":            15e-06,
        "cache_read_input_token_cost":      3e-07,
        "cache_creation_input_token_cost":  3.75e-06
      },
      "gpt-4o": {
        "input_cost_per_token":  2.5e-06,
        "output_cost_per_token": 1e-05
      }
    }

Unknown models or missing keys produce ``None`` cost — never zero,
never a silent default. Operators are expected to keep the table
current; we don't ship hardcoded prices that would drift the moment
a provider updates rates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Litellm-compatible field names. Any key not in this set is ignored.
_LITELLM_INPUT = "input_cost_per_token"
_LITELLM_OUTPUT = "output_cost_per_token"
_LITELLM_CACHE_READ = "cache_read_input_token_cost"
_LITELLM_CACHE_CREATION = "cache_creation_input_token_cost"


class PricingTable:
    """Per-model pricing snapshot loaded from JSON.

    Stateless after construction. Lookups by exact model name; no
    fuzzy matching, no version stripping — the operator decides
    whether ``claude-sonnet-4-5`` and ``claude-sonnet-4-5-20250514``
    share an entry.
    """

    def __init__(self, prices: Dict[str, Dict[str, float]]) -> None:
        self._prices = prices

    @classmethod
    def empty(cls) -> "PricingTable":
        """An empty table — every lookup returns ``None``.

        Used as the default when no pricing JSON is found, so the
        framework keeps emitting events unchanged (just with
        ``cost_usd=None``).
        """
        return cls({})

    @classmethod
    def from_paths(cls, paths: list[Path]) -> "PricingTable":
        """Load and merge pricing from a list of JSON file paths.

        Files are processed in **decreasing precedence order** —
        earlier paths win on key conflicts. Missing files are
        skipped silently; malformed files are logged and skipped
        (no exception bubbles up to break session creation).
        """
        merged: Dict[str, Dict[str, float]] = {}
        for path in reversed(paths):  # reverse so earlier paths overwrite
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("pricing: failed to load %s: %s", path, exc)
                continue
            if not isinstance(data, dict):
                logger.warning(
                    "pricing: %s is not a JSON object; skipping", path
                )
                continue
            for model, fields in data.items():
                if isinstance(fields, dict):
                    merged[str(model)] = {
                        k: float(v) for k, v in fields.items()
                        if isinstance(v, (int, float))
                    }
        return cls(merged)

    def has(self, model_name: str) -> bool:
        """Return whether the table has an entry for ``model_name``."""
        return model_name in self._prices

    def cost_for_usage(
        self,
        model_name: str,
        prompt_tokens: int,
        output_tokens: int,
        cache_read_tokens: Optional[int] = None,
        cache_creation_tokens: Optional[int] = None,
    ) -> Optional[float]:
        """Compute cost in USD from token counts.

        Returns ``None`` when the model isn't in the table OR when
        none of its entries are populated. ``None`` means "I don't
        know"; consumers must not interpret it as "free".

        Token counts default to 0 — a missing field on the wire
        contributes nothing rather than poisoning the total.
        """
        entry = self._prices.get(model_name)
        if not entry:
            return None

        total = 0.0
        any_known = False

        input_rate = entry.get(_LITELLM_INPUT)
        if input_rate is not None and prompt_tokens:
            total += input_rate * prompt_tokens
            any_known = True

        output_rate = entry.get(_LITELLM_OUTPUT)
        if output_rate is not None and output_tokens:
            total += output_rate * output_tokens
            any_known = True

        cache_read_rate = entry.get(_LITELLM_CACHE_READ)
        if cache_read_rate is not None and cache_read_tokens:
            total += cache_read_rate * cache_read_tokens
            any_known = True

        cache_creation_rate = entry.get(_LITELLM_CACHE_CREATION)
        if cache_creation_rate is not None and cache_creation_tokens:
            total += cache_creation_rate * cache_creation_tokens
            any_known = True

        # Even a model entry with all zero rates returns 0.0 (deliberate),
        # but a model entry that's literally empty / has no recognised
        # keys returns None (we couldn't compute anything).
        return total if any_known else None


def discover_pricing_paths(workspace_path: Optional[str] = None) -> list[Path]:
    """Build the precedence-ordered list of pricing files to consult.

    Workspace overrides user. Caller can extend this list (e.g.
    premium ships its own pricing source) by passing the result
    through ``PricingTable.from_paths``.
    """
    paths = []
    if workspace_path:
        paths.append(Path(workspace_path) / ".jaato" / "pricing.json")
    paths.append(Path.home() / ".jaato" / "pricing.json")
    return paths


def load_pricing(workspace_path: Optional[str] = None) -> PricingTable:
    """Convenience: discover paths and load. Empty table if no files."""
    return PricingTable.from_paths(discover_pricing_paths(workspace_path))
