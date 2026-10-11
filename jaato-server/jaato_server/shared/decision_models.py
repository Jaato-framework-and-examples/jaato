"""The ``decision_models:`` profile block (decision-models.md §7).

A decision model (TypeSafe Jev, ConvAI Laya) answers typed questions
about a state through ``provider.decide()``; it cannot hold a
conversation.  So it is never a tier: a profile binds it by NAME in a
block of its own, and a consumer (the ``decide`` tool, the permission
gate) asks for an entry by that name.  Swapping one decision model for
another is then a profile change, not a consumer change.

```yaml
decision_models:
  triage:
    provider: openrouter
    model: typesafe/jev-1.13
```

This module is the one definition of what an entry is.  The profile
loader, the session and ``jaato-scaffold validate`` all read it:

- :func:`parse_decision_models` normalises a raw block and raises
  :class:`DecisionModelsConfigError` naming the entry it refuses;
- :class:`DecisionModelBinding` is one parsed entry;
- :func:`merge_decision_models` is the inheritance rule (child replaces,
  by entry name).

Stdlib only, no jaato imports, so the loader can import it before
anything else is built.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

#: An entry name: the tier-name rule, so the two vocabularies read alike.
ENTRY_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")

#: How many entries one profile may bind.  Each is a provider instance a
#: session may build, so the bound is on what one session can hold.
MAX_ENTRIES = 16

#: The keys an entry may carry.  ``model`` is required; ``provider``
#: defaults to the session's own provider, as a tier's does.
ENTRY_KEYS = frozenset({"provider", "model"})


class DecisionModelsConfigError(ValueError):
    """A ``decision_models:`` block that cannot be used.

    A ``ValueError`` so the profile loader's existing ``except ValueError``
    reports it the way it reports a bad ``budget_control``.
    """


@dataclass(frozen=True)
class DecisionModelBinding:
    """One parsed ``decision_models:`` entry.

    Attributes:
        name: The entry name a consumer refers to.
        model: The decision model's id on its provider.
        provider: The provider serving it, or ``None`` for the session's
            own provider (resolved when the session builds the binding).
    """

    name: str
    model: str
    provider: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        """The entry as it is written in a profile (``provider`` only when set)."""
        out = {"model": self.model}
        if self.provider:
            out["provider"] = self.provider
        return out


def _nonempty_str(entry: str, key: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DecisionModelsConfigError(
            f"decision_models.{entry}.{key} must be a non-empty string, "
            f"got {value!r}"
        )
    return value.strip()


def _parse_entry(name: Any, raw: Any) -> DecisionModelBinding:
    if not isinstance(name, str) or not ENTRY_NAME_RE.match(name):
        raise DecisionModelsConfigError(
            f"decision_models entry name {name!r} must match "
            f"{ENTRY_NAME_RE.pattern}"
        )
    if not isinstance(raw, Mapping):
        raise DecisionModelsConfigError(
            f"decision_models.{name} must be an object with 'model' "
            f"(and optionally 'provider'), got {type(raw).__name__}"
        )
    unknown = sorted(set(raw) - ENTRY_KEYS)
    if unknown:
        raise DecisionModelsConfigError(
            f"decision_models.{name} has unknown key(s) {unknown}; an entry "
            f"carries only {sorted(ENTRY_KEYS)}"
        )
    if "model" not in raw:
        raise DecisionModelsConfigError(
            f"decision_models.{name} is missing the required 'model'"
        )
    model = _nonempty_str(name, "model", raw["model"])
    provider = raw.get("provider")
    if provider is not None:
        provider = _nonempty_str(name, "provider", provider)
    return DecisionModelBinding(name=name, model=model, provider=provider)


def parse_decision_models(raw: Any) -> Dict[str, DecisionModelBinding]:
    """Normalise a ``decision_models:`` block into bindings.

    Args:
        raw: The block as read from a profile.  ``None`` and ``{}`` mean
            no decision models.

    Returns:
        ``{entry name: binding}``, in the block's order.

    Raises:
        DecisionModelsConfigError: The block is not a mapping, has more
            than :data:`MAX_ENTRIES` entries, or an entry is malformed.
            The message names the entry.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise DecisionModelsConfigError(
            f"decision_models must be an object mapping entry names to "
            f"{{provider, model}}, got {type(raw).__name__}"
        )
    if len(raw) > MAX_ENTRIES:
        raise DecisionModelsConfigError(
            f"decision_models declares {len(raw)} entries; at most "
            f"{MAX_ENTRIES} are allowed"
        )
    return {name: _parse_entry(name, entry) for name, entry in raw.items()}


def decision_models_to_dict(
    bindings: Mapping[str, DecisionModelBinding],
) -> Dict[str, Dict[str, str]]:
    """The inverse of :func:`parse_decision_models`, for snapshots and the envelope."""
    return {name: b.to_dict() for name, b in bindings.items()}


def merge_decision_models(
    parent: Mapping[str, Any], child: Mapping[str, Any],
) -> Dict[str, Any]:
    """Inheritance: the child's entry replaces the parent's of the same name.

    Entries only one side declares are kept.  An entry is replaced whole,
    never merged key by key: a child naming ``triage`` with another model
    and no ``provider`` means the session's provider, not the parent's.
    """
    merged = dict(parent or {})
    merged.update(child or {})
    return merged


class UnknownDecisionModelError(LookupError):
    """A consumer named a ``decision_models:`` entry this session does not bind.

    Carries the names that ARE bound, so the caller can say what to use.
    """

    def __init__(self, name: str, available: "list[str]") -> None:
        self.name = name
        self.available = list(available)
        bound = ", ".join(self.available) if self.available else "none"
        super().__init__(
            f"no decision model named {name!r} is bound in this session's "
            f"profile (decision_models: {bound})"
        )


class DecisionModelUnavailableError(RuntimeError):
    """An entry's provider was built and cannot serve ``decide()``.

    Raised when the provider the entry names does not implement the
    decision wire (it does not declare ``capabilities.decisions``).
    ``jaato-scaffold validate`` reports the same binding before any
    session exists, as ``decision_model_provider_unsupported``.
    """

    def __init__(self, name: str, provider: str, model: str) -> None:
        self.name = name
        self.provider = provider
        self.model = model
        super().__init__(
            f"decision_models.{name} binds {model!r} on provider "
            f"{provider!r}, which does not implement decide(); bind it on a "
            f"provider that declares the decisions capability (openrouter)"
        )
