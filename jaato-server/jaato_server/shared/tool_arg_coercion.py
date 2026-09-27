"""Coerce string tool arguments to the type their schema declares (#1358).

Some models send a JSON-encoded string where a tool's parameter schema
declares an array, an object, an integer, a number or a boolean:
``"operations": "[{...}]"``, ``"max_results": "5"``.  Nothing validates
arguments against the schema before dispatch, so each plugin met the string
in its own way -- ``multiFileEdit`` refused it, ``web_search`` handed it to a
library that divided by it, and ``store_memory`` iterated it character by
character and reported every tag as too short.

:func:`coerce_args_to_schema` is the one answer, applied by
``ToolExecutor`` before the permission gate so a policy or evaluator sees the
same arguments the executor will.  It converts only when the conversion is
unambiguous and yields the declared type; anything else is left as it
arrived, so the plugin's own error still applies.  Only TOP-LEVEL
parameters are touched.  Stdlib only.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

_INTEGER = re.compile(r"[+-]?\d+")
_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_BOOLEANS = {"true": True, "false": False}

#: Declared types this module converts a string into, in the order tried
#: when a parameter declares a union of them.
_TARGET_TYPES = ("integer", "number", "boolean", "array", "object")


def _declared_types(prop: Dict[str, Any]) -> Tuple[str, ...]:
    """The property's declared ``type`` values, as a tuple of strings."""
    declared = prop.get("type")
    if isinstance(declared, str):
        return (declared,)
    if isinstance(declared, list):
        return tuple(t for t in declared if isinstance(t, str))
    return ()


def _convert(text: str, target: str) -> Tuple[bool, Any]:
    """Convert ``text`` to ``target``; ``(False, None)`` when it does not fit."""
    stripped = text.strip()
    if target == "integer":
        if _INTEGER.fullmatch(stripped):
            return True, int(stripped)
    elif target == "number":
        if _NUMBER.fullmatch(stripped):
            value = float(stripped)
            return True, int(value) if _INTEGER.fullmatch(stripped) else value
    elif target == "boolean":
        if stripped.lower() in _BOOLEANS:
            return True, _BOOLEANS[stripped.lower()]
    elif target in ("array", "object"):
        try:
            value = json.loads(stripped)
        except ValueError:
            return False, None
        wanted = list if target == "array" else dict
        if isinstance(value, wanted):
            return True, value
    return False, None


def coerce_value(value: Any, prop: Dict[str, Any]) -> Tuple[bool, Any]:
    """Coerce one argument against its property schema.

    Args:
        value: The argument as the model sent it.
        prop: That parameter's JSON Schema property.

    Returns:
        ``(True, converted)`` when ``value`` is a string, the property
        declares a non-string type, and the string converts to one of them.
        ``(False, value)`` otherwise -- including when the property also
        accepts ``string``, since a string is then a legitimate value.
    """
    if not isinstance(value, str) or not isinstance(prop, dict):
        return False, value
    declared = _declared_types(prop)
    if not declared or "string" in declared:
        return False, value
    for target in _TARGET_TYPES:
        if target in declared:
            ok, converted = _convert(value, target)
            if ok:
                return True, converted
    return False, value


def coerce_args_to_schema(
    args: Dict[str, Any], parameters: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[str]]:
    """Coerce a call's top-level string arguments to their declared types.

    Args:
        args: The call's arguments.
        parameters: The tool's parameter schema (``ToolSchema.parameters``),
            or ``None`` when it is not known.

    Returns:
        ``(args, changes)``.  ``args`` is a NEW dict when anything changed
        and the same object otherwise; ``changes`` names each converted
        parameter as ``"<name>:str-><type>"`` for the caller to trace.
    """
    if not isinstance(args, dict) or not args or not isinstance(parameters, dict):
        return args, []
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return args, []
    coerced: Optional[Dict[str, Any]] = None
    changes: List[str] = []
    for key, value in args.items():
        ok, converted = coerce_value(value, properties.get(key))
        if ok:
            if coerced is None:
                coerced = dict(args)
            coerced[key] = converted
            changes.append(f"{key}:str->{type(converted).__name__}")
    return (coerced if coerced is not None else args), changes
