"""Resolver for ``spawn_subagent`` agent_params schemas declared on profiles.

A profile's ``spawn_payload_schema`` field is the symmetric counterpart to
``completion_payload_schema``: it constrains the shape of ``agent_params``
that callers must pass to ``spawn_subagent(profile=<name>, agent_params=...)``.
The symmetry stops at the type system — a spawn payload is a **string-shaped**
boundary, ratified as such by #883; see the module notes below the resolver
for the decision, and :func:`validate_spawn_params` for the check both spawn
boundaries share.
The framework validates the dict against the schema BEFORE creating the
session, so missing-required-field bugs surface at the spawn boundary
(where the caller can fix them in a retry) instead of at the body-wired
prefetch's runtime check (where the cascade has already happened).

Accepts either:

- An **inline dict** — used as-is.
- A **string path** — resolved through the standard tier
  (absolute → ``<config_root>/spawn_schemas/<path>`` →
  ``<workspace>/.jaato/spawn_schemas/<path>`` →
  ``~/.jaato/spawn_schemas/<path>``) and the JSON file is loaded.

Returns the parsed JSON Schema dict ready for ``jsonschema.validate``.

Mirrors ``shared/completion_schema_loader.py`` in spirit; the loaders
intentionally stay separate (parallel one-purpose functions) rather than
share a generic helper, so the conventions for the two file roots stay
visibly distinct in the codebase.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


SPAWN_SCHEMAS_SUBDIR = "spawn_schemas"


def resolve_spawn_schema(
    schema_ref: Union[str, Dict[str, Any], None],
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve a profile's ``spawn_payload_schema`` field to a JSON Schema dict.

    Args:
        schema_ref: The raw value of ``profile.spawn_payload_schema``.
            Accepted shapes:

            - ``None`` → returns ``None`` (caller skips spawn validation).
            - ``dict`` → returned as-is.
            - ``str`` → treated as a path; resolved through the standard
              tier and loaded as JSON.
        workspace_path: Workspace root for resolving the workspace-relative
            tier (``<workspace>/.jaato/spawn_schemas/<path>``).
        config_root: Optional override for the workspace tier.  When set,
            ``<config_root>/spawn_schemas/<path>`` replaces
            ``<workspace>/.jaato/spawn_schemas/<path>``.

    Returns:
        The resolved JSON Schema dict, or ``None`` when ``schema_ref`` is
        ``None`` or resolution fails (file missing, JSON invalid, wrong
        shape). Failures are logged at WARNING level. The caller decides
        what "missing or broken schema" means for its feature (typically:
        skip spawn validation rather than refuse the spawn).
    """
    if schema_ref is None:
        return None

    if isinstance(schema_ref, dict):
        return schema_ref

    if not isinstance(schema_ref, str):
        logger.warning(
            "spawn_payload_schema must be a dict or string path, got %s",
            type(schema_ref).__name__,
        )
        return None

    resolved = _resolve_schema_path(schema_ref, workspace_path, config_root)
    if resolved is None:
        logger.warning(
            "spawn_payload_schema path not found in any tier: %s "
            "(tried absolute, config_root/%s/, <workspace>/.jaato/%s/, ~/.jaato/%s/)",
            schema_ref, SPAWN_SCHEMAS_SUBDIR,
            SPAWN_SCHEMAS_SUBDIR, SPAWN_SCHEMAS_SUBDIR,
        )
        return None

    try:
        content = resolved.read_text(encoding='utf-8')
        data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning("Invalid JSON in spawn schema %s: %s", resolved, e)
        return None
    except OSError as e:
        logger.warning("Cannot read spawn schema %s: %s", resolved, e)
        return None

    if not isinstance(data, dict):
        logger.warning(
            "Spawn schema must be a JSON object, got %s in %s",
            type(data).__name__, resolved,
        )
        return None

    return data


def _resolve_schema_path(
    path: str,
    workspace_path: Optional[str],
    config_root: Optional[str] = None,
) -> Optional[Path]:
    """Three-tier path resolution for spawn schemas.

    Canonical convention (since path-symmetry refactor): paths in
    profile JSON are config-root-relative and include the
    ``spawn_schemas/`` subdir explicitly — same shape as
    ``scripts/<name>.py`` for renderers and
    ``completion_schemas/<name>.json`` for completion validators.
    Profile authors write::

        "spawn_payload_schema": "spawn_schemas/refund.json"

    Resolution order mirrors ``completion_schema_loader``: explicit
    form first, then a backward-compat auto-prefix fallback (logged
    at INFO with a deprecation hint), then the user tier.
    """
    p = Path(path)
    if p.is_absolute():
        return p if p.is_file() else None

    # ── Explicit form (canonical) ────────────────────────────────────
    if config_root:
        cr_path = Path(config_root).expanduser().resolve() / path
        if cr_path.is_file():
            return cr_path
    elif workspace_path:
        ws_path = Path(workspace_path) / ".jaato" / path
        if ws_path.is_file():
            return ws_path

    home_path_explicit = Path.home() / ".jaato" / path
    if home_path_explicit.is_file():
        return home_path_explicit

    # ── Backward-compat auto-prefix fallback ─────────────────────────
    if config_root:
        cr_legacy = Path(config_root).expanduser().resolve() / SPAWN_SCHEMAS_SUBDIR / path
        if cr_legacy.is_file():
            logger.info(
                "spawn_payload_schema %r resolved via legacy auto-prefix "
                "(<config_root>/%s/<path>); migrate to explicit "
                "%r for forward compatibility.",
                path, SPAWN_SCHEMAS_SUBDIR, f"{SPAWN_SCHEMAS_SUBDIR}/{path}",
            )
            return cr_legacy
    elif workspace_path:
        ws_legacy = Path(workspace_path) / ".jaato" / SPAWN_SCHEMAS_SUBDIR / path
        if ws_legacy.is_file():
            logger.info(
                "spawn_payload_schema %r resolved via legacy auto-prefix "
                "(<workspace>/.jaato/%s/<path>); migrate to explicit "
                "%r for forward compatibility.",
                path, SPAWN_SCHEMAS_SUBDIR, f"{SPAWN_SCHEMAS_SUBDIR}/{path}",
            )
            return ws_legacy

    home_path_legacy = Path.home() / ".jaato" / SPAWN_SCHEMAS_SUBDIR / path
    if home_path_legacy.is_file():
        logger.info(
            "spawn_payload_schema %r resolved via legacy auto-prefix "
            "(~/.jaato/%s/<path>); migrate to explicit %r for forward compatibility.",
            path, SPAWN_SCHEMAS_SUBDIR, f"{SPAWN_SCHEMAS_SUBDIR}/{path}",
        )
        return home_path_legacy

    return None


# ─────────────────────────────────────────────────────────────────────
# The string-only spawn contract (#883)
# ─────────────────────────────────────────────────────────────────────
#
# ``spawn_payload_schema`` reads as the input-boundary mirror of
# ``completion_payload_schema``, and the symmetry is real everywhere
# EXCEPT the type system.  A completion payload is JSON the model
# emitted; a spawn payload is not JSON on the wire at all.
# ``IPCClient.create_session`` flattens ``agent_params`` into argv
# tokens::
#
#     args.append(f"{key}={value}")          # jaato_sdk/client/ipc.py
#
# and ``command_router._handle_session_new`` partitions them back on the
# first ``=``.  Every value therefore reaches the daemon as a STRING,
# and a property declared ``integer`` / ``number`` / ``boolean`` /
# ``object`` / ``array`` is refused on EVERY spawn whatever the caller
# passes.  Two features that are each correct in isolation cannot both
# be honoured, and the wire wins.
#
# #883 ratifies the wire's behaviour as the contract: **a spawn payload
# is a string-shaped boundary.**  ``pattern`` carries the shape,
# consumers parse.  The alternatives were rejected on record — carrying
# agent_params as JSON restores a symmetry a ``pattern`` already
# expresses, at the cost of the argv protocol and the string-oriented
# ``{{param}}`` persona substitution; coercing at the boundary invents a
# second, undocumented type system in which ``"1"`` is unambiguous and
# ``"true"`` / ``"null"`` / ``"[1,2]"`` are not.
#
# Ratifying it means the rule holds at BOTH spawn boundaries.  The
# model-driven ``spawn_subagent`` call is an in-process function call,
# so a model that emits ``{"iteration": 1}`` hands the validator a real
# ``int`` — a typed schema passed there and failed over IPC, so one
# profile meant two different things depending on who spawned it.
# :func:`spawn_params_for_validation` renders the same string view the
# wire produces, and both call sites validate that.  The rendering is
# for VALIDATION ONLY: the params handed to the session, the persona and
# the prefetch are untouched.

#: JSON-Schema types a spawn payload can carry.  ``string`` is what the
#: wire delivers; ``null`` earns its place because ``["string", "null"]``
#: is satisfiable by the string that arrives.
WIRE_SAFE_SPAWN_TYPES = frozenset({"string", "null"})


def spawn_params_for_validation(
    agent_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Render ``agent_params`` the way the IPC wire renders them.

    The daemon receives ``key=value`` argv tokens, so every value it
    validates is ``str(value)``.  This reproduces that view so a profile's
    ``spawn_payload_schema`` means the same thing at the in-process
    ``spawn_subagent`` boundary as it does over IPC.

    ``None`` is passed through rather than rendered as ``"None"``: it is
    the one value a ``null`` type can name, and a caller who passes it
    means "absent", not the four-character string the wire would make of
    it.  Everything else — ``int``, ``bool``, ``list``, ``dict`` — becomes
    its ``str()``, which is exactly what ``f"{key}={value}"`` produces.

    Args:
        agent_params: The caller's spawn params, or ``None``.

    Returns:
        A new dict; the caller's is never mutated.
    """
    if not agent_params:
        return {}
    return {
        key: value if (value is None or isinstance(value, str)) else str(value)
        for key, value in agent_params.items()
    }


def unreachable_spawn_types(
    schema: Optional[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Map each property to the declared types no spawn can deliver.

    Args:
        schema: A resolved spawn schema, or ``None``.

    Returns:
        ``{property_name: [offending type, ...]}``, empty when the schema
        is string-shaped (or declares no ``properties`` at all).  A union
        such as ``["string", "integer"]`` reports its ``integer`` member:
        the string branch is satisfiable, but the integer branch is dead
        code and saying so is cheaper than letting an author believe the
        typed form works.
    """
    offenders: Dict[str, List[str]] = {}
    if not isinstance(schema, dict):
        return offenders
    for key, spec in (schema.get("properties") or {}).items():
        if not isinstance(spec, dict):
            continue
        declared = spec.get("type")
        types = {declared} if isinstance(declared, str) else set(declared or ())
        bad = sorted(t for t in types if t not in WIRE_SAFE_SPAWN_TYPES)
        if bad:
            offenders[key] = bad
    return offenders


def spawn_type_contract_note(offenders: Dict[str, List[str]]) -> str:
    """Explain a validation failure that no caller could have avoided.

    A schema carrying a non-string property refuses every spawn, and the
    bare ``jsonschema`` message for it (``'1' is not of type 'integer'``)
    reads as a caller mistake — it names the value the caller passed and
    blames it, when the value could not have been anything else.  This
    sentence is appended to the refusal so the profile author, not the
    caller, is pointed at.

    Args:
        offenders: The mapping :func:`unreachable_spawn_types` returned.

    Returns:
        The note, with a trailing space so it concatenates; ``""`` when
        the schema is string-shaped and the failure is a real one.
    """
    if not offenders:
        return ""
    listed = ", ".join(
        f"{key} ({'/'.join(types)})" for key, types in sorted(offenders.items())
    )
    return (
        f"THIS SCHEMA REFUSES EVERY SPAWN: agent_params are a string-shaped "
        f"boundary (they cross the IPC wire as `key=value` argv tokens), and "
        f"these properties declare a non-string type: {listed}. Fix the "
        f"PROFILE, not the call — declare them `string` and put the shape in "
        f"a `pattern` (e.g. '^[0-9]+$'), then parse in the prefetch or "
        f"persona. `jaato-scaffold validate` reports this as "
        f"spawn_schema_type_unreachable. "
    )


def validate_spawn_params(
    schema_ref: Union[str, Dict[str, Any], None],
    agent_params: Optional[Dict[str, Any]],
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """Check ``agent_params`` against a profile's ``spawn_payload_schema``.

    The single chokepoint both spawn boundaries use — the model-driven
    ``spawn_subagent`` tool and the daemon's ``create_session`` — so a
    profile's schema cannot mean one thing on one path and another on the
    other.  Validation runs against the string view
    (:func:`spawn_params_for_validation`), per the #883 contract above.

    Args:
        schema_ref: The profile's raw ``spawn_payload_schema`` value.
        agent_params: The caller's params, in whatever types they arrived.
        workspace_path: Workspace root, for path-form schema resolution.
        config_root: Config-root override, for path-form schema resolution.

    Returns:
        ``None`` when the spawn may proceed — the schema passed, was not
        declared, could not be resolved, or the check itself failed (a
        broken schema must not block a spawn; the loader logs it).

        Otherwise a ``details`` fragment the caller embeds in its own
        refusal message: either the full set of missing required fields
        (so a supervisor fixes them all in one retry instead of hammering
        the spawn loop) or the first failure, followed by
        :func:`spawn_type_contract_note` when the schema is the cause.
    """
    if schema_ref is None:
        return None
    try:
        schema = resolve_spawn_schema(
            schema_ref, workspace_path=workspace_path, config_root=config_root,
        )
        if schema is None:
            return None

        import jsonschema
        instance = spawn_params_for_validation(agent_params)
        try:
            jsonschema.validate(instance=instance, schema=schema)
        except jsonschema.ValidationError as exc:
            missing = [
                field_name for field_name in (schema.get('required') or [])
                if field_name not in instance
            ]
            details = (
                f"missing required fields: {missing}. " if missing
                else f"first failure: {exc.message}. "
            )
            return details + spawn_type_contract_note(
                unreachable_spawn_types(schema))
    except Exception as exc:  # noqa: BLE001 - degrade, never block a spawn
        logger.warning(
            "spawn_payload_schema validation skipped (%s: %s)",
            type(exc).__name__, exc,
        )
    return None
