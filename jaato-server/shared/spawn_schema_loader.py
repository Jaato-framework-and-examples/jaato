"""Resolver for ``spawn_subagent`` agent_params schemas declared on profiles.

A profile's ``spawn_payload_schema`` field is the symmetric counterpart to
``completion_payload_schema``: it constrains the shape of ``agent_params``
that callers must pass to ``spawn_subagent(profile=<name>, agent_params=...)``.
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
from typing import Any, Dict, Optional, Union

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

    Resolution order:

    1. If ``path`` is absolute, use it directly (must exist).
    2. **Workspace tier** —
       * if ``config_root`` is set: ``<config_root>/spawn_schemas/<path>``;
       * else if ``workspace_path`` is set: ``<workspace>/.jaato/spawn_schemas/<path>``.
    3. **User tier** — ``~/.jaato/spawn_schemas/<path>``.

    Returns the resolved ``Path`` pointing at an existing file, or
    ``None`` when no tier matches.
    """
    p = Path(path)
    if p.is_absolute():
        return p if p.is_file() else None

    if config_root:
        cr_path = Path(config_root).expanduser().resolve() / SPAWN_SCHEMAS_SUBDIR / path
        if cr_path.is_file():
            return cr_path
    elif workspace_path:
        ws_path = Path(workspace_path) / ".jaato" / SPAWN_SCHEMAS_SUBDIR / path
        if ws_path.is_file():
            return ws_path

    home_path = Path.home() / ".jaato" / SPAWN_SCHEMAS_SUBDIR / path
    if home_path.is_file():
        return home_path

    return None
