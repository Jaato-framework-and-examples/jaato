"""Resolver for ``signal_completion`` payload schemas declared on profiles.

A profile's ``completion_payload_schema`` field accepts either:

- An **inline dict** — used as-is.
- A **string path** — resolved through the standard tier
  (absolute → ``<workspace>/.jaato/completion_schemas/<path>`` →
  ``~/.jaato/completion_schemas/<path>``) and the JSON file is loaded.

Returns the parsed JSON Schema dict ready to be embedded in
``signal_completion``'s tool parameters and used for ``jsonschema.validate``.

Mirrors ``shared/script_loader.py`` in spirit but targets a different
filesystem convention (``.jaato/completion_schemas/``) and produces a
schema dict, not a callable.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


COMPLETION_SCHEMAS_SUBDIR = "completion_schemas"


def resolve_completion_schema(
    schema_ref: Union[str, Dict[str, Any], None],
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve a profile's ``completion_payload_schema`` field to a JSON Schema dict.

    Args:
        schema_ref: The raw value of ``profile.completion_payload_schema``.
            Accepted shapes:

            - ``None`` → returns ``None`` (caller falls back to the legacy
              ``summary: str`` parameter).
            - ``dict`` → returned as-is.
            - ``str`` → treated as a path; resolved through the standard
              tier and loaded as JSON.
        workspace_path: Workspace root for resolving the workspace-relative
            tier (``<workspace>/.jaato/completion_schemas/<path>``). When
            ``None``, only the absolute-path and home-tier
            (``~/.jaato/completion_schemas/<path>``) are tried for relative
            references.
        config_root: Optional override for the workspace tier.  When set,
            ``<config_root>/completion_schemas/<path>`` replaces
            ``<workspace>/.jaato/completion_schemas/<path>``.

    Returns:
        The resolved JSON Schema dict, or ``None`` when ``schema_ref`` is
        ``None`` or resolution fails (file missing, JSON invalid, wrong
        shape). Failures are logged at WARNING level. The caller decides
        what "missing or broken schema" means for its feature (typically:
        fall back to the legacy untyped ``summary`` parameter).
    """
    if schema_ref is None:
        return None

    if isinstance(schema_ref, dict):
        return schema_ref

    if not isinstance(schema_ref, str):
        logger.warning(
            "completion_payload_schema must be a dict or string path, got %s",
            type(schema_ref).__name__,
        )
        return None

    resolved = _resolve_schema_path(schema_ref, workspace_path, config_root)
    if resolved is None:
        logger.warning(
            "completion_payload_schema path not found in any tier: %s "
            "(tried absolute, config_root/%s/, <workspace>/.jaato/%s/, ~/.jaato/%s/)",
            schema_ref, COMPLETION_SCHEMAS_SUBDIR,
            COMPLETION_SCHEMAS_SUBDIR, COMPLETION_SCHEMAS_SUBDIR,
        )
        return None

    try:
        content = resolved.read_text(encoding='utf-8')
        data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning("Invalid JSON in completion schema %s: %s", resolved, e)
        return None
    except OSError as e:
        logger.warning("Cannot read completion schema %s: %s", resolved, e)
        return None

    if not isinstance(data, dict):
        logger.warning(
            "Completion schema must be a JSON object, got %s in %s",
            type(data).__name__, resolved,
        )
        return None

    return data


def _resolve_schema_path(
    path: str,
    workspace_path: Optional[str],
    config_root: Optional[str] = None,
) -> Optional[Path]:
    """Three-tier path resolution for completion schemas.

    Resolution order:

    1. If ``path`` is absolute, use it directly (must exist).
    2. **Workspace tier** —
       * if ``config_root`` is set: ``<config_root>/completion_schemas/<path>``;
       * else if ``workspace_path`` is set: ``<workspace>/.jaato/completion_schemas/<path>``.
    3. **User tier** — ``~/.jaato/completion_schemas/<path>``.

    Returns the resolved ``Path`` pointing at an existing file, or
    ``None`` when no tier matches.
    """
    p = Path(path)
    if p.is_absolute():
        return p if p.is_file() else None

    if config_root:
        cr_path = Path(config_root).expanduser().resolve() / COMPLETION_SCHEMAS_SUBDIR / path
        if cr_path.is_file():
            return cr_path
    elif workspace_path:
        ws_path = Path(workspace_path) / ".jaato" / COMPLETION_SCHEMAS_SUBDIR / path
        if ws_path.is_file():
            return ws_path

    home_path = Path.home() / ".jaato" / COMPLETION_SCHEMAS_SUBDIR / path
    if home_path.is_file():
        return home_path

    return None
