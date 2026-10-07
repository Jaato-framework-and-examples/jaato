"""Which config root a session ran under, and which one a client is in (#1584).

A client that connects without an identity (a shared WS bearer token, or
local IPC) used to be handed EVERY session on the daemon, in
``session.list`` and in the ``SessionInfoEvent`` snapshot that answers
``session.new`` / ``session.attach``.  On a shared daemon that is another
application's session ids, workspace paths, models and descriptions -- and
in production a 1.2 MB frame that broke the client's receive ceiling, so
``create_session`` timed out for every new conversation.

The boundary is the CONFIG ROOT, not the workspace path: two clients may
share a workspace under different config roots, and one config root may
serve several workspaces (a kb cascade runs sessions under
``<repo>/tests/runs/<run>/`` with ``<repo>/.jaato`` as their root).

Stdlib only, so the session manager (which builds the rows) and the
command router (which filters them) read one definition.
"""

from __future__ import annotations

import os
from typing import Optional

#: The config-root directory a workspace implies when nothing names another
#: one -- the same derivation ``IPCClient`` applies to ``workspace_path`` and
#: ``SessionManager._resolve_restore_config_root`` applies at create (#1293)
#: and revive.
DEFAULT_CONFIG_DIR = ".jaato"


def effective_config_root(
    config_root: Optional[str], workspace_path: Optional[str],
) -> Optional[str]:
    """The config root a session ran under, or ``None`` when unknown.

    The recorded value wins.  A record with none but WITH a workspace ran
    under ``<workspace>/.jaato``: since #1293 every create resolves that
    value and records it, and a record predating that (or 2.4's
    persistence) was created with no override, where the config search
    path appends ``<workspace>/.jaato`` -- the same derivation the session
    itself used, so it is evidence about this session, not a guess.  A row
    with neither is unknown and matches no client.
    """
    if config_root:
        return config_root
    if workspace_path:
        return os.path.join(workspace_path, DEFAULT_CONFIG_DIR)
    return None


def config_root_key(path: Optional[str]) -> Optional[str]:
    """*path* resolved and normalised for comparison, or ``None``.

    Symlinks are resolved so a root spelled through a link and the same
    root spelled directly compare equal; ``normcase`` for platforms whose
    filesystem is case-insensitive.
    """
    if not path or not isinstance(path, str):
        return None
    try:
        return os.path.normcase(os.path.realpath(path))
    except (OSError, ValueError):
        return None
