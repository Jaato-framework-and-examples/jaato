"""Regression: the runner bootstrap seeds the workspace ContextVar + env.

Runner-tier path plugins (filesystem_query / file_edit / cli / notebook) read
the session workspace from ``session_context.get_workspace_root()`` (+ an
``os.environ['JAATO_WORKSPACE_ROOT']`` fallback) at ``initialize()``.  The
runner historically set NEITHER — unlike the daemon's
``JaatoServer._in_workspace`` — so pool-/cold-served sessions initialized with
``workspace=none`` and path tools were denied, UNLESS ``JAATO_WORKSPACE_ROOT``
was exported globally (the env mask that hid this build-wide).  Confirmed via a
plain ``IPCClient(workspace_path=ws)`` session logging ``workspace=none`` on an
unmasked daemon; fixed by ``_set_runner_workspace_context`` before ``expose_all``.
"""

import os

from server.runner.session import _set_runner_workspace_context
from shared.session_context import (
    get_workspace_root, get_config_root,
    set_workspace_root, reset_workspace_root,
    set_config_root, reset_config_root,
)


def _clean_env():
    os.environ.pop("JAATO_WORKSPACE_ROOT", None)
    os.environ.pop("JAATO_CONFIG_ROOT", None)


def test_seeds_contextvar_and_env_from_workspace_path():
    wt, ct = set_workspace_root(None), set_config_root(None)
    _clean_env()
    try:
        _set_runner_workspace_context("/ws/x", "/cfg/y")
        assert get_workspace_root() == "/ws/x"            # ContextVar (cached at init)
        assert get_config_root() == "/cfg/y"
        assert os.environ["JAATO_WORKSPACE_ROOT"] == "/ws/x"  # os.environ fallback
        assert os.environ["JAATO_CONFIG_ROOT"] == "/cfg/y"
    finally:
        reset_workspace_root(wt); reset_config_root(ct); _clean_env()


def test_noop_on_falsy_workspace():
    # The daemon-tier / no-workspace path must not crash or set a stale root.
    wt = set_workspace_root(None)
    _clean_env()
    try:
        _set_runner_workspace_context(None, None)
        assert get_workspace_root() is None
        assert "JAATO_WORKSPACE_ROOT" not in os.environ
    finally:
        reset_workspace_root(wt); _clean_env()


def test_workspace_without_config_root():
    wt, ct = set_workspace_root(None), set_config_root(None)
    _clean_env()
    try:
        _set_runner_workspace_context("/ws/only", None)
        assert get_workspace_root() == "/ws/only"
        assert os.environ["JAATO_WORKSPACE_ROOT"] == "/ws/only"
        assert "JAATO_CONFIG_ROOT" not in os.environ      # config_root falsy → untouched
    finally:
        reset_workspace_root(wt); reset_config_root(ct); _clean_env()
