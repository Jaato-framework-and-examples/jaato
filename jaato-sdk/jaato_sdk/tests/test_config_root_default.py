"""`config_root` defaults to `<workspace_path>/.jaato`, on every transport.

The contract was written down three times — this parameter's own docstring,
`shared/config_resolver.py`, and `jaato-scaffold explain paths` — and applied
in one place: the in-process client, whose comment names the reason
("config-rooted plugins like file_edit fail to init without a config_root").
The daemon transports left it `None`, so the SAME driver got a different
session depending on how it connected.

The asymmetry that hid it: `config_root` has two consumers and only one falls
back.  The config SEARCH PATH appends `<workspace>/.jaato` daemon-side whether
or not the value is set, so profiles and agents always resolved and the
session looked healthy.  A plugin that WRITES under the root reads the VALUE —
`file_edit` raises at `initialize()` without one — so the session came up with
no `writeNewFile` at all, logged at ERROR in the daemon and reported to
neither the model nor the driver.

Derived, not required: an explicit value still wins, so pairing an
elsewhere-rooted `config_root` with a `.jaato`-free workspace (to keep
framework config out of the agent's filesystem tools) works as before.
"""

from pathlib import Path

import pytest

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.client.recovery import IPCRecoveryClient
from jaato_sdk.events import ClientType


def _ipc(**kw):
    return IPCClient("/tmp/jaato-test.sock", client_type=ClientType.API,
                     env_file="/tmp/.env", **kw)


def test_derived_from_the_workspace():
    assert _ipc(workspace_path="/tmp/ws").config_root == str(
        Path("/tmp/ws") / ".jaato")


def test_an_explicit_value_still_wins():
    c = _ipc(workspace_path="/tmp/ws", config_root="/elsewhere/.jaato")
    assert c.config_root == "/elsewhere/.jaato"


def test_no_workspace_derives_nothing():
    """Nothing to derive from — and inventing a cwd-relative root would be
    the #742 mistake (the daemon resolves against ITS cwd, not ours)."""
    assert _ipc().config_root is None


def test_the_derived_value_is_absolute():
    """It crosses the daemon boundary, where a relative path is refused."""
    assert Path(_ipc(workspace_path="/tmp/ws").config_root).is_absolute()


def test_a_relative_workspace_is_still_refused_not_derived_from():
    from jaato_sdk.path_boundary import RelativePathAcrossBoundaryError
    with pytest.raises(RelativePathAcrossBoundaryError):
        _ipc(workspace_path="ws")


def test_the_recovery_client_inherits_it_through_the_inner_client():
    """Recovery forwards config_root verbatim, so it must not need its own
    copy of the rule — its docstring already records what an unset one cost
    ("file_edit lost its backup subtree")."""
    r = IPCRecoveryClient("/tmp/jaato-test.sock", client_type=ClientType.API,
                          env_file="/tmp/.env", workspace_path="/tmp/ws")
    inner = IPCClient("/tmp/jaato-test.sock", client_type=ClientType.API,
                      env_file="/tmp/.env", workspace_path="/tmp/ws",
                      config_root=r._config_root)
    assert inner.config_root == str(Path("/tmp/ws") / ".jaato")
