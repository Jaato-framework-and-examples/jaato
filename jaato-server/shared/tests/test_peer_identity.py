"""The IPC transport knows who is on the other end, and acts only on
paths that account could reach itself.

Two properties, and they fail differently, so they are tested apart:

ATTRIBUTION
    ``EventSink.get_client_user`` was a hardcoded ``None`` on IPC, so a
    session created over a socket several accounts share had no
    ``created_by``, no ledger ``user_id`` and no ``user.id`` span — every
    row anonymous, on the one transport where the kernel could have said.

ENTITLEMENT
    ``workspace_path`` / ``config_root`` arrive as plain strings and the
    daemon acts on them with ITS credential — as does the runner, a
    separate process under the same uid.  Nothing asked whether the
    CONNECTING account could have reached them, so on a shared socket the
    daemon was a confused deputy.

The deny cases use a peer uid that is neither 0 nor this process's own,
because both of those short-circuit by design (root bypasses DAC; the
daemon's own account skips the check entirely) — a test run as root that
asserted a refusal for uid 0 would be asserting the opposite of the
contract.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import pytest

from shared.peer_identity import (
    PeerCredentials,
    daemon_uid,
    describe_unreachable_path,
    path_checks_disabled,
    path_reachable_by,
    peer_credentials,
    peer_group_ids,
    peer_is_the_daemon,
    unreachable_client_paths,
)

#: An account this process is not and root is not.  ``nobody`` on most
#: systems; the number is what matters, not whether it resolves.
OTHER_UID = 65534


def _other_peer() -> PeerCredentials:
    return PeerCredentials(uid=OTHER_UID, gid=OTHER_UID, pid=None, username=None)


def _self_peer() -> PeerCredentials:
    return PeerCredentials(uid=os.getuid(), gid=os.getgid(), pid=os.getpid())


@pytest.fixture
def open_dir():
    """A directory whose ANCESTORS are all traversable by anybody.

    ``tmp_path`` is not usable here: pytest roots it at
    ``/tmp/pytest-of-<user>/``, which is ``0700``, so the ancestor walk
    refuses before the leaf's own mode is ever consulted.  Every deny
    assertion would then pass for the wrong reason and prove nothing —
    the vacuous-test shape this suite exists to avoid.  Rooting at
    ``/tmp`` (``1777``) leaves the leaf as the only variable.
    """
    import shutil
    import tempfile

    path = Path(tempfile.mkdtemp(prefix="jaato-peer-"))
    os.chmod(path, 0o755)
    try:
        yield path
    finally:
        os.chmod(path, 0o755)
        shutil.rmtree(path, ignore_errors=True)


def test_the_fixture_is_not_vacuous(open_dir: Path) -> None:
    """The control for every deny case below: with the leaf open, the same
    peer IS allowed.  Without this, a refusal proves only that something
    on the path said no."""
    os.chmod(open_dir, 0o755)
    assert path_reachable_by(str(open_dir), _other_peer()) is True


# ----------------------------------------------------------------------
# Reading the credential
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(socket, "SO_PEERCRED"), reason="SO_PEERCRED is Linux-only",
)
def test_peer_credentials_reads_this_process_off_a_socketpair() -> None:
    """The kernel answers for a real connected socket."""
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        cred = peer_credentials(a)
        assert cred is not None
        assert cred.uid == os.getuid()
        assert cred.gid == os.getgid()
        assert cred.pid == os.getpid()
    finally:
        a.close()
        b.close()


def test_no_socket_is_no_credential_not_an_error() -> None:
    """A transport with no socket object (a Windows pipe) answers None."""
    assert peer_credentials(None) is None


def test_a_socket_the_option_is_refused_on_answers_none() -> None:
    """A closed socket cannot be interrogated, and that is not a crash.

    The whole module is best-effort by contract: a transport that cannot
    report a peer leaves the daemon exactly where it was before this
    existed.
    """
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.close()
    assert peer_credentials(s) is None


def test_identity_falls_back_to_the_numeric_form() -> None:
    """An unresolvable uid must not become an unattributable session."""
    assert PeerCredentials(uid=4242, gid=1).identity == "uid:4242"
    assert PeerCredentials(uid=4242, gid=1, username="ana").identity == "ana"


def test_group_ids_always_include_the_primary_gid() -> None:
    """A uid with no passwd entry contributes its primary gid alone —
    the safe direction, since a missing group can only refuse."""
    assert 7 in peer_group_ids(PeerCredentials(uid=4242, gid=7))


# ----------------------------------------------------------------------
# peer_is_the_daemon — a question about ONE connection
# ----------------------------------------------------------------------


def test_the_daemons_own_account_is_recognised() -> None:
    assert peer_is_the_daemon(_self_peer()) is True
    assert daemon_uid() == os.getuid()


def test_no_peer_is_not_the_daemon() -> None:
    """``None`` means "could not tell", never "it is me"."""
    assert peer_is_the_daemon(None) is False


def test_another_account_is_not_the_daemon() -> None:
    assert peer_is_the_daemon(_other_peer()) is False


# ----------------------------------------------------------------------
# path_reachable_by
# ----------------------------------------------------------------------


def test_a_world_readable_directory_is_reachable(open_dir: Path) -> None:
    os.chmod(open_dir, 0o755)
    assert path_reachable_by(str(open_dir), _other_peer()) is True


def test_a_private_directory_is_not_reachable(open_dir: Path) -> None:
    """The 0700 home directory case — the protection the check exists to
    honour."""
    os.chmod(open_dir, 0o700)
    assert path_reachable_by(str(open_dir), _other_peer()) is False


def test_an_unreadable_ancestor_denies_a_readable_leaf(open_dir: Path) -> None:
    """Without search permission on the way down, the leaf's own mode is
    irrelevant — which is exactly what the kernel does."""
    parent = open_dir / "private"
    child = parent / "public"
    child.mkdir(parents=True)
    os.chmod(child, 0o777)
    os.chmod(parent, 0o700)
    try:
        assert path_reachable_by(str(child), _other_peer()) is False
    finally:
        os.chmod(parent, 0o755)


def test_a_path_that_does_not_exist_asks_about_creating_it(open_dir: Path) -> None:
    """The daemon provisions workspaces, so a not-yet-created directory is
    the ordinary case: the question becomes whether the peer could make it."""
    target = open_dir / "not-yet"
    os.chmod(open_dir, 0o755)          # readable, NOT writable by others
    assert path_reachable_by(str(target), _other_peer()) is False
    os.chmod(open_dir, 0o777)          # now creatable
    assert path_reachable_by(str(target), _other_peer()) is True


def test_a_read_only_tree_is_reachable(open_dir: Path) -> None:
    """Deliberately NOT requiring write: an org-wide ``config_root`` of
    shared profiles, readable by everyone and writable by none of them, is
    the shape this whole feature is for."""
    os.chmod(open_dir, 0o555)
    try:
        assert path_reachable_by(str(open_dir), _other_peer()) is True
    finally:
        os.chmod(open_dir, 0o755)


def test_root_reaches_any_absolute_path() -> None:
    """root bypasses DAC in the kernel; a check that refused it would be
    reporting something untrue."""
    assert path_reachable_by("/root", PeerCredentials(uid=0, gid=0)) is True


def test_a_relative_path_is_unknowable_even_for_root() -> None:
    """The question there is what the string MEANS, not who may read it —
    and #742 refuses it one layer up for the same reason."""
    assert path_reachable_by("rel/ative", PeerCredentials(uid=0, gid=0)) is None
    assert path_reachable_by("", _other_peer()) is None


def test_symlinks_are_resolved_before_judging(open_dir: Path) -> None:
    """A link planted in a world-writable directory is judged by its
    TARGET — the rule the socket path itself already follows."""
    secret = open_dir / "secret"
    secret.mkdir()
    os.chmod(secret, 0o700)
    link = open_dir / "link"
    link.symlink_to(secret)
    os.chmod(open_dir, 0o777)
    try:
        assert path_reachable_by(str(link), _other_peer()) is False
    finally:
        os.chmod(open_dir, 0o755)


# ----------------------------------------------------------------------
# describe_unreachable_path / unreachable_client_paths
# ----------------------------------------------------------------------


def test_an_absent_field_is_not_a_violation() -> None:
    assert describe_unreachable_path("config_root", None, _other_peer()) is None
    assert describe_unreachable_path("config_root", "", _other_peer()) is None


def test_the_message_names_the_field_the_value_and_the_account(
    open_dir: Path,
) -> None:
    os.chmod(open_dir, 0o700)
    message = describe_unreachable_path("workspace", str(open_dir), _other_peer())
    assert message is not None
    assert "workspace" in message
    assert str(open_dir) in message
    assert f"uid:{OTHER_UID}" in message


def test_an_unknowable_path_refuses_rather_than_granting() -> None:
    """Positive evidence only: the caller must not be handed a pass because
    the daemon could not look."""
    message = describe_unreachable_path("workspace", "relative", _other_peer())
    assert message is not None
    assert "could not be checked" in message


def test_no_peer_means_the_check_does_not_apply(open_dir: Path) -> None:
    """WS, Windows pipes, non-Linux sockets: that transport's own access
    control is what applies, and this must not become a denial."""
    os.chmod(open_dir, 0o700)
    assert unreachable_client_paths([("workspace", str(open_dir))], None) == []


def test_the_daemons_own_account_skips_the_check(open_dir: Path) -> None:
    """A connection from the daemon's own uid, which can already reach
    anything the daemon can, so a refusal would deny nothing.  Per
    CONNECTION: the same daemon still checks every other account."""
    os.chmod(open_dir, 0o700)
    assert unreachable_client_paths(
        [("workspace", str(open_dir))], _self_peer(),
    ) == []


def test_a_shared_socket_arms_the_check_with_no_knob(open_dir: Path) -> None:
    """Self-arming is the point: a control nobody remembers to enable is a
    control nobody has."""
    os.chmod(open_dir, 0o700)
    violations = unreachable_client_paths(
        [("workspace", str(open_dir))], _other_peer(),
    )
    assert len(violations) == 1


def test_every_offending_field_is_reported_at_once(open_dir: Path) -> None:
    """One error listing the whole set, not one field per round trip — the
    shape ``_reject_relative_client_paths`` already established."""
    a = open_dir / "a"
    b = open_dir / "b"
    a.mkdir()
    b.mkdir()
    os.chmod(a, 0o700)
    os.chmod(b, 0o700)
    violations = unreachable_client_paths(
        [("workspace", str(a)), ("config_root", str(b))], _other_peer(),
    )
    assert len(violations) == 2


def test_the_operator_opt_out_disables_it_and_is_announced(
    open_dir: Path, monkeypatch: Any, caplog: Any,
) -> None:
    """A weakened boundary is never silent — the posture
    ``--ws-unsafe-no-auth`` and ``scrub_secret_env: none`` already take."""
    import shared.peer_identity as pi

    os.chmod(open_dir, 0o700)
    monkeypatch.setattr(pi, "_announced_opt_out", False)
    monkeypatch.setenv("JAATO_IPC_TRUST_PEER_PATHS", "1")
    with caplog.at_level("WARNING"):
        assert unreachable_client_paths(
            [("workspace", str(open_dir))], _other_peer(),
        ) == []
    assert any(
        "JAATO_IPC_TRUST_PEER_PATHS" in record.message for record in caplog.records
    )


def test_the_opt_out_is_off_unless_explicitly_truthy(monkeypatch: Any) -> None:
    monkeypatch.delenv("JAATO_IPC_TRUST_PEER_PATHS", raising=False)
    assert path_checks_disabled() is False
    monkeypatch.setenv("JAATO_IPC_TRUST_PEER_PATHS", "no")
    assert path_checks_disabled() is False
    monkeypatch.setenv("JAATO_IPC_TRUST_PEER_PATHS", "")
    assert path_checks_disabled() is False
