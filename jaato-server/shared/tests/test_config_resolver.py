"""Tests for :mod:`shared.config_resolver` — the read-only-config search-path
helper that lets a session decouple its workspace (the agent's filesystem)
from its config_root (where the daemon reads profiles, agents, prompts,
etc.)."""

from pathlib import Path

import pytest

from shared.config_resolver import (
    JAATO_DIRNAME,
    TIER_USER,
    TIER_WORKSPACE,
    resolve_config_search_path,
    resolve_domain_search_path,
    resolve_secret_uri,
    workspace_state_path,
)


class TestResolveConfigSearchPath:
    """Tests for :func:`resolve_config_search_path`.

    The helper produces an ordered list of root directories the daemon
    scans for read-only framework config.  The contract is documented
    in §2 of the design doc (task #5):

      1. Primary tier — explicit ``config_root`` if set, else
         ``<workspace>/.jaato``, else omitted.
      2. Secondary tier — ``~/.jaato`` (always appended).

    The user tier must be honored even when ``config_root`` is set —
    a user requirement called out explicitly in the design.
    """

    def test_default_today_behavior_unchanged(self, tmp_path):
        """No config_root → workspace/.jaato then user/.jaato (today's chain)."""
        ws = tmp_path / "proj"
        ws.mkdir()
        home = tmp_path / "home"
        home.mkdir()

        roots = resolve_config_search_path(str(ws), user_home=home)

        assert roots == [ws / ".jaato", home / ".jaato"]

    def test_explicit_config_root_replaces_workspace_tier(self, tmp_path):
        """When config_root is set, the workspace tier is replaced by it.

        The user-tier ``~/.jaato`` is **still** appended — that is the
        explicit requirement from the design ("the second tier in
        ~/.jaato needs to be honored as well").
        """
        ws = tmp_path / "proj" / "sandbox"
        ws.mkdir(parents=True)
        config_root = tmp_path / "proj" / ".jaato"
        config_root.mkdir(parents=True)
        home = tmp_path / "home"
        home.mkdir()

        roots = resolve_config_search_path(
            str(ws), config_root=str(config_root), user_home=home,
        )

        # Primary tier is the explicit config_root, NOT ws/.jaato.
        # User tier is preserved.
        assert roots == [config_root.resolve(), home / ".jaato"]

    def test_config_root_path_used_as_is_no_jaato_join(self, tmp_path):
        """``config_root`` is treated as already-anchored — the helper
        does NOT append ``.jaato`` to it.

        This is critical: callers pass e.g. ``/proj/.jaato`` (the
        directory that holds ``profiles/``, ``agents/`` etc.), not
        ``/proj``.  Auto-joining ``.jaato`` would make the path resolve
        to ``/proj/.jaato/.jaato`` and break every loader.
        """
        config_root = tmp_path / "myconfig"
        config_root.mkdir()

        roots = resolve_config_search_path(
            None, config_root=str(config_root), user_home=tmp_path / "home",
        )

        # First root is config_root itself, no ``.jaato`` segment added.
        assert roots[0] == config_root.resolve()

    def test_no_workspace_no_config_root_returns_only_user_tier(self, tmp_path):
        """Daemon-internal lookups without a session-bound workspace
        still get the user tier — useful for one-off discovery without
        a workspace context."""
        home = tmp_path / "home"
        home.mkdir()

        roots = resolve_config_search_path(None, user_home=home)

        assert roots == [home / ".jaato"]

    def test_workspace_path_expansion(self, tmp_path, monkeypatch):
        """Tilde-prefixed workspace paths expand via ``Path.expanduser``.

        The daemon receives workspace_path as a string from the SDK
        and must not crash on ``~/projects/x``-style inputs.
        """
        # Point HOME at tmp_path so expanduser resolves predictably.
        monkeypatch.setenv("HOME", str(tmp_path))
        ws_target = tmp_path / "proj"
        ws_target.mkdir()

        roots = resolve_config_search_path(
            "~/proj", user_home=tmp_path / "user_home",
        )

        # First root expanded to tmp_path/proj/.jaato; user tier from
        # the override (not affected by HOME monkeypatch).
        assert roots[0] == (ws_target / ".jaato").resolve()

    def test_empty_string_config_root_is_treated_as_unset(self, tmp_path):
        """Defensive: an empty-string ``config_root`` (e.g. a client
        passing ``""`` over the wire) must fall back to the workspace
        tier rather than silently using ``Path("")`` which resolves to
        the cwd."""
        ws = tmp_path / "proj"
        ws.mkdir()
        home = tmp_path / "home"
        home.mkdir()

        roots = resolve_config_search_path(
            str(ws), config_root="", user_home=home,
        )

        # Empty string is falsy → falls back to workspace tier.
        assert roots == [ws / ".jaato", home / ".jaato"]


class TestResolveDomainSearchPath:
    """Tests for :func:`resolve_domain_search_path`, the domain-scoped
    shim used by loaders that want today's
    ``[(path, tier_name)]`` shape."""

    def test_joins_domain_under_each_root(self, tmp_path):
        ws = tmp_path / "proj"
        ws.mkdir()
        home = tmp_path / "home"
        home.mkdir()

        result = resolve_domain_search_path(
            str(ws), domain="profiles", user_home=home,
        )

        assert result == [
            (ws / ".jaato" / "profiles", TIER_WORKSPACE),
            (home / ".jaato" / "profiles", TIER_USER),
        ]

    def test_config_root_replaces_workspace_tier_in_domain_form(self, tmp_path):
        config_root = tmp_path / "explicit_config"
        config_root.mkdir()
        home = tmp_path / "home"
        home.mkdir()

        result = resolve_domain_search_path(
            str(tmp_path / "proj" / "sandbox"),
            config_root=str(config_root),
            domain="references",
            user_home=home,
        )

        assert result == [
            (config_root.resolve() / "references", TIER_WORKSPACE),
            (home / ".jaato" / "references", TIER_USER),
        ]

    def test_no_workspace_no_config_root_only_user_tier(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()

        result = resolve_domain_search_path(
            None, domain="agents", user_home=home,
        )

        # User-only result is tagged TIER_USER.  No workspace tier exists.
        assert result == [(home / ".jaato" / "agents", TIER_USER)]

    def test_multi_segment_domain(self, tmp_path):
        """Domain may be a Path with multiple segments (e.g.
        ``services/_discovered``).  The helper joins it onto each
        root verbatim."""
        ws = tmp_path / "proj"
        ws.mkdir()
        home = tmp_path / "home"
        home.mkdir()

        result = resolve_domain_search_path(
            str(ws),
            domain=Path("services") / "_discovered",
            user_home=home,
        )

        assert result == [
            (ws / ".jaato" / "services" / "_discovered", TIER_WORKSPACE),
            (home / ".jaato" / "services" / "_discovered", TIER_USER),
        ]


class TestWorkspaceStatePath:
    """Tests for :func:`workspace_state_path` — the deliberately
    config_root-unaware helper for runtime/state writes."""

    def test_anchors_under_workspace_jaato_dir(self, tmp_path):
        ws = tmp_path / "proj"
        ws.mkdir()

        path = workspace_state_path(str(ws), "sessions/abc123.json")

        assert path == ws / ".jaato" / "sessions" / "abc123.json"

    def test_path_subpath_accepted(self, tmp_path):
        ws = tmp_path / "proj"
        ws.mkdir()

        path = workspace_state_path(str(ws), Path("logs") / "session.log")

        assert path == ws / ".jaato" / "logs" / "session.log"

    def test_does_not_create_directories(self, tmp_path):
        """The helper is a pure path computation — it must not create
        anything on disk.  Callers control mkdir/permissions."""
        ws = tmp_path / "proj"
        ws.mkdir()

        path = workspace_state_path(str(ws), "vision/img.png")

        assert not (ws / ".jaato").exists()
        assert not path.exists()

    def test_jaato_dirname_constant_consistent(self, tmp_path):
        """Sanity: the constant ``JAATO_DIRNAME`` matches what the
        helpers actually use, so external callers building paths with
        it stay in sync with the resolver."""
        assert JAATO_DIRNAME == ".jaato"

        ws = tmp_path / "proj"
        ws.mkdir()

        # Build the same path two ways and confirm equality.
        via_helper = workspace_state_path(str(ws), "x")
        via_constant = ws / JAATO_DIRNAME / "x"

        assert via_helper == via_constant


class TestResolveSecretUri:
    """Tests for :func:`resolve_secret_uri` — the public entry point for
    secret-URI resolution.

    These cases are resolver-independent (they exercise only the
    pass-through contract), so they need no ``jaato.premium`` secret
    resolver installed and run deterministically in CI. The actual
    ``scheme://`` resolution is the premium resolver's contract, tested
    where those resolvers live.
    """

    def test_public_name_is_importable(self):
        """The function is reachable at the public location, not only via
        the private ``shared.plugins.subagent.config._resolve_secret_uri``."""
        assert callable(resolve_secret_uri)
        assert resolve_secret_uri.__module__ == "shared.config_resolver"

    def test_plain_string_passes_through(self):
        """A non-URI credential value is returned unchanged."""
        assert resolve_secret_uri("sk-or-plain-key-123") == "sk-or-plain-key-123"

    def test_network_scheme_passes_through(self):
        """Standard network schemes are never treated as secret URIs."""
        assert resolve_secret_uri("https://example.com/x") == "https://example.com/x"
        assert resolve_secret_uri("wss://host:8099/?token=t") == "wss://host:8099/?token=t"

    def test_unresolved_var_passes_through(self):
        """A still-present ``${VAR}`` reference is not a secret URI."""
        assert resolve_secret_uri("${OPENROUTER_API_KEY}") == "${OPENROUTER_API_KEY}"
