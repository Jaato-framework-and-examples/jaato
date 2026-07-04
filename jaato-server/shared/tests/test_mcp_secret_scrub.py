"""MCP secrets-broker scrub: a model-invokable MCP server subprocess must not
inherit the runner's framework secrets unless the operator opts in.

Closes the MCP half of feature #10 (the cli half shipped in bda2dfb9). The leak
was ServerConfig.to_stdio_params passing the full os.environ to every stdio
server subprocess.
"""

from shared.mcp_context_manager import MCPClientManager, ServerConfig


def _env(params):
    """Extract the env dict a StdioServerParameters would spawn with."""
    return params.env or {}


def test_no_patterns_is_noop(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "s3cret")
    cfg = ServerConfig(name="s", command="x")  # scrub_secret_env defaults to ()
    assert _env(cfg.to_stdio_params()).get("GITHUB_TOKEN") == "s3cret"


def test_declared_secret_stripped_from_inherited_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "s3cret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    monkeypatch.setenv("PATH", "/usr/bin")  # non-secret survives
    cfg = ServerConfig(
        name="s", command="x",
        scrub_secret_env=["*_TOKEN", "*_API_KEY"],
    )
    env = _env(cfg.to_stdio_params())
    assert "GITHUB_TOKEN" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert env.get("PATH") == "/usr/bin"


def test_case_insensitive_match(monkeypatch):
    monkeypatch.setenv("github_token", "s3cret")
    cfg = ServerConfig(name="s", command="x", scrub_secret_env=["*_TOKEN"])
    assert "github_token" not in _env(cfg.to_stdio_params())


def test_explicit_server_env_is_a_grant_not_scrubbed(monkeypatch):
    # A secret the operator DELIBERATELY hands this server in .mcp.json 'env'
    # must still reach it, even if it matches a scrub pattern — only the
    # inherited os.environ is scrubbed.
    monkeypatch.setenv("GITHUB_TOKEN", "inherited-leak")
    cfg = ServerConfig(
        name="s", command="x",
        env={"GITHUB_TOKEN": "granted-on-purpose"},
        scrub_secret_env=["*_TOKEN"],
    )
    env = _env(cfg.to_stdio_params())
    # The explicit grant wins and is present with the operator's value.
    assert env["GITHUB_TOKEN"] == "granted-on-purpose"


def test_manager_threads_patterns_into_serverconfig():
    # The manager stores the patterns and stamps them onto every ServerConfig
    # it builds (so all connect paths are covered at the to_stdio_params choke).
    mgr = MCPClientManager(scrub_secret_env=["*_TOKEN"])
    assert mgr._scrub_secret_env == ("*_TOKEN",)


def test_manager_default_is_empty():
    mgr = MCPClientManager()
    assert mgr._scrub_secret_env == ()
