"""Active JAATO_* env-override reporting (startup line + --status)."""

from server.env_report import active_overrides, format_overrides


def test_only_jaato_vars_and_sorted():
    env = {"PATH": "/bin", "HOME": "/h", "JAATO_B": "2", "JAATO_A": "1"}
    assert active_overrides(env) == [("JAATO_A", "1"), ("JAATO_B", "2")]


def test_secrets_redacted_by_name():
    env = {
        "JAATO_WS_TOKEN": "supersecret",
        "JAATO_NIM_API_KEY": "nvapi-xxx",
        "JAATO_EGRESS_NFT_ENFORCE": "1",
        "JAATO_CGROUPS_ROOT": "/sys/fs/cgroup/jaato",
    }
    got = dict(active_overrides(env))
    assert got["JAATO_WS_TOKEN"] == "<set>"
    assert got["JAATO_NIM_API_KEY"] == "<set>"
    # non-secret knobs shown verbatim
    assert got["JAATO_EGRESS_NFT_ENFORCE"] == "1"
    assert got["JAATO_CGROUPS_ROOT"] == "/sys/fs/cgroup/jaato"


def test_format_none_when_no_overrides():
    line = format_overrides({"PATH": "/bin"})
    assert "none (all defaults" in line
    assert "explain env" in line   # points to the full catalog


def test_format_lists_overrides():
    line = format_overrides({"JAATO_EGRESS_NFT_ENFORCE": "1",
                             "JAATO_WS_TOKEN": "s"})
    assert "JAATO_EGRESS_NFT_ENFORCE=1" in line
    assert "JAATO_WS_TOKEN=<set>" in line
    assert "supersecret" not in line and "s" != line  # token value not leaked
