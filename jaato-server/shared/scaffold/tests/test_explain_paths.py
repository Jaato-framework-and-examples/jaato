"""jaato-scaffold explain paths: surfaces the daemon-global ~/.jaato vs
per-session workspace/config_root isolation model, to stop the common
'override $HOME to isolate a run' mistake."""
from shared.scaffold import explain


def test_paths_renders_both_layers_and_the_home_warning():
    data, text = explain.paths()
    # daemon-global layer named + the don't-override-HOME guidance
    assert "DAEMON-GLOBAL" in text and "~/.jaato" in text
    assert "Do NOT override $HOME" in text
    assert "reactors" in text and "auth" in text          # what lives there
    # per-session layer = workspace + config_root, not HOME
    assert "PER-SESSION" in text and "config_root" in text
    assert "JAATO_WORKSPACE_ROOT" in text
    # structured data mirrors the text for --json consumers
    assert data["daemon_global"]["note"] == "do NOT override $HOME to isolate a run"
    assert "workspace_root_env" in data["per_session"]


def test_paths_is_listed_in_overview():
    _data, text = explain.overview()
    assert "jaato-scaffold explain paths" in text
