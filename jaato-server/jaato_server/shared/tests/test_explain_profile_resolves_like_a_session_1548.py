"""``explain profile <name>`` reports the profile a session would get (#1548).

It used to find the profile with its own lookup: an ``rglob`` over
``.jaato/profiles/`` returning the first file whose stem matched, in sorted
order, read as a raw dict.  That second resolver disagreed with the one
sessions use (``discover_profiles``) in two ways:

* a set's subdirectory (``profiles/fast/worker.yaml``) sorts before the
  top-level file, so it reported a set nobody selected, and it never read
  ``JAATO_PROFILE_SET``;
* it ignored ``inherits:``, so a ``suppress_base_instructions`` inherited
  from a base profile was invisible and the "inherited on EVERY turn"
  figure counted layers the resolved profile drops.

It now resolves through ``discover_profiles`` with the workspace's profile
set (``--set``, else the workspace ``.env``), reports the file that won
precedence (``ProfileDiscoveryResult.sources``) and reads the suppression off
the merged profile.
"""

from __future__ import annotations

import pytest

from jaato_server.shared.scaffold import explain
from jaato_server.shared.scaffold.introspection_verbs import render_topic
from jaato_server.shared.tests.reversion import Reversion

REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/explain.py",
        find="    pset = profile_set or workspace_profile_set(str(ws))\n",
        replace="    pset = profile_set\n",
        test="test_the_workspace_env_selects_the_set",
        because="the workspace .env's JAATO_PROFILE_SET is ignored, so the "
                "report describes the default set while sessions run another",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/explain.py",
        find='        getattr(prof, "suppress_base_instructions", None))\n',
        replace="        None)\n",
        test="test_an_inherited_suppression_counts",
        because="the suppression the resolved profile carries is not read, "
                "so inherited layers it drops are reported as a cost",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/plugins/subagent/config.py",
        find="        sources[name] = str(file_path)\n",
        replace="        pass\n",
        test="test_with_no_set_it_reports_the_top_level_file",
        because="discovery no longer records which file won, so the report "
                "cannot name the file the session's profile came from",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/introspection_verbs.py",
        find="        return spec.render_named(name, ws, profile_set=profile_set)\n",
        replace="        return spec.render_named(name, ws)\n",
        test="test_set_reaches_the_named_renderer",
        because="`explain profile <name> --set S` answers about the .env's "
                "set instead of the one the caller asked for",
    ),
]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """No developer HOME, premium install or ambient profile set."""
    from jaato_server.shared.plugins.subagent import config as _cfg
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))
    monkeypatch.delenv("JAATO_PROFILE_SET", raising=False)
    monkeypatch.setattr(_cfg, "_discover_premium_profiles", lambda: {})
    monkeypatch.setattr(explain, "_instruction_search_order",
                        lambda ws: [("workspace", ws / ".jaato" / "instructions")])


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _issue_layout(ws, base_extra=""):
    """The reproduction from #1548, plus a 4,000-byte instruction layer."""
    p = ws / ".jaato" / "profiles"
    _write(p / "_base.yaml",
           "name: _base\ndescription: b\nplugins: [cli, todo]\n" + base_extra)
    _write(p / "worker.yaml",
           "name: worker\ndescription: w\nplugins: []\n"
           "inherits: [_base]\nprovider: anthropic\n")
    _write(p / "fast" / "worker.yaml",
           "name: worker\ndescription: w\nplugins: []\n"
           "inherits: [_base]\nprovider: openrouter\n")
    _write(ws / ".jaato" / "instructions" / "00-base.md", "x" * 4_000)
    return ws


def test_with_no_set_it_reports_the_top_level_file(tmp_path):
    """The sorted rglob picked ``fast/worker.yaml``; no set was selected."""
    ws = _issue_layout(tmp_path)
    data, text = explain.profile_cost("worker", str(ws))
    assert data["found"] is True
    assert data["profile_set"] is None
    expected = str((ws / ".jaato" / "profiles" / "worker.yaml").resolve())
    assert data["profile_file"] == expected
    assert expected in text


def test_the_workspace_env_selects_the_set(tmp_path):
    ws = _issue_layout(tmp_path)
    _write(ws / ".env", "JAATO_PROFILE_SET=fast\n")
    data, text = explain.profile_cost("worker", str(ws))
    assert data["profile_set"] == "fast"
    assert data["profile_file"].endswith("/profiles/fast/worker.yaml")
    assert "profile set 'fast'" in text


def test_set_reaches_the_named_renderer(tmp_path):
    """``--set`` outranks the .env, as it does for ``validate``."""
    ws = _issue_layout(tmp_path)
    _write(ws / ".env", "JAATO_PROFILE_SET=nope\n")
    ok, data, _text, error = render_topic("profile", "worker", str(ws),
                                          profile_set="fast")
    assert ok, error
    assert data["profile_set"] == "fast"
    assert data["profile_file"].endswith("/profiles/fast/worker.yaml")


def test_an_inherited_suppression_counts(tmp_path):
    """The child declares nothing; its base drops the disk layer."""
    ws = _issue_layout(tmp_path,
                       base_extra="suppress_base_instructions: {disk: true}\n")
    data, text = explain.profile_cost("worker", str(ws))
    assert data["disk_layer_suppressed"] is True
    assert data["inherited_bytes"] == 0
    assert "SUPPRESSED" in text


def test_without_suppression_the_layer_is_a_cost(tmp_path):
    """The control for the case above: same layout, nothing suppressed."""
    ws = _issue_layout(tmp_path)
    data, _ = explain.profile_cost("worker", str(ws))
    assert data["disk_layer_suppressed"] is False
    assert data["inherited_bytes"] == 4_000


def test_a_profile_discovery_refuses_is_reported_with_the_refusal(tmp_path):
    """Not replaced by another file of the same name, and not 'not found'."""
    p = tmp_path / ".jaato" / "profiles"
    _write(p / "worker.yaml",
           "name: worker\ndescription: w\nplugins: []\ninherits: [missing]\n")
    data, text = explain.profile_cost("worker", str(tmp_path))
    assert data["found"] is False
    assert data.get("refused")
    assert "does not resolve" in text


def test_the_second_resolver_is_gone():
    assert not hasattr(explain, "_find_profile_file")
