"""Two directories the framework reads and `explain` never named.

``.jaato/agents/`` and ``.jaato/services/`` are both load-bearing and both
were absent from every ``explain`` scope, with the same consequence in each
case: a session that could not find the supported path took an unsupported
one and nothing corrected it.

  agents    ``explain profile`` listed ``system_instructions`` — marked
            DEPRECATED, and the only instruction-shaped key on the page.  An
            author who never found the agents directory used it, and it works.

  services  ``service_connector`` caches OpenAPI specs and auth under
            ``.jaato/services/`` and calls them by alias.  Unnamed, the
            fallback is a raw URL with the base URL, auth header and
            pagination re-derived by hand on every call.

The search orders here are READ from the runtime's own helpers, never
restated — a documented order that disagrees with the loaded one is worse
than no documentation.
"""

from pathlib import Path

import pytest

from shared.plugins.subagent.config import AGENT_FILE_FORMS, agent_search_dirs
from shared.scaffold.explain import agents, services
from shared.scaffold.__main__ import _WORKSPACE_SCOPES, _SCOPES_HELP


# ------------------------------------------------------------------- wiring

@pytest.mark.parametrize("scope", ["agents", "services", "sets"])
def test_scope_is_reachable_and_advertised(scope):
    """A scope the help does not name is a scope nobody types (jaato #716)."""
    assert scope in _WORKSPACE_SCOPES
    assert scope in _SCOPES_HELP


# ------------------------------------------------------------------- agents

def test_agents_reports_the_runtime_search_order_verbatim(tmp_path):
    data, text = agents(str(tmp_path))
    expected = [str(d) for d in agent_search_dirs(str(tmp_path))]
    assert data["search_order"] == expected
    for d in expected:
        assert d in text


def test_agents_lists_every_accepted_filename_form(tmp_path):
    _, text = agents(str(tmp_path))
    for form in AGENT_FILE_FORMS:
        assert form in text


def test_agents_names_the_deprecated_alternative_and_the_replacement(tmp_path):
    """The whole point: steer an author off `system_instructions:`."""
    _, text = agents(str(tmp_path))
    assert "system_instructions" in text and "DEPRECATED" in text
    assert "default_agent" in text
    assert "agent='researcher'" in text or 'agent="researcher"' in text


def test_agents_repeats_the_credential_rule(tmp_path):
    """A rendered persona is persisted, so an agent_param secret is on disk."""
    _, text = agents(str(tmp_path))
    assert "NEVER PASS A CREDENTIAL" in text
    assert "pass://" in text


def test_agents_discovers_what_is_actually_on_disk(tmp_path):
    d = tmp_path / ".jaato" / "agents"
    d.mkdir(parents=True)
    (d / "researcher.md").write_text("You research things.\n")
    (d / "writer.md").write_text("You write things.\n")
    data, text = agents(str(tmp_path))
    assert {a["name"] for a in data["discovered"]} == {"researcher", "writer"}
    assert "researcher" in text and "writer" in text


def test_agents_on_an_empty_workspace_says_so_rather_than_failing(tmp_path):
    data, text = agents(str(tmp_path))
    assert data["discovered"] == []
    assert "no agent files found" in text


# ----------------------------------------------------------------- services

def test_services_documents_the_three_step_flow(tmp_path):
    _, text = services(str(tmp_path))
    for call in ("discover_service", "configure_service_auth", "call_service"):
        assert call in text


def test_services_reports_both_tiers_and_which_one_is_writable(tmp_path):
    data, text = services(str(tmp_path))
    tiers = {t["tier"]: t for t in data["tiers"]}
    assert set(tiers) == {"workspace", "user"}
    assert tiers["workspace"]["path"] == str(
        Path(tmp_path).resolve() / ".jaato" / "services")
    assert data["writable_tier"] == "workspace"
    assert "writable" in text and "workspace shadows home" in text


def test_services_says_when_an_alias_beats_a_raw_url(tmp_path):
    """The decision the topic exists to inform."""
    _, text = services(str(tmp_path))
    assert "MANAGED ALIAS vs RAW URL" in text
    assert "needs auth" in text and "paginated" in text


def test_services_lists_discovered_and_hand_written_services(tmp_path):
    root = tmp_path / ".jaato" / "services"
    (root / "_discovered").mkdir(parents=True)
    (root / "_discovered" / "gitlab.yaml").write_text("openapi: 3.0.0\n")
    (root / "internal").mkdir()
    (root / "internal" / "_service.yaml").write_text("name: internal\n")
    data, text = services(str(tmp_path))
    kinds = {k["service"]: k["kind"] for k in data["known"]}
    assert kinds == {"gitlab": "discovered", "internal": "defined"}
    assert "gitlab" in text and "internal" in text


def test_services_on_an_empty_workspace_says_so(tmp_path):
    data, text = services(str(tmp_path))
    assert data["known"] == []
    assert "nothing has been discovered yet" in text
