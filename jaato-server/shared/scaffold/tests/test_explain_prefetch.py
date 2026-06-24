"""scaffold explain surfaces the prefetch-script capability (Daniel-flagged gap).

`scaffold explain` is the self-configuration discovery path; before this it never
mentioned prefetch ({{!py:...}} dynamic-instruction expansion), so an author/agent
couldn't learn the deterministic per-agent session-start capability exists.
"""
from shared.scaffold import explain


def test_prefetch_documents_the_capability():
    _data, text = explain.prefetch()
    assert "{{!py:" in text and "{{!py?:" in text          # both directive forms
    assert "def render(context, args)" in text             # the script contract
    assert ".jaato/agents/" in text and "scripts/" in text  # where it lives
    assert "MANDATORY" in text and "OPTIONAL" in text       # failure semantics
    assert "agent_params" in text and "registry" in text    # context attrs
    assert "prefetch_kyc_aml" in text                       # worked example


def test_overview_surfaces_prefetch():
    # The discovery path: `explain` (no args) must list `explain prefetch`.
    _data, text = explain.overview()
    assert "explain prefetch" in text
