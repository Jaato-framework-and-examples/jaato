"""Every file a profile's completion gate names must resolve — checked here.

Nothing checked them.  An unresolvable `completion_payload_schema` or
`completion_processors[].script` produced one WARNING in the runner log and no
other signal: with no schema the gate is dropped, `signal_completion` is hidden
from the model entirely, the framework spends its nudge budget re-prompting an
agent hunting for a tool that will never appear, and the driver is handed
`None` by a session that looked like it ran.

The commonest way to get there is spelling the prefix the resolver adds.  Every
relative reference is joined onto the CONFIG ROOT — `<workspace>/.jaato` by
default — so `.jaato/completion_schemas/x.json` resolves to
`<ws>/.jaato/.jaato/completion_schemas/x.json`.  It is an easy mistake because
every other path an author writes is spelled from the workspace root, and it is
reported under its own code so the message does not send them hunting on disk
for a file that is exactly where they put it.
"""

import json
import types

import pytest

from shared.scaffold.validate import _check_completion_assets, _redundant_prefix


def _profile(name="p", schema=None, scripts=()):
    return types.SimpleNamespace(
        name=name,
        completion_payload_schema=schema,
        completion_processors=[types.SimpleNamespace(script=s) for s in scripts],
    )


def _run(tmp_path, profile):
    out = []
    _check_completion_assets({profile.name: profile}, tmp_path,
                             str(tmp_path / ".jaato"), out)
    return out


@pytest.fixture
def ws(tmp_path):
    cr = tmp_path / ".jaato"
    (cr / "completion_schemas").mkdir(parents=True)
    (cr / "completion_schemas" / "step.json").write_text(json.dumps({"type": "object"}))
    (cr / "scripts" / "processors").mkdir(parents=True)
    (cr / "scripts" / "processors" / "acc.py").write_text("def validate(p, c):\n    return []\n")
    return tmp_path


# ------------------------------------------------------------------ clean

def test_a_correct_gate_reports_nothing(ws):
    assert _run(ws, _profile(schema="completion_schemas/step.json",
                             scripts=["scripts/processors/acc.py"])) == []


def test_an_inline_schema_is_not_a_path(ws):
    assert _run(ws, _profile(schema={"type": "object"})) == []


def test_no_gate_at_all_reports_nothing(ws):
    assert _run(ws, _profile()) == []


# ------------------------------------------------------- the .jaato/ prefix

@pytest.mark.parametrize("prefix", [".jaato/", "./.jaato/"])
def test_the_redundant_prefix_is_its_own_finding(ws, prefix):
    d = _run(ws, _profile(schema=f"{prefix}completion_schemas/step.json"))
    assert len(d) == 1
    assert d[0].code == "redundant_config_root_prefix"
    assert d[0].severity == "error"
    # It names the fix, not just the problem.
    assert "'completion_schemas/step.json'" in d[0].message


def test_the_prefix_is_flagged_on_a_processor_script_too(ws):
    d = _run(ws, _profile(scripts=[".jaato/scripts/processors/acc.py"]))
    assert [x.code for x in d] == ["redundant_config_root_prefix"]
    assert d[0].where == "completion_processors[0].script"


def test_the_prefix_finding_says_what_it_costs_for_THAT_asset(ws):
    """A missing schema hides the tool; a missing script blocks completions.
    Telling an author the wrong one sends them to the wrong place."""
    schema = _run(ws, _profile(schema=".jaato/completion_schemas/step.json"))[0]
    script = _run(ws, _profile(scripts=[".jaato/scripts/processors/acc.py"]))[0]
    assert "HIDDEN" in schema.message
    assert "HIDDEN" not in script.message and "blocks every completion" in script.message


def test_prefix_helper_leaves_ordinary_paths_alone():
    assert _redundant_prefix("completion_schemas/x.json") is None
    assert _redundant_prefix("scripts/processors/acc.py") is None
    assert _redundant_prefix(".jaatoish/x.json") is None      # not the prefix


# --------------------------------------------------------------- missing

def test_a_path_that_resolves_nowhere_is_an_error(ws):
    d = _run(ws, _profile(schema="completion_schemas/nope.json"))
    assert [x.code for x in d] == ["completion_asset_missing"]
    assert d[0].severity == "error"


def test_a_missing_processor_script_is_flagged(ws):
    d = _run(ws, _profile(scripts=["scripts/processors/gone.py"]))
    assert [x.code for x in d] == ["completion_asset_missing"]


def test_each_processor_entry_is_checked_by_index(ws):
    d = _run(ws, _profile(scripts=["scripts/processors/acc.py",
                                   "scripts/processors/gone.py"]))
    assert len(d) == 1
    assert d[0].where == "completion_processors[1].script"


def test_an_absolute_path_is_left_to_the_author(ws):
    """It bypasses the tier chain entirely — judging it here would be a guess
    about a machine the validator is not running on."""
    assert _run(ws, _profile(schema="/opt/schemas/step.json")) == []


def test_findings_name_the_profile(ws):
    d = _run(ws, _profile(name="documentalista",
                          schema="completion_schemas/nope.json"))
    assert d[0].profile == "documentalista"
