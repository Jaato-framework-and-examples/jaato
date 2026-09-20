"""The ``regulatory:`` profile block -- what an application declares under the EU AI Act.

Regulation (EU) 2024/1689 asks the PROVIDER of an AI system to determine
and document whether it is high-risk (Art. 6(4)), to disclose that a
person is talking to an AI (Art. 50(1)) and, for a high-risk system, to
keep records, provide for human oversight and be robust (Arts. 12, 14,
15).  None of that is derivable from a profile's plugins or model: it is
a fact only the author knows, so it is DECLARED in a typed block and
never inferred.  ``docs/design/eu-ai-act.md`` §4.1 is the design.

What this module pins:

A. the block parses, refuses what it should, and round-trips through a
   snapshot and the isolated-runner payload;
B. inheritance is child-replaces per field and MOST-RESTRICTIVE-WINS on
   ``risk_class`` -- a child cannot un-declare a base's ``high``;
C. ``jaato-scaffold validate`` reports ``disclosure_absent`` for a
   persona-bound profile that declares nothing, escalates the named
   warnings to errors under ``risk_class: high``, and adds the four
   high-risk-only findings -- and changes nothing for a profile that
   declares no class.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from shared.plugins.subagent.config import (
    PROFILE_FILE_KEYS,
    RecordKeepingConfig,
    RISK_CLASSES,
    RegulatoryProfileConfig,
    SubagentProfile,
    merge_regulatory,
    parse_regulatory_block,
    profile_from_snapshot,
    profile_to_snapshot,
    resolve_profiles,
    validate_profile as config_validate_profile,
)
from shared.scaffold import introspect
from shared.scaffold.validate import (
    HIGH_RISK_ESCALATED_CODES,
    Diagnostic,
    _check_regulatory,
    _escalate_for_risk_class,
    validate_profile,
)
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

_CONFIG = "jaato-server/shared/plugins/subagent/config.py"
_VALIDATE = "jaato-server/shared/scaffold/validate.py"

REVERSIONS = [
    Reversion(
        target=_CONFIG,
        find="    risk = max(ranked, key=_RISK_RANK.__getitem__) if ranked else None",
        replace="    risk = ranked[0] if ranked else None",
        because=(
            "a child declaring `minimal` under a base that says `high` "
            "would un-make the base author's determination by contradiction"
        ),
        test="test_a_child_cannot_declare_itself_below_its_base",
    ),
    Reversion(
        target=_VALIDATE,
        find='            d.severity = "error"\n'
             '            d.message += "  (error because regulatory.risk_class is high)"',
        replace='            d.message += "  (error because regulatory.risk_class is high)"',
        because=(
            "under risk_class: high the named warnings must become errors, "
            "or a high-risk profile with no ceiling validates with exit 0"
        ),
        test="test_named_warnings_become_errors_under_high",
    ),
    Reversion(
        target=_VALIDATE,
        # Anchored on the WHOLE call, `env_keys` included: the argument is
        # what carries the workspace `.env`'s own `LEDGER_PATH` into
        # `disclosure_unrecorded` (#1157), so a call site that lost it would
        # be a check wired but under-informed -- the shape one line down.
        find="    _check_regulatory(profile, add, env_keys=env_keys)\n",
        replace="",
        because=(
            "the check must be WIRED into validate_profile, not merely "
            "defined -- a defined-but-uncalled check is the #735 shape"
        ),
        test="test_the_checks_are_wired_into_validate_profile",
    ),
]


FULL = {
    "intended_purpose": "Pre-screens job applications for a recruiter's review.",
    "risk_class": "high",
    "annex_iii": "4a",
    "provider": {"name": "Acme Talent GmbH", "contact": "compliance@acme.example"},
    "interacts_with_persons": True,
    "disclosure_text": "You are talking to Acme's screening assistant, an AI system.",
}


# ---------------------------------------------------------------- A. parsing

def test_a_full_block_parses_and_round_trips_its_file_shape():
    reg = RegulatoryProfileConfig.from_dict(FULL)
    assert reg.is_high_risk
    assert reg.provider_name == "Acme Talent GmbH"
    assert reg.provider_contact == "compliance@acme.example"
    assert reg.interacts_with_persons is True
    assert reg.to_dict() == FULL


def test_an_absent_or_empty_block_is_none():
    assert parse_regulatory_block({}) is None
    assert parse_regulatory_block({"regulatory": None}) is None
    assert parse_regulatory_block({"regulatory": {}}) is None


def test_an_undeclared_class_validates_as_minimal_and_says_it_is_undeclared():
    reg = RegulatoryProfileConfig.from_dict({"intended_purpose": "x"})
    assert reg.risk_class is None
    assert reg.effective_risk_class == RISK_CLASSES[0] == "minimal"
    assert not reg.is_high_risk


@pytest.mark.parametrize("bad, fragment", [
    ({"risk_class": "severe"}, "risk_class"),
    ({"risk_klass": "high"}, "unknown key"),
    ({"interacts_with_persons": "yes"}, "boolean"),
    ({"provider": "Acme"}, "provider must be a mapping"),
    ({"provider": {"name": "Acme", "phone": "1"}}, "provider: unknown key"),
    ({"intended_purpose": ""}, "non-empty string"),
    ("high", "must be a mapping"),
])
def test_a_malformed_block_is_refused_by_name(bad, fragment):
    with pytest.raises(ValueError) as exc:
        RegulatoryProfileConfig.from_dict(bad)
    assert fragment in str(exc.value)


def test_config_validate_profile_reports_the_parsers_refusal():
    ok, errors, _ = config_validate_profile({
        "name": "x", "description": "d", "plugins": [],
        "regulatory": {"risk_class": "severe"},
    })
    assert not ok
    assert any("risk_class" in e for e in errors)


def test_regulatory_is_a_declared_file_key():
    assert "regulatory" in PROFILE_FILE_KEYS


def test_the_block_survives_a_snapshot_round_trip():
    profile = SubagentProfile(
        name="p", description="d", plugins=[],
        regulatory=RegulatoryProfileConfig.from_dict(FULL),
    )
    snap = profile_to_snapshot(profile)
    assert snap["regulatory"] == FULL
    assert profile_from_snapshot(snap).regulatory == profile.regulatory


def test_an_old_snapshot_without_the_block_still_revives():
    snap = profile_to_snapshot(SubagentProfile(name="p", description="d", plugins=[]))
    snap.pop("regulatory")
    assert profile_from_snapshot(snap).regulatory is None


def test_the_isolated_runner_payload_carries_and_checks_it():
    from server.runner_rpc_handlers.profile_payload_schema import (
        PROFILE_PAYLOAD_ALLOWED_KEYS, validate_profile_payload,
    )
    assert "regulatory" in PROFILE_PAYLOAD_ALLOWED_KEYS
    validate_profile_payload({"name": "p", "plugins": [], "regulatory": FULL})
    with pytest.raises(ValueError) as exc:
        validate_profile_payload(
            {"name": "p", "plugins": [], "regulatory": {"risk_class": "severe"}})
    assert "profile_payload.regulatory" in str(exc.value)


# ------------------------------------------------------------ B. inheritance

def _prof(name, reg=None, inherits=None):
    return SubagentProfile(
        name=name, description=name, plugins=[], inherits=inherits,
        regulatory=RegulatoryProfileConfig.from_dict(reg) if reg else None,
    )


def test_a_child_inherits_the_base_providers_block_field_by_field():
    resolved, errors = resolve_profiles({
        "base": _prof("base", {"provider": {"name": "Acme"}, "risk_class": "limited"}),
        "child": _prof("child", {"intended_purpose": "screen"}, inherits=["base"]),
    })
    assert not errors
    reg = resolved["child"].regulatory
    assert reg.provider_name == "Acme"
    assert reg.intended_purpose == "screen"
    assert reg.risk_class == "limited"


def test_a_child_cannot_declare_itself_below_its_base():
    resolved, errors = resolve_profiles({
        "base": _prof("base", {"risk_class": "high"}),
        "child": _prof("child", {"risk_class": "minimal"}, inherits=["base"]),
    })
    assert not errors
    assert resolved["child"].regulatory.risk_class == "high"


def test_a_child_may_raise_the_class_above_its_base():
    resolved, _ = resolve_profiles({
        "base": _prof("base", {"risk_class": "minimal"}),
        "child": _prof("child", {"risk_class": "high"}, inherits=["base"]),
    })
    assert resolved["child"].regulatory.risk_class == "high"


def test_a_child_scalar_replaces_the_parents():
    merged = merge_regulatory(
        _prof("c", {"provider": {"name": "Child Co"}}),
        [_prof("p", {"provider": {"name": "Parent Co", "contact": "x@p"}})],
    )
    assert merged.provider_name == "Child Co"
    assert merged.provider_contact == "x@p"


def test_no_layer_declaring_it_merges_to_none():
    assert merge_regulatory(_prof("c"), [_prof("p")]) is None


# --------------------------------------------------------------- C. validate

def _run(profile):
    out: list = []

    def add(severity, code, message, where=None):
        out.append(Diagnostic(severity, code, message, where=where))

    _check_regulatory(profile, add)
    return out


def _ns(reg=None, **kw):
    base = dict(default_agent=None, system_instructions=None, plugins=[],
                plugin_configs={}, trace=None, env={},
                suppress_base_instructions=frozenset())
    base.update(kw)
    return SimpleNamespace(regulatory=reg, **base)


def test_a_persona_bound_profile_declaring_nothing_is_told_to_disclose():
    found = _run(_ns(default_agent="screener"))
    assert [d.code for d in found] == ["disclosure_absent"]
    assert found[0].severity == "warn"


def test_an_explicit_false_is_a_declaration_and_is_silent():
    reg = RegulatoryProfileConfig.from_dict({"interacts_with_persons": False})
    assert _run(_ns(reg, default_agent="screener")) == []


def test_an_abstract_base_owes_no_disclosure():
    assert _run(_ns()) == []


def test_no_class_declared_changes_nothing():
    out = [Diagnostic("warn", "budget_control_absent", "m")]
    _escalate_for_risk_class(_ns(), out)
    assert out[0].severity == "warn"
    reg = RegulatoryProfileConfig.from_dict({"risk_class": "limited"})
    _escalate_for_risk_class(_ns(reg), out)
    assert out[0].severity == "warn"


def test_named_warnings_become_errors_under_high():
    reg = RegulatoryProfileConfig.from_dict({"risk_class": "high"})
    out = [Diagnostic("warn", code, "m") for code in sorted(HIGH_RISK_ESCALATED_CODES)]
    out.append(Diagnostic("warn", "discovery_gated_tools", "m"))
    _escalate_for_risk_class(_ns(reg), out)
    assert all(d.severity == "error" for d in out if d.code in HIGH_RISK_ESCALATED_CODES)
    assert out[-1].severity == "warn", "a code outside the set keeps its severity"


def test_the_high_risk_only_findings_fire_and_name_the_remedy():
    reg = RegulatoryProfileConfig.from_dict({"risk_class": "high"})
    found = _run(_ns(reg, plugins=["interactive_shell"]))
    codes = {d.code for d in found}
    assert codes == {
        "high_risk_without_intended_purpose",
        "high_risk_without_oversight_policy",
        # The record is WRITTEN (#1109) ...
        "high_risk_without_record_keeping",
        # ... and KEPT (#1119).  The pair: one says nothing records, the
        # other says nothing keeps what records.  Neither is useful alone.
        "high_risk_without_retention",
        "high_risk_shell_unconfined",
    }
    # And the sixth, when the disclosure piece is suppressed by name.
    dropped = _run(_ns(reg, suppress_base_instructions=frozenset({"disclosure"})))
    assert "high_risk_disclosure_suppressed" in {d.code for d in dropped}
    assert all(d.severity == "error" for d in found)


def test_each_high_risk_finding_is_satisfied_by_the_mechanism_it_names():
    reg = RegulatoryProfileConfig.from_dict(FULL)
    found = _run(_ns(
        reg, plugins=["interactive_shell"],
        plugin_configs={
            "permission": {"policy": {"defaultPolicy": "ask"}},
            "interactive_shell": {"require_confinement": True},
        },
        trace=SimpleNamespace(session_log=".jaato/logs/trace.jsonl",
                              provider_log=None,
                              # interacts_with_persons is True, so the
                              # announcement must reach a FILE (#1157).
                              ledger=".jaato/logs/ledger.jsonl"),
        record_keeping=RecordKeepingConfig(retention_days=180),
    ))
    assert found == []


def test_the_checks_are_wired_into_validate_profile():
    profile = SubagentProfile(
        name="screener", description="d", plugins=[], default_agent="screener",
        regulatory=RegulatoryProfileConfig.from_dict({"risk_class": "high"}),
    )
    out = validate_profile(
        profile, providers=introspect.providers(), plugins=introspect.plugins(),
        gc_names=[],
    )
    by_code = {d.code: d for d in out}
    assert by_code["high_risk_without_oversight_policy"].severity == "error"
    # An escalated one: a warning everywhere else, an error here.
    assert by_code["budget_control_absent"].severity == "error"
    assert "regulatory.risk_class is high" in by_code["budget_control_absent"].message
    # The persona is bound and interacts_with_persons is undeclared, so
    # disclosure_absent fires -- and it is in the escalated set.
    assert by_code["disclosure_absent"].severity == "error"
