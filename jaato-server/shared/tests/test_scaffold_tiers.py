"""Scaffold tooling coverage for model_tiers / V2 cross-provider tiers:
explain renders + introspects the real tier names; validate flags tier typos +
cross-provider tiers naming an uninstalled provider (now that model_tiers
survives the inherits/set merge — see config._merge_profiles).
"""

from shared.scaffold import explain
from shared.scaffold import validate as V


def test_explain_tiers_introspects_tier_names():
    data, text = explain.tiers()
    assert "vision" in data["tier_names"]          # from VALID_TIER_NAMES
    assert "cross_provider" in data                # V2 documented
    assert "enter_tier" in text


def test_validate_flags_bad_model_tiers(tmp_path):
    pdir = tmp_path / ".jaato" / "profiles" / "setT"
    pdir.mkdir(parents=True)
    (tmp_path / ".jaato" / "profiles" / "_base_t.yaml").write_text(
        "name: _base_t\ndescription: b\nplugins: []\n")
    (pdir / "t.yaml").write_text(
        "name: t\ninherits: [_base_t]\nplugins: []\n"
        "model_tiers:\n"
        "  executor: {model: m, provider: nebius}\n"
        "  visionn: {model: v, provider: openrouter}\n"     # tier-name typo
        "  planner: {model: p, provider: notaprovider}\n"   # unknown provider
        "  initial: executor\n  fallback: executor\n")
    diags = V.validate_workspace(str(tmp_path), profile_set="setT", only="t")
    codes = {d.code for d in diags}
    assert "unknown_tier" in codes        # visionn
    assert "unknown_provider" in codes    # notaprovider


def test_validate_accepts_cross_provider_tiers(tmp_path):
    # V2: a vision tier on a DIFFERENT (real) provider is valid.
    pdir = tmp_path / ".jaato" / "profiles" / "setOK"
    pdir.mkdir(parents=True)
    (tmp_path / ".jaato" / "profiles" / "_base_ok.yaml").write_text(
        "name: _base_ok\ndescription: b\nplugins: []\n")
    (pdir / "ok.yaml").write_text(
        "name: ok\ninherits: [_base_ok]\nplugins: []\n"
        "model_tiers:\n"
        "  executor: {model: glm-4.6, provider: zhipuai}\n"
        "  vision: {model: google/gemini-2.5-flash-lite, provider: openrouter}\n"
        "  initial: executor\n  fallback: executor\n")
    diags = V.validate_workspace(str(tmp_path), profile_set="setOK", only="ok")
    tier_codes = {d.code for d in diags
                  if d.code in ("unknown_tier", "unknown_provider")}
    assert tier_codes == set()   # cross-provider accepted, no tier findings


def test_explain_tiers_documents_the_description_key():
    data, text = explain.tiers()
    assert "description" in data
    assert "description" in data["shape"]
    assert "DESCRIPTION" in text


def test_validate_flags_a_malformed_tier_description(tmp_path):
    # A description reaches the MODEL (it becomes the tier's bullet in the
    # enter_tier schema), so a malformed one is worth catching statically.
    pdir = tmp_path / ".jaato" / "profiles" / "setD"
    pdir.mkdir(parents=True)
    (tmp_path / ".jaato" / "profiles" / "_base_d.yaml").write_text(
        "name: _base_d\ndescription: b\nplugins: []\n")
    (pdir / "d.yaml").write_text(
        "name: d\ninherits: [_base_d]\nplugins: []\n"
        "model_tiers:\n"
        "  executor: {model: m, description: 'grind through edits'}\n"  # ok
        "  planner:  {model: p, description: 42}\n"                    # bad
        "  initial: executor\n  fallback: executor\n")
    diags = V.validate_workspace(str(tmp_path), profile_set="setD", only="d")
    bad = [d for d in diags if d.code == "invalid_tier_description"]
    assert len(bad) == 1
    assert "planner" in bad[0].where


def test_explain_tiers_documents_modality_roles():
    data, text = explain.tiers()
    assert "image" in data["modalities"]
    assert "text" not in data["modalities"]      # never declarable on a tier
    assert "modalities" in data["shape"]
    assert "MODALITY ROLES" in text


def test_validate_flags_bad_tier_modalities(tmp_path):
    # A typo here fails silently at runtime in the worst way: the gate finds
    # no tier for an image and tells the agent none exists, while the profile
    # plainly declares one.
    pdir = tmp_path / ".jaato" / "profiles" / "setM"
    pdir.mkdir(parents=True)
    (tmp_path / ".jaato" / "profiles" / "_base_m.yaml").write_text(
        "name: _base_m\ndescription: b\nplugins: []\n")
    (pdir / "m.yaml").write_text(
        "name: m\ninherits: [_base_m]\nplugins: []\n"
        "model_tiers:\n"
        "  executor: {model: e}\n"
        "  vision:   {model: v, modalities: [image]}\n"        # ok
        "  planner:  {model: p, modalities: [smell]}\n"        # unknown
        "  dispatcher: {model: d, modalities: image}\n"        # not a list
        "  initial: executor\n  fallback: executor\n")
    diags = V.validate_workspace(str(tmp_path), profile_set="setM", only="m")
    bad = [d for d in diags if d.code == "invalid_tier_modalities"]
    assert {d.where for d in bad} == {
        "model_tiers.planner.modalities", "model_tiers.dispatcher.modalities"}


def test_validate_flags_text_modality_with_an_explanation(tmp_path):
    pdir = tmp_path / ".jaato" / "profiles" / "setT2"
    pdir.mkdir(parents=True)
    (tmp_path / ".jaato" / "profiles" / "_base_t2.yaml").write_text(
        "name: _base_t2\ndescription: b\nplugins: []\n")
    (pdir / "t2.yaml").write_text(
        "name: t2\ninherits: [_base_t2]\nplugins: []\n"
        "model_tiers:\n"
        "  executor: {model: e, modalities: [text]}\n"
        "  initial: executor\n  fallback: executor\n")
    diags = V.validate_workspace(str(tmp_path), profile_set="setT2", only="t2")
    bad = [d for d in diags if d.code == "invalid_tier_modalities"]
    assert len(bad) == 1
    assert "asserts nothing" in bad[0].message


def test_explain_tiers_documents_directions():
    data, text = explain.tiers()
    assert set(data["modality_directions"]) == {
        "inbound", "outbound", "bidirectional"}
    assert "outbound_is_inert" in data
    assert "inbound" in text


def test_validate_warns_that_outbound_is_inert(tmp_path):
    # Outbound parses and is stored, so it must NOT be an error — a profile
    # should be writable ahead of the delivery work. But it is inert, so
    # staying silent would be the silent-no-op failure mode.
    pdir = tmp_path / ".jaato" / "profiles" / "setO"
    pdir.mkdir(parents=True)
    (tmp_path / ".jaato" / "profiles" / "_base_o.yaml").write_text(
        "name: _base_o\ndescription: b\nplugins: []\n")
    (pdir / "o.yaml").write_text(
        "name: o\ninherits: [_base_o]\nplugins: []\n"
        "model_tiers:\n"
        "  executor: {model: e}\n"
        "  planner: {model: p, modalities: {audio: outbound}}\n"
        "  initial: executor\n  fallback: executor\n")
    diags = V.validate_workspace(str(tmp_path), profile_set="setO", only="o")
    warns = [d for d in diags if d.code == "outbound_modality_not_deliverable"]
    assert len(warns) == 1
    assert warns[0].severity == "warning"
    assert not [d for d in diags if d.code == "invalid_tier_modalities"]


def test_validate_flags_a_bad_direction_and_suggests_bidirectional(tmp_path):
    pdir = tmp_path / ".jaato" / "profiles" / "setB"
    pdir.mkdir(parents=True)
    (tmp_path / ".jaato" / "profiles" / "_base_b.yaml").write_text(
        "name: _base_b\ndescription: b\nplugins: []\n")
    (pdir / "b.yaml").write_text(
        "name: b\ninherits: [_base_b]\nplugins: []\n"
        "model_tiers:\n"
        "  executor: {model: e, modalities: {image: both}}\n"
        "  initial: executor\n  fallback: executor\n")
    diags = V.validate_workspace(str(tmp_path), profile_set="setB", only="b")
    bad = [d for d in diags if d.code == "invalid_tier_modalities"]
    assert len(bad) == 1
    assert "bidirectional" in bad[0].message


def _write_tier_profile(tmp_path, tag, body):
    """One profile-set with a single non-base profile carrying ``body``."""
    pdir = tmp_path / ".jaato" / "profiles" / f"set{tag}"
    pdir.mkdir(parents=True)
    (tmp_path / ".jaato" / "profiles" / f"_base_{tag}.yaml").write_text(
        f"name: _base_{tag}\ndescription: b\nplugins: []\n")
    (pdir / f"{tag}.yaml").write_text(
        f"name: {tag}\ninherits: [_base_{tag}]\nplugins: []\n" + body)
    return V.validate_workspace(
        str(tmp_path), profile_set=f"set{tag}", only=tag)


def test_validate_warns_that_an_unmarshalled_inbound_role_is_inert(tmp_path):
    """#830's shape, caught before a session exists.

    A profile could declare ``audio: inbound`` against a provider whose
    converter has no audio branch: it parses, it is stored, the session-time
    modality check passes (the CATALOG says the model listens), and every
    clip is then withheld at the last step with nothing upstream having said
    why.  The outbound half has warned about its own inertness since #824;
    the inbound half was silent.
    """
    diags = _write_tier_profile(tmp_path, "IA", (
        "provider: nim\n"
        "model_tiers:\n"
        "  executor: {model: e}\n"
        "  planner: {model: p, modalities: {audio: inbound}}\n"
        "  initial: executor\n  fallback: executor\n"))
    warns = [d for d in diags if d.code == "inbound_modality_not_marshalled"]
    assert len(warns) == 1
    assert warns[0].severity == "warning"
    assert "audio_input" in warns[0].message
    # Inert is not invalid: the profile must stay writable.
    assert not [d for d in diags if d.code == "invalid_tier_modalities"]


def test_validate_is_silent_when_the_wire_really_carries_the_modality(tmp_path):
    diags = _write_tier_profile(tmp_path, "IB", (
        "provider: openrouter\n"
        "model_tiers:\n"
        "  executor: {model: e}\n"
        "  planner: {model: p, modalities: {audio: inbound}}\n"
        "  initial: executor\n  fallback: executor\n"))
    assert not [d for d in diags
                if d.code == "inbound_modality_not_marshalled"]


def test_validate_does_not_invent_a_verdict_for_an_uncheckable_role(tmp_path):
    """``video`` has no capability column, so nothing is known about it.

    A false INERT is the failure mode this whole warning family already
    learned once: the outbound version fired unconditionally and told every
    author of a working speaking tier that their profile did nothing.
    """
    diags = _write_tier_profile(tmp_path, "IC", (
        "provider: nim\n"
        "model_tiers:\n"
        "  executor: {model: e}\n"
        "  planner: {model: p, modalities: {video: inbound}}\n"
        "  initial: executor\n  fallback: executor\n"))
    assert not [d for d in diags
                if d.code == "inbound_modality_not_marshalled"]


def test_bidirectional_audio_warns_on_both_halves_separately(tmp_path):
    """A tier can be inert in one direction and live in the other.

    ``google_genai`` carries audio INPUT (any inline_data rides as a Blob)
    and declares no ``output_media``, so the outbound warning must fire, the
    inbound one must not, and the outbound message must say the inbound half
    is live rather than condemning the whole role.
    """
    diags = _write_tier_profile(tmp_path, "ID", (
        "provider: google_genai\n"
        "model_tiers:\n"
        "  executor: {model: e}\n"
        "  planner: {model: gemini-2.5-pro, modalities: {audio: bidirectional}}\n"
        "  initial: executor\n  fallback: executor\n"))
    assert not [d for d in diags
                if d.code == "inbound_modality_not_marshalled"]
    out = [d for d in diags if d.code == "outbound_modality_not_deliverable"]
    assert len(out) == 1
    assert "inbound half of this role IS live" in out[0].message

