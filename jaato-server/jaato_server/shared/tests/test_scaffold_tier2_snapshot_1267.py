"""SDK-only ``new`` emits what it can and says what it skipped (#1267, tier 2).

Tier 1 moved ``jaato-scaffold new`` into jaato-sdk and REFUSED the four
invocations that still reached into jaato-server.  Tier 2 moves no server
module and replaces three of those refusals, each attached to how it goes
wrong:

* **Profile facts in the snapshot.**  ``authoring_snapshot.json`` gains a
  ``profiles`` section: the file keys, derived and removed keys, field types,
  closed vocabularies and the discovery layout.  It must equal the live
  projection, and the layout must be what the scan actually reads (a
  constant nobody reads is documentation that drifts).
* **``new profile-set``** emits with the SDK alone.  The framework validator
  re-checks the set only when jaato-server's ``validate`` verb is installed;
  otherwise ONE line says it was not validated and how to validate it, and
  the emitted keys are checked against the snapshot's facts.
* **``new client --profile X`` never refuses.**  ``X`` is written as given,
  ``--set Y`` writes ``JAATO_PROFILE_SET=Y`` with no lookup, and with
  jaato-server installed a name the workspace does not resolve gets one note.
* **``new processor`` / gated ``new sweep``** write their files and skip the
  probe with one notice when there is no framework loader to run it with.

``new dossier`` stays refused without jaato-server; that refusal is guarded
in ``test_scaffold_ships_with_the_sdk_1267.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaato_sdk.scaffold import authoring_contracts as contracts
from jaato_sdk.scaffold import build, cli
from jaato_server.shared.plugins.subagent import config as cfg
from jaato_server.shared.tests.reversion import Reversion
from jaato_server.shared.tests.test_scaffold_ships_with_the_sdk_1267 import _sdk_only

_BUILD = "jaato-sdk/jaato_sdk/scaffold/build.py"
_CONTRACTS = "jaato-sdk/jaato_sdk/scaffold/authoring_contracts.py"
_CONFIG = "jaato-server/jaato_server/shared/plugins/subagent/config.py"

REVERSIONS = [
    Reversion(
        target=_BUILD,
        find=("    if not _server_validator_installed():\n"
              "        return _report_unvalidated(ws, set_name, emitted)\n"),
        replace="",
        because="an SDK-only profile-set would import the framework "
                "validator after writing the set, and die half-way",
        test="test_sdk_only_profile_set_emits_and_says_it_was_not_validated",
    ),
    Reversion(
        target=_BUILD,
        find=("    _note_named_profile(args, archetype, profile_name)\n"
              "    return None, None, None\n"),
        replace=("    code = _check_named_profile(args, archetype, profile_name)\n"
                 "    return (code, None, None) if code else (None, None, None)\n"),
        because="a profile that does not exist YET is a normal order of work; "
                "refusing it takes `--profile` away from the author who "
                "writes the client first",
        test="test_with_the_server_a_missing_profile_writes_the_client_and_notes",
    ),
    Reversion(
        target=_BUILD,
        find=("    if not _server_available():\n"
              "        return\n"
              "    ws_arg = getattr(args, \"workspace\", None)\n"),
        replace=("    return\n"
                 "    ws_arg = getattr(args, \"workspace\", None)\n"),
        because="without the note a typo is found only on the client's first "
                "run, although the resolver that would have seen it was here",
        test="test_with_the_server_a_missing_profile_writes_the_client_and_notes",
    ),
    Reversion(
        target=_BUILD,
        find=("    if not _server_available():\n"
              "        print(\"\\n\" + PROBE_SKIPPED.format(what=\"processor\"))\n"),
        replace="    if False:\n        pass\n",
        because="an SDK-only processor would run the probe, which imports the "
                "framework loader that is not installed",
        test="test_sdk_only_processor_writes_its_files_and_skips_the_probe",
    ),
    Reversion(
        target=_BUILD,
        find=("    if not _server_available():\n"
              "        print(\"\\n\" + PROBE_SKIPPED.format(what=\"completion gate\"))\n"
              "        return None\n"),
        replace="",
        because="an SDK-only gated sweep would drive its gate through a "
                "loader that is not installed",
        test="test_sdk_only_gated_sweep_writes_its_gate_and_skips_the_probe",
    ),
    Reversion(
        target=_CONTRACTS,
        find='            "extensions": list(cfg.PROFILE_FILE_EXTENSIONS),\n',
        replace='            "extensions": [".json"],\n',
        because="a projection change not followed by --write leaves the "
                "profile section of the snapshot stale",
        test="test_the_profile_section_is_the_live_projection",
    ),
    Reversion(
        target=_CONFIG,
        find=("        if file_path.suffix not in PROFILE_FILE_EXTENSIONS:\n"
              "            continue\n\n"
              "        name, data, error = _parse_profile_file(file_path)\n"),
        replace=("        if file_path.suffix not in ('.json', '.yaml', '.yml'):\n"
                 "            continue\n\n"
                 "        name, data, error = _parse_profile_file(file_path)\n"),
        because="the snapshot would record extensions the scan no longer "
                "reads, so its layout would be a copy free to drift",
        test="test_the_scan_reads_the_recorded_extensions",
    ),
]


# ------------------------------------------------------------------ helpers


def _profile(ws: Path, name: str, subdir: str = "") -> None:
    d = ws / ".jaato" / "profiles" / subdir
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.yaml").write_text(
        f"name: {name}\ndescription: d\nmodel: m\nprovider: anthropic\n"
        "plugins: []\n", encoding="utf-8")


# ----------------------------------------------------------- the snapshot


def test_the_profile_section_is_the_live_projection():
    snap = json.loads(contracts.SNAPSHOT_FILE.read_text(encoding="utf-8"))
    assert snap["profiles"] == contracts.project_profiles(cfg), (
        "the profile section of authoring_snapshot.json differs from the "
        "tree; regenerate: python -m "
        "jaato_server.shared.scaffold.authoring_contracts --write")


def test_live_and_snapshot_profile_facts_agree(monkeypatch):
    live = contracts.profile_facts()
    monkeypatch.setattr(contracts, "_FORCE_SNAPSHOT", True)
    assert contracts.profile_facts() == live


def test_the_profile_facts_are_the_loaders():
    facts = contracts.profile_facts()
    assert facts.file_keys == cfg.PROFILE_FILE_KEYS
    assert set(facts.derived_keys) == set(cfg.PROFILE_DERIVED_FIELDS)
    assert set(facts.removed_keys) == set(cfg.PROFILE_REMOVED_FIELDS)
    assert not facts.file_keys & set(facts.derived_keys)
    assert facts.enums["regulatory.risk_class"] == tuple(sorted(cfg.RISK_CLASSES))
    layout = facts.layout
    assert tuple(layout["extensions"]) == cfg.PROFILE_FILE_EXTENSIONS
    assert layout["profile_set_env_var"] == "JAATO_PROFILE_SET"
    assert [t["tier"] for t in layout["tiers"]] == [
        "profile_set", "workspace", "user", "premium"]
    assert all(f"/{layout['profiles_subdir']}/" in t["location"]
               for t in layout["tiers"][:3])


def test_the_scan_reads_the_recorded_extensions(tmp_path, monkeypatch):
    """The recorded extensions are the ones the scan reads, not a copy."""
    _profile(tmp_path, "yamlone")
    (tmp_path / ".jaato" / "profiles" / "jsonone.json").write_text(json.dumps(
        {"name": "jsonone", "description": "d", "model": "m",
         "provider": "anthropic", "plugins": []}), encoding="utf-8")
    monkeypatch.setattr(cfg, "PROFILE_FILE_EXTENSIONS", (".yaml",))
    found = cfg.discover_profiles(".jaato/profiles", base_path=str(tmp_path),
                                  config_root=str(tmp_path / ".jaato"))
    assert "yamlone" in found.profiles
    assert "jsonone" not in found.profiles


# --------------------------------------------------------------- SDK-only


def test_sdk_only_profile_set_emits_and_says_it_was_not_validated(tmp_path):
    ws = tmp_path / "ws"
    rc, out, err, leaked = _sdk_only(
        "new", "profile-set", "--workspace", str(ws), "--set", "s",
        "--provider", "anthropic", "--model", "m", "--agents", "a,b",
        cwd=tmp_path)
    assert rc == 0, out + err
    assert (ws / ".jaato" / "profiles" / "s" / "a.yaml").is_file()
    assert (ws / ".jaato" / "profiles" / "_base_b.yaml").is_file()
    lines = [ln for ln in out.splitlines() if "not validated" in ln]
    assert len(lines) == 1, out
    assert "pip install jaato-server" in lines[0]
    assert f"jaato-scaffold validate {ws} --set s" in lines[0]
    assert "re-validating" not in out
    assert leaked == "[]"


def test_sdk_only_client_with_a_profile_nobody_declared_is_written(tmp_path):
    ws = tmp_path / "ws"
    rc, out, err, leaked = _sdk_only(
        "new", "client", "--workspace", str(ws), "--profile", "nonexistent",
        "--set", "later", cwd=tmp_path)
    assert rc == 0, out + err
    assert '"nonexistent"' in (ws / "run_client.py").read_text()
    assert "JAATO_PROFILE_SET=later" in (ws / ".env").read_text()
    assert "note:" not in out, "the SDK has no resolver to note with"
    assert leaked == "[]"


def test_sdk_only_processor_writes_its_files_and_skips_the_probe(tmp_path):
    ws = tmp_path / "ws"
    rc, out, err, leaked = _sdk_only("new", "processor", "--name", "gate",
                                     "--workspace", str(ws), cwd=tmp_path)
    assert rc == 0, out + err
    assert (ws / ".jaato" / "scripts" / "processors" / "gate.py").is_file()
    assert out.count("probe skipped") == 1, out
    assert leaked == "[]"


def test_sdk_only_gated_sweep_writes_its_gate_and_skips_the_probe(tmp_path):
    ws = tmp_path / "ws"
    rc, out, err, leaked = _sdk_only("new", "sweep", "--workspace", str(ws),
                                     "--provider", "anthropic", "--model", "m",
                                     cwd=tmp_path)
    assert rc == 0, out + err
    assert (ws / "run_sweep.py").is_file()
    assert list((ws / ".jaato" / "scripts" / "processors").glob("*.py"))
    assert out.count("probe skipped") == 1, out
    assert leaked == "[]"


def test_the_emitted_keys_are_checked_against_the_facts(monkeypatch):
    monkeypatch.setattr(contracts, "_FORCE_SNAPSHOT", True)
    ok = {Path("a.yaml"): build._set_profile_yaml("a", "anthropic", "m")}
    assert build._unread_emitted_keys(ok) == []
    bad = {Path("b.yaml"): "name: b\nmax_turns: 3\n# nope: x\n  nested: y\n"}
    assert build._unread_emitted_keys(bad) == ["b.yaml: max_turns"]


# ------------------------------------------------------ with jaato-server


def test_with_the_server_a_missing_profile_writes_the_client_and_notes(
        tmp_path, capsys):
    ws = tmp_path / "ws"
    _profile(ws, "other")
    rc = cli.main(["new", "client", "--workspace", str(ws),
                   "--profile", "nonexistent"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert '"nonexistent"' in (ws / "run_client.py").read_text()
    notes = [ln for ln in out.splitlines() if ln.startswith("note:")]
    assert len(notes) == 1, out
    assert "doesn't exist yet" in notes[0] and "create_session" in notes[0]


def test_with_the_server_a_profile_in_another_set_is_named(tmp_path, capsys):
    ws = tmp_path / "ws"
    _profile(ws, "scoped", subdir="alt")
    rc = cli.main(["new", "client", "--workspace", str(ws),
                   "--profile", "scoped"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert (ws / "run_client.py").is_file()
    assert "only in profile-set alt" in out and "--set alt" in out


def test_with_the_server_a_known_profile_gets_no_note(tmp_path, capsys):
    ws = tmp_path / "ws"
    _profile(ws, "worker")
    assert cli.main(["new", "client", "--workspace", str(ws),
                     "--profile", "worker"]) == 0
    assert "note:" not in capsys.readouterr().out


def test_with_the_server_profile_set_is_revalidated(tmp_path, capsys):
    ws = tmp_path / "ws"
    rc = cli.main(["new", "profile-set", "--workspace", str(ws), "--set", "s",
                   "--provider", "anthropic", "--model", "m", "--agents", "a"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "re-validating scaffolded set" in out
    assert "not validated" not in out


@pytest.mark.parametrize("argv", [
    ("--profile", "w", "--provider", "anthropic", "--model", "m"),
    ("--profile", "w", "--transport", "in_process"),
])
def test_the_binding_refusals_stay(tmp_path, capsys, argv):
    rc = cli.main(["new", "client", "--workspace", str(tmp_path / "ws"), *argv])
    assert rc == 2
    assert not (tmp_path / "ws" / "run_client.py").exists()


def test_profile_on_an_archetype_with_no_session_is_refused(tmp_path, capsys):
    rc = cli.main(["new", "cascade", "--workspace", str(tmp_path / "ws"),
                   "--profile", "w"])
    assert rc == 2
    assert "Drop --profile" in capsys.readouterr().out
