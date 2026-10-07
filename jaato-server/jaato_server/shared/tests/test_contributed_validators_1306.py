"""A package can contribute ``validate`` findings (#1306).

``jaato-scaffold`` had two extension seams, verbs and ``explain`` topics, and
none for ``validate``.  So a package that taught an author how to write an
asset (premium's ``explain reactors`` for ``reactors.json``) had no way to
check the file the author then wrote: a malformed rule file passed
``validate`` silently.  ``jaato.scaffold_validators`` is the third seam.

What these tests hold it to:

* a contributed finding reaches the run, stamped with its contributor, and
  the contributor is handed the RESOLVED profiles the framework checked;
* with nothing installed the output is exactly what it was before the seam:
  ``Diagnostic.as_dict()`` keeps its six keys, and the text line is unchanged;
* a contributor that fails to load, raises, or returns something malformed is
  REPORTED, never dropped: a check that did not run must not read as a pass;
* the attribution is the framework's stamp, not the contributor's claim.

Entry points are faked at ``_validator_entry_points`` (the one place the group
is read), except for one test that fakes ``importlib.metadata.entry_points``
to show which group is scanned.
"""

from __future__ import annotations

import importlib.metadata
import types
from pathlib import Path

import pytest

from jaato_server.shared.scaffold import api
from jaato_server.shared.scaffold import introspection_verbs
from jaato_server.shared.scaffold import validate as V
from jaato_server.shared.tests.reversion import Reversion

_VALIDATE = "jaato-server/jaato_server/shared/scaffold/validate.py"

REVERSIONS = [
    Reversion(
        target=_VALIDATE,
        find=("    out.extend(contributed_findings(\n"
              "        workspace=str(ws), config_root=config_root, profile_set=profile_set,\n"
              "        only=only, profiles=result.profiles, providers=providers,\n"
              "        plugins=plugins, gc_names=gc_names))\n"),
        replace="",
        because="a validator that is discovered and never called contributes "
                "nothing, and the asset it checks passes validate silently -- "
                "the gap #1306 reports",
        test="test_a_contributed_finding_reaches_the_run_attributed",
    ),
    Reversion(
        target=_VALIDATE,
        find=("    try:\n"
              "        result = validator.validate(request)\n"
              "    except Exception as exc:\n"
              "        return [_malformed(source, f\"raised {type(exc).__name__}: {exc}\")]\n"),
        replace="    result = validator.validate(request)\n",
        because="one broken contributor would take the framework's own checks "
                "down with it",
        test="test_a_validator_that_raises_is_reported_and_the_rest_still_run",
    ),
    Reversion(
        target=_VALIDATE,
        find=("        if self.source is not None:\n"
              "            out[\"source\"] = self.source\n"),
        replace="        out[\"source\"] = self.source\n",
        because="every --json consumer would see a new key on every built-in "
                "finding: a run with no contributor would no longer be the run "
                "it was before the seam",
        test="test_with_nothing_installed_the_output_shape_is_unchanged",
    ),
    Reversion(
        target=_VALIDATE,
        find=("        except Exception as exc:\n"
              "            failures.append(Diagnostic(\n"
              "                \"warn\", \"validator_unavailable\",\n"),
        replace=("        except Exception as exc:\n"
                 "            continue\n"
                 "            failures.append(Diagnostic(\n"
                 "                \"warn\", \"validator_unavailable\",\n"),
        because="a validator that failed to load is checks that did not run; "
                "saying nothing reads as a pass over files nobody looked at",
        test="test_a_validator_that_cannot_load_is_reported",
    ),
    Reversion(
        target=_VALIDATE,
        find=("        tier=getattr(item, \"tier\", None) or \"workspace\",\n"
              "        source=source)\n"),
        replace=("        tier=getattr(item, \"tier\", None) or \"workspace\",\n"
                 "        source=getattr(item, \"source\", None) or source)\n"),
        because="a contributor could pass its finding off as the framework's, "
                "or as another package's",
        test="test_the_attribution_is_the_frameworks_stamp",
    ),
]


# ---------------------------------------------------------------- fixtures

class _EP:
    """A hand-built entry point: what ``importlib.metadata`` would hand back."""

    def __init__(self, name, obj, dist="jaato-premium"):
        self.name = name
        self._obj = obj
        self.dist = types.SimpleNamespace(name=dist) if dist else None

    def load(self):
        if isinstance(self._obj, BaseException):
            raise self._obj
        return self._obj


class _Validator:
    def __init__(self, name, findings=None, raises=None):
        self.name = name
        self._findings = findings or []
        self._raises = raises
        self.requests = []

    def validate(self, request):
        self.requests.append(request)
        if self._raises is not None:
            raise self._raises
        return self._findings


@pytest.fixture
def install(monkeypatch):
    """Install fake contributed validators for one test."""
    def _install(*eps):
        V.reset_external_validators()
        monkeypatch.setattr(V, "_validator_entry_points", lambda: list(eps))
    yield _install
    V.reset_external_validators()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace with a base profile and a child that inherits its plugins."""
    prof = tmp_path / ".jaato" / "profiles"
    prof.mkdir(parents=True)
    (prof / "_base.yaml").write_text(
        "name: _base\ndescription: shared base\nplugins: [todo]\n",
        encoding="utf-8")
    (prof / "worker.yaml").write_text(
        "name: worker\ndescription: does the work\ninherits: [_base]\n"
        "plugins: [cli]\n", encoding="utf-8")
    return tmp_path


def _contributed(diags):
    return [d for d in diags if d.source is not None]


# ------------------------------------------------------------------- tests

def test_a_contributed_finding_reaches_the_run_attributed(install, workspace):
    finding = V.Diagnostic("error", "reactor_rule_invalid",
                           "unknown event_type 'tool.end'",
                           where="reactors/rules.json")
    validator = _Validator("reactors", [finding])
    install(_EP("reactors", validator))

    diags = V.validate_workspace(str(workspace))

    [d] = _contributed(diags)
    assert (d.severity, d.code, d.source, d.tier) == (
        "error", "reactor_rule_invalid", "jaato-premium:reactors", "workspace")
    # The framework's own findings are still there, and come first.
    assert diags.index(d) == len(diags) - 1
    assert any(x.source is None for x in diags)


def test_the_request_carries_the_resolved_profiles(install, workspace):
    """A contributor judges what the daemon would load, inherits applied."""
    validator = _Validator("reactors")
    install(_EP("reactors", validator))

    V.validate_workspace(str(workspace), only="worker")

    [request] = validator.requests
    assert isinstance(request, api.ValidationRequest)
    assert request.workspace == str(workspace.resolve())
    assert request.config_root == str(workspace.resolve() / ".jaato")
    assert request.only == "worker"
    assert set(request.profiles["worker"].plugins) >= {"cli", "todo"}
    assert "cli" in request.plugins


def test_with_nothing_installed_the_output_shape_is_unchanged(install, workspace):
    install()
    diags = V.validate_workspace(str(workspace))
    assert diags, "the fixture should draw at least one built-in finding"
    assert all(d.source is None for d in diags)
    for d in diags:
        assert list(d.as_dict()) == [
            "severity", "code", "message", "profile", "where", "tier"]
    assert V.contributed_findings(
        workspace=str(workspace), config_root="", profile_set=None, only=None,
        profiles={}, providers={}, plugins={}, gc_names=[]) == []


def test_a_validator_that_raises_is_reported_and_the_rest_still_run(
        install, workspace):
    broken = _Validator("broken", raises=RuntimeError("boom"))
    fine = _Validator("fine", [V.Diagnostic("warn", "fine_says", "hello")])
    install(_EP("broken", broken), _EP("fine", fine))

    diags = _contributed(V.validate_workspace(str(workspace)))

    by_code = {d.code: d for d in diags}
    failed = by_code["validator_failed"]
    assert failed.severity == "warn"
    assert failed.source == "jaato-premium:broken"
    assert "RuntimeError: boom" in failed.message
    assert "not a pass" in failed.message
    assert by_code["fine_says"].source == "jaato-premium:fine"


def test_a_validator_that_cannot_load_is_reported(install, workspace):
    install(_EP("reactors", ImportError("no module named jaato_premium.x")))

    [d] = _contributed(V.validate_workspace(str(workspace)))

    assert (d.severity, d.code) == ("warn", "validator_unavailable")
    assert "did NOT run" in d.message
    assert "ImportError" in d.message


def test_an_object_that_is_not_a_validator_is_reported(install, workspace):
    install(_EP("odd", types.SimpleNamespace(name="odd")))
    [d] = _contributed(V.validate_workspace(str(workspace)))
    assert d.code == "validator_unavailable"
    assert "ScaffoldValidator" in d.message


def test_a_duplicate_name_runs_once_and_says_so(install, workspace):
    first = _Validator("reactors", [V.Diagnostic("info", "first", "one")])
    second = _Validator("reactors", [V.Diagnostic("info", "second", "two")])
    install(_EP("reactors", first, dist="a-pkg"),
            _EP("reactors", second, dist="b-pkg"))

    codes = [d.code for d in _contributed(V.validate_workspace(str(workspace)))]

    assert "first" in codes and "second" not in codes
    assert "validator_unavailable" in codes


def test_malformed_findings_are_reported_not_passed_through(install, workspace):
    install(
        _EP("notalist", _Validator("notalist", findings="oops")),
        _EP("badsev", _Validator("badsev", [
            V.Diagnostic("warning", "x", "y"),
            V.Diagnostic("error", "", "no code")])),
    )
    diags = _contributed(V.validate_workspace(str(workspace)))
    assert [d.code for d in diags] == ["validator_failed"] * 3
    assert all(d.severity == "warn" for d in diags)


def test_the_attribution_is_the_frameworks_stamp(install, workspace):
    liar = V.Diagnostic("error", "unknown_plugin", "pretend",
                        source="framework")
    install(_EP("reactors", _Validator("reactors", [liar])))

    [d] = _contributed(V.validate_workspace(str(workspace)))

    assert d.source == "jaato-premium:reactors"
    assert d is not liar


def test_a_contributed_error_fails_the_cli_and_is_named(
        install, workspace, capsys):
    install(_EP("reactors", _Validator("reactors", [
        V.Diagnostic("error", "reactor_rule_invalid", "bad rule")])))
    args = types.SimpleNamespace(target=str(workspace), set=None,
                                 profile=None, json=False)

    code = introspection_verbs._cmd_validate(args)

    line = next(l for l in capsys.readouterr().out.splitlines()
                if "reactor_rule_invalid" in l)
    assert code == 1
    assert line.endswith("(from jaato-premium:reactors)")


def test_a_built_in_line_prints_as_it_always_did():
    d = V.Diagnostic("warn", "unknown_knob", "msg", profile="p", where="w",
                     tier="workspace")
    assert introspection_verbs._format_diagnostic(d) == (
        "[warn] [workspace] p: unknown_knob: msg @ w")


def test_the_group_scanned_is_the_documented_one(monkeypatch):
    asked = []

    def fake_entry_points(*, group):
        asked.append(group)
        return []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    assert V._validator_entry_points() == []
    assert asked == ["jaato.scaffold_validators"]
    assert api.VALIDATOR_ENTRY_POINT_GROUP == "jaato.scaffold_validators"
    assert api.SCAFFOLD_EXTENSION_API == "1.2"
