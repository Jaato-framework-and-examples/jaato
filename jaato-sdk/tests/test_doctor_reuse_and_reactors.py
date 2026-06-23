"""jaato-doctor: stale premium-reactor entry-point detection + the reuse-vs-fresh
advisory — so an authoring agent makes the reuse decision from the actual
environment (via the tool), not from reading framework source."""
from types import SimpleNamespace
from jaato_sdk import doctor


def test_premium_pyproject_reactors_reads_entry_points(tmp_path):
    (tmp_path / "jaato_premium").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[project.entry-points."jaato.premium_reactors"]\n'
        'drift_monitor = "x:a"\n'
        'reliability = "y:b"\n')
    spec = SimpleNamespace(submodule_search_locations=[str(tmp_path / "jaato_premium")])
    assert doctor._premium_pyproject_reactors(spec) == ["drift_monitor", "reliability"]


def test_premium_pyproject_reactors_none_when_not_editable():
    assert doctor._premium_pyproject_reactors(
        SimpleNamespace(submodule_search_locations=None)) is None


def test_check_premium_reactors_warns_on_stale(monkeypatch):
    import importlib.util, importlib.metadata
    monkeypatch.setattr(importlib.util, "find_spec", lambda n: SimpleNamespace())
    monkeypatch.setattr(importlib.metadata, "entry_points",
                        lambda group=None: [SimpleNamespace(name="drift_monitor")])
    monkeypatch.setattr(doctor, "_premium_pyproject_reactors",
                        lambda spec: ["drift_monitor", "reliability"])
    c = doctor.check_premium_reactors()[0]
    assert c.status == doctor.WARN
    assert "reliability" in c.detail and "--no-deps" in c.detail


def test_check_premium_reactors_pass_when_in_sync(monkeypatch):
    import importlib.util, importlib.metadata
    monkeypatch.setattr(importlib.util, "find_spec", lambda n: SimpleNamespace())
    monkeypatch.setattr(importlib.metadata, "entry_points",
                        lambda group=None: [SimpleNamespace(name="drift_monitor"),
                                            SimpleNamespace(name="reliability")])
    monkeypatch.setattr(doctor, "_premium_pyproject_reactors",
                        lambda spec: ["drift_monitor", "reliability"])
    assert doctor.check_premium_reactors()[0].status == doctor.PASS


def test_check_premium_reactors_na_when_not_installed(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda n: None)
    c = doctor.check_premium_reactors()[0]
    assert c.status == doctor.PASS and "N/A" in c.detail


def test_reuse_advice_printed_for_daemon_checks(capsys):
    doctor._print([doctor.Check("premium reactors", doctor.PASS, "x"),
                   doctor.Check("socket", doctor.PASS, "y")])
    out = capsys.readouterr().out
    assert "reuse vs fresh" in out and "explain paths" in out


def test_reuse_advice_skipped_without_daemon_checks(capsys):
    doctor._print([doctor.Check("session", doctor.PASS, "x")])
    assert "reuse vs fresh" not in capsys.readouterr().out
