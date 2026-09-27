"""``jaato-doctor`` survives a source it may not read (#1360).

In a confined session ``jaato-doctor`` died with ``PermissionError`` on an
editable install's ``pyproject.toml``: the profile let the runner stat the
file and refused the read, and ``dist_state`` read it with no ``try``.  The
first diagnostic an agent is told to run failed on the environment it was
asked to diagnose.  Now the source version is "unknown", the dependency
check says the skew was not checked, and any check that still raises is
reported under its own name while the others run.
"""

from __future__ import annotations

import json
from pathlib import Path

from jaato_sdk import doctor
from jaato_server.shared.scaffold import dependencies
from jaato_server.shared.tests.reversion import Reversion

_DEPS = "jaato-server/jaato_server/shared/scaffold/dependencies.py"
_DOCTOR = "jaato-sdk/jaato_sdk/doctor.py"

REVERSIONS = [
    Reversion(
        target=_DEPS,
        find="    except OSError as exc:\n        return None, exc.strerror or type(exc).__name__\n",
        replace="    except ValueError as exc:\n        return None, str(exc)\n",
        test="test_an_unreadable_source_is_unknown_not_a_traceback",
        because="a refused read of the source pyproject.toml propagates",
    ),
    Reversion(
        target=_DOCTOR,
        find="        elif st.get(\"source_unreadable\"):\n",
        replace="        elif False:\n",
        test="test_doctor_does_not_pass_a_comparison_it_did_not_make",
        because="an unread source is reported as agreeing with its metadata",
    ),
    Reversion(
        target=_DOCTOR,
        find="    checks += _guarded(lambda: check_dependency_coherence())\n",
        replace="    checks += check_dependency_coherence()\n",
        test="test_a_check_that_raises_is_reported_and_the_rest_still_run",
        because="one check raising takes the whole doctor run down",
    ),
]


def _refuse_reads(monkeypatch, target: Path) -> None:
    real = Path.read_text

    def read_text(self, *args, **kwargs):
        if self == target:
            raise PermissionError(13, "Permission denied", str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)


class _Dist:
    def __init__(self, src: Path) -> None:
        self._url = json.dumps({"url": f"file://{src}",
                                "dir_info": {"editable": True}})

    def read_text(self, name):
        return self._url if name == "direct_url.json" else None


def test_an_unreadable_source_is_unknown_not_a_traceback(monkeypatch, tmp_path):
    pp = tmp_path / "pyproject.toml"
    pp.write_text('version = "9.9.9"\n')
    _refuse_reads(monkeypatch, pp)
    monkeypatch.setattr(dependencies, "version", lambda name: "1.0.0")
    monkeypatch.setattr(dependencies, "distribution", lambda name: _Dist(tmp_path))

    st = dependencies.dist_state("jaato-sdk")

    assert st["editable"] is True
    assert st["source_version"] is None
    assert st["source_unreadable"] == "Permission denied"
    assert st["skew"] is False


def test_a_readable_source_still_reports_its_version(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text('name = "x"\nversion = "9.9.9"\n')
    monkeypatch.setattr(dependencies, "version", lambda name: "1.0.0")
    monkeypatch.setattr(dependencies, "distribution", lambda name: _Dist(tmp_path))

    st = dependencies.dist_state("jaato-sdk")

    assert st["source_version"] == "9.9.9"
    assert st["source_unreadable"] is None
    assert st["skew"] is True


def test_doctor_does_not_pass_a_comparison_it_did_not_make(monkeypatch):
    def dist_state(name):
        return {"name": name, "installed": "1.0.0", "editable": True,
                "source": "/src", "source_version": None,
                "source_unreadable": "Permission denied", "skew": False}

    monkeypatch.setattr(dependencies, "dist_state", dist_state)
    monkeypatch.setattr(dependencies, "framework_dists", lambda: ["jaato-sdk"])

    [check] = doctor.check_dependency_coherence()

    assert check.status == doctor.WARN
    assert "skew was not checked" in check.detail


def test_a_check_that_raises_is_reported_and_the_rest_still_run(monkeypatch):
    ran = []
    for name in dir(doctor):
        if name.startswith("check_") and callable(getattr(doctor, name)):
            def ok(*args, _name=name, **kwargs):
                ran.append(_name)
                return [doctor.Check(_name, doctor.PASS, "")]
            monkeypatch.setattr(doctor, name, ok)

    def check_dependency_coherence():
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(doctor, "check_dependency_coherence",
                        check_dependency_coherence)
    monkeypatch.setattr(doctor, "probe_daemon", lambda *a, **k: None)
    monkeypatch.setattr(doctor, "load_known_env_vars", lambda: {})

    checks = doctor.run_checks(
        socket_path="/nonexistent.sock", pidfile="/nonexistent.pid",
        workspace="/nonexistent", config_root=None, env_file=None,
        secret=None, auto_start=False, release_check=False,
    )

    failed = [c for c in checks if c.name == "dependency coherence"]
    assert len(failed) == 1
    assert failed[0].status == doctor.WARN
    assert "PermissionError" in failed[0].detail
    assert "check_driver" in ran, "the checks after the failing one did not run"
