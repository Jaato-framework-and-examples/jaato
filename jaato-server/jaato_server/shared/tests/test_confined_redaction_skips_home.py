"""A confined runner does not scan ~/.jaato for stored credentials.

The phase 2b kernel run logged an enforced ``read`` denial on ``~/.jaato``
from the runner at bootstrap.  Neither LSM lets a confined runner read
``~/.jaato/*_auth.json``, so a stored credential cannot reach its output;
the scan (#1215) only produced the denial.  An unconfined runner still
scans its home, as before.
"""

import json

from jaato_server.shared import secret_redaction as sr
from jaato_server.shared.tests.reversion import Reversion

REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/shared/secret_redaction.py",
        find="        include_home=not src.get(\"confined\", False),\n",
        replace="        include_home=True,\n",
        test="test_a_confined_runner_skips_the_home_scan",
        because="a confined runner lists ~/.jaato at every bootstrap and the "
                "kernel refuses it (an enforced AVC on every session)",
    ),
]


def _home_with_a_credential(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".jaato").mkdir(parents=True)
    (home / ".jaato" / "nim_auth.json").write_text(json.dumps({"api_key": "nvapi-SECRET-123456"}))
    monkeypatch.setenv("HOME", str(home))
    return home


def test_a_confined_runner_skips_the_home_scan(tmp_path, monkeypatch):
    _home_with_a_credential(tmp_path, monkeypatch)
    try:
        redactor = sr.configure_redaction_sources({}, confined=True)
        # Not scanned, so not known: the value is left as it is.
        assert redactor.redact_text("x nvapi-SECRET-123456") == "x nvapi-SECRET-123456"
        assert not any("nim" in n for n in redactor.names)
    finally:
        sr.reset_redaction_sources()


def test_an_unconfined_runner_still_scans_its_home(tmp_path, monkeypatch):
    _home_with_a_credential(tmp_path, monkeypatch)
    try:
        redactor = sr.configure_redaction_sources({}, confined=False)
        assert "nvapi-SECRET-123456" not in redactor.redact_text("x nvapi-SECRET-123456")
    finally:
        sr.reset_redaction_sources()


def test_a_reload_keeps_the_confined_answer(tmp_path, monkeypatch):
    _home_with_a_credential(tmp_path, monkeypatch)
    try:
        sr.configure_redaction_sources({}, confined=True)
        redactor = sr.configure_redaction_sources({})
        assert not any("nim" in n for n in redactor.names)
    finally:
        sr.reset_redaction_sources()


def test_the_home_directory_is_left_out_only_when_asked(tmp_path, monkeypatch):
    home = _home_with_a_credential(tmp_path, monkeypatch)
    assert str(home / ".jaato") in sr.stored_auth_directories(None, None)
    assert str(home / ".jaato") not in sr.stored_auth_directories(None, None, include_home=False)
