"""scaffold validate checks prefetch directives — the validate counterpart to
`explain prefetch`.

A {{!py:scripts/<f>.py}} in an agent persona / base instruction pointing at a
missing script — or a script without `def render(context, args)` — raises
PrefetchError at session-prep (mandatory) or silently degrades (optional).
validate surfaces it before runtime.
"""
import tempfile
from pathlib import Path

from shared.scaffold.validate import validate_workspace


def _ws(tmp):
    ws = Path(tmp)
    (ws / ".jaato" / "agents").mkdir(parents=True)
    (ws / ".jaato" / "scripts").mkdir(parents=True)
    (ws / ".jaato" / "profiles").mkdir(parents=True)
    return ws


def _codes(diags):
    return {d.code for d in diags}


def test_valid_prefetch_directive_no_finding():
    with tempfile.TemporaryDirectory() as tmp:
        ws = _ws(tmp)
        (ws / ".jaato" / "agents" / "a.md").write_text(
            "Persona.\n\n{{!py:scripts/ok.py arg1}}\n")
        (ws / ".jaato" / "scripts" / "ok.py").write_text(
            "def render(context, args):\n    return 'hi'\n")
        codes = _codes(validate_workspace(str(ws)))
        assert "prefetch_script_missing" not in codes
        assert "prefetch_render_missing" not in codes
        assert "prefetch_render_signature" not in codes


def test_mandatory_missing_script_is_error():
    with tempfile.TemporaryDirectory() as tmp:
        ws = _ws(tmp)
        (ws / ".jaato" / "agents" / "a.md").write_text("{{!py:scripts/__vt_nope.py}}\n")
        miss = [d for d in validate_workspace(str(ws))
                if d.code == "prefetch_script_missing"]
        assert miss and miss[0].severity == "error"


def test_optional_missing_script_is_warn():
    with tempfile.TemporaryDirectory() as tmp:
        ws = _ws(tmp)
        (ws / ".jaato" / "agents" / "a.md").write_text("{{!py?:scripts/__vt_nope.py}}\n")
        miss = [d for d in validate_workspace(str(ws))
                if d.code == "prefetch_script_missing"]
        assert miss and miss[0].severity == "warn"


def test_script_without_render_is_error():
    with tempfile.TemporaryDirectory() as tmp:
        ws = _ws(tmp)
        (ws / ".jaato" / "agents" / "a.md").write_text("{{!py:scripts/norender.py}}\n")
        (ws / ".jaato" / "scripts" / "norender.py").write_text(
            "def not_render(c, a):\n    return ''\n")
        assert "prefetch_render_missing" in _codes(validate_workspace(str(ws)))


def test_wrong_render_signature_is_warn():
    with tempfile.TemporaryDirectory() as tmp:
        ws = _ws(tmp)
        (ws / ".jaato" / "agents" / "a.md").write_text("{{!py:scripts/onearg.py}}\n")
        (ws / ".jaato" / "scripts" / "onearg.py").write_text(
            "def render(context):\n    return ''\n")
        sig = [d for d in validate_workspace(str(ws))
               if d.code == "prefetch_render_signature"]
        assert sig and sig[0].severity == "warn"
