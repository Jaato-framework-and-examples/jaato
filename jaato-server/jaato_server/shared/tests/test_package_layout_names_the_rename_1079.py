"""``jaato-doctor`` names the 1.0 namespace rename (#1079).

Before the rename the server shipped as top-level ``shared`` / ``server``
packages and the daemon ran as ``python -m server``.  Both now live under
``jaato_server``; the old names no longer resolve.  ``check_package_layout``
is the migration hint: it WARNs when a top-level ``shared`` or ``server``
STILL resolves (a leftover pre-1.0 install, or a third-party collision) and
is silent otherwise.

WHY THIS ASSERTS VERDICTS AND NOT OUTPUT.  "the doctor prints a line" is
satisfied by a check that always prints; the contract is that it WARNs on
positive evidence and stays PASS without it — the #1014 / #1023 posture
(absence of the old names is not a finding, or the check warns about a
correct install).  Both directions are asserted, and a control the
reversion cannot flip keeps the "silent" half honest.

This guard lives under ``jaato-server/jaato_server/shared/tests/`` (not
beside its subject in ``jaato-sdk``) because the reversion meta-suite walks
only ``shared/tests`` and ``server/tests``; ``test_doctor_detects_checkout_skew_823.py``
is the precedent for a guard here that targets ``jaato-sdk/jaato_sdk/doctor.py``.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

from jaato_sdk import doctor as D
from jaato_server.shared.tests.reversion import Reversion


#: Neutralise the positive-evidence branch so a resolving top-level ``shared``
#: is never recorded — the check then always PASSes, which is exactly the
#: "a gate that reads nothing passes everything" failure this guard exists to
#: catch.  ``test_warns_when_a_top_level_shared_resolves`` must go red.
REVERSIONS = [
    Reversion(
        target="jaato-sdk/jaato_sdk/doctor.py",
        find=(
            "        if spec is not None:\n"
            "            stale.append((name, getattr(spec, \"origin\", None) or \"?\"))"
        ),
        replace=(
            "        if False:\n"
            "            stale.append((name, getattr(spec, \"origin\", None) or \"?\"))"
        ),
        because=(
            "check_package_layout stops noticing a stale top-level "
            "shared/server package — the migration hint then never fires"
        ),
        test="test_warns_when_a_top_level_shared_resolves",
    ),
]


@pytest.fixture
def _top_level_shared_on_path():
    """Fabricate a genuine top-level ``shared`` package for ``find_spec``.

    ``find_spec`` searches ``sys.path`` finders; planting a ``shared.py`` on a
    fresh path entry makes ``find_spec("shared")`` resolve exactly as a
    leftover pre-1.0 install would, without importing anything.  Removed
    afterwards so the fabrication cannot leak into another test.
    """
    import importlib
    d = tempfile.mkdtemp(prefix="jaato-stale-layout-")
    open(os.path.join(d, "shared.py"), "w").close()
    sys.path.insert(0, d)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        try:
            sys.path.remove(d)
        except ValueError:
            pass
        sys.modules.pop("shared", None)
        importlib.invalidate_caches()


def test_warns_when_a_top_level_shared_resolves(_top_level_shared_on_path):
    """A resolving top-level ``shared`` is a WARN naming the new command."""
    checks = D.check_package_layout()
    assert len(checks) == 1
    c = checks[0]
    assert c.status == D.WARN, (
        f"expected WARN when a top-level shared resolves, got {c.status}"
    )
    assert "python -m jaato_server" in c.detail
    assert "shared" in c.detail


def test_silent_when_only_jaato_server(monkeypatch):
    """No top-level ``shared`` / ``server`` -> PASS, never a warning.

    The control the reversion cannot flip: with the positive-evidence branch
    neutralised the check is PASS here anyway, so this half stays green in
    both worlds and only ``test_warns_...`` distinguishes them.  It exists to
    pin the #1014 posture — the absence of the old names is not a finding.
    """
    import importlib.util as _u
    orig = _u.find_spec

    def fake(name, *a, **k):
        if name in ("shared", "server"):
            return None
        return orig(name, *a, **k)

    monkeypatch.setattr(_u, "find_spec", fake)
    checks = D.check_package_layout()
    assert len(checks) == 1
    assert checks[0].status == D.PASS


def test_the_preflight_run_actually_includes_the_layout_check(monkeypatch):
    """``run_checks`` wires the check in — a check nothing calls is inert."""
    import inspect
    src = inspect.getsource(D.run_checks)
    assert "check_package_layout()" in src, (
        "run_checks does not call check_package_layout(); the migration hint "
        "would never run in a real doctor invocation"
    )
