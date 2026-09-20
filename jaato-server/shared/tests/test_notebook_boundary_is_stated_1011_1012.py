"""The notebook's execution boundary is STATED, not discovered by refusal.

Issues #1011 and #1012.  The behavioural suite lives beside the plugin
(``shared/plugins/notebook/tests/test_boundary_visibility_1012.py``); this
module holds the subset that carries ``REVERSIONS``, because
``test_every_guard_detects_its_own_reversion`` discovers guard modules under
``shared/tests`` and ``server/tests`` only — a reversion declared in the
plugin package would never be applied, and an unrun guard must not read as a
working one.

WHAT WENT WRONG.  ``establish_containment`` picks one of four tiers and the
choice materially changes what a cell can do.  On the audit tier ``import
ctypes`` is refused — CPython runs ``pythonapi = PyDLL(None)`` at module
import — and that takes numpy, pandas, scipy, matplotlib, osmnx and torch
with it.  The model was told none of this.  A live session hit the refusal,
read its wording ("loading native code would bypass the notebook's filesystem
boundary") as *the sandbox forbids native code*, repeated that to the user
three times with increasing confidence, and steered them to an external API.
It had ``cli`` throughout, where one ``python -c "import numpy"`` — a
subprocess, unaudited — would have falsified the theory immediately.

So the claims below are about TEXT, and the thing that makes a text guard
decorative is matching a word that survives the reversion.  Each reversion
here therefore puts back the exact prior behaviour, and each names the one
test that must notice it.
"""

import pytest

from shared.plugins.notebook.backends.subprocess_kernel import (
    SubprocessKernelBackend,
)
from shared.plugins.notebook.kernel_sandbox import (
    BOUNDARY_APPARMOR,
    BOUNDARY_AUDIT,
    ContainmentPolicy,
    NotebookContainmentError,
    _check_dlopen,
    boundary_notice,
)
from shared.plugins.notebook.plugin import NotebookPlugin
from shared.plugins.notebook.types import NotebookInfo

_APPARMOR = (
    "shared.plugins.notebook.backends.subprocess_kernel."
    "apparmor_enforced_profile")


@pytest.fixture
def policy(tmp_path):
    return ContainmentPolicy(str(tmp_path), allow_tmp=False)


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    monkeypatch.setattr(_APPARMOR, lambda: None)
    monkeypatch.delenv("JAATO_NOTEBOOK_ALLOW_UNCONTAINED_EXEC", raising=False)
    p = NotebookPlugin()
    p.initialize({"workspace_root": str(tmp_path), "backend": "subprocess"})
    yield p
    p.shutdown()


def test_an_unnamed_tier_produces_no_claim():
    """A backend naming no tier must not be described as some other one.

    #1012's constraint: render what exists, never add a second mechanism
    that can disagree with it.  Falling back to a plausible default here
    would tell a Kaggle session — whose cells run on Kaggle's machines —
    that ``import ctypes`` is refused.
    """
    assert boundary_notice(None) == ()
    assert boundary_notice("some-future-tier") == ()


def test_the_audit_notice_separates_containment_from_availability():
    """The distinction the failing session could not make.

    "Refused by the boundary" and "not installed" call for opposite next
    moves, and the agent took the second for the first.
    """
    text = " ".join(boundary_notice(BOUNDARY_AUDIT))
    assert "NOT a missing package" in text
    assert "import ctypes" in text
    assert "numpy" in text


def test_the_audit_notice_says_a_subprocess_is_unaudited():
    """The single fact that would have ended the incident.

    A spawned child is bounded by the OS and nothing else — the module's own
    docstring has always said ``!pip install X`` works — so the same import
    the cell cannot do runs through ``cli`` or ``!python``.
    """
    text = " ".join(boundary_notice(BOUNDARY_AUDIT))
    assert "subprocess is NOT audited" in text
    assert "`cli`" in text


def test_the_apparmor_notice_does_not_carry_the_audit_tier_s_cost():
    """No hook is installed on the AppArmor tier, so nothing is refused.

    One notice for all tiers would cost a correctly-confined session the
    whole scientific stack for no reason.
    """
    text = " ".join(boundary_notice(BOUNDARY_APPARMOR))
    assert "import normally" in text
    assert "REFUSED" not in text


def test_the_dlopen_refusal_denies_the_over_generalisation(policy):
    """#1011 ask 3: the message the agent actually read.

    It must name ``ctypes`` as the culprit, deny the general claim about
    native code, and point at a surface where the work does run.  The tier
    is named too — this hook is installed on the audit tier and no other,
    so its appearance is evidence of which tier a session really got.
    """
    with pytest.raises(NotebookContainmentError) as exc:
        _check_dlopen(policy, (None,))
    message = str(exc.value)
    assert "audit tier" in message
    assert "import ctypes" in message
    assert "numpy" in message
    assert "import normally" in message
    assert "`cli`" in message


def test_the_active_tier_reaches_the_system_prompt(plugin):
    """Shape (1): the standing fact, before the first cell."""
    instructions = plugin.get_system_instructions()
    assert "Notebook execution boundary: audit" in instructions
    assert "subprocess is NOT audited" in instructions


def test_the_first_result_from_a_real_kernel_reports_its_own_tier(plugin):
    """Shape (2), end to end.

    The tier comes off the kernel's READY frame — the process that actually
    established it — rather than being re-derived daemon-side, which is what
    makes a kernel that came up under a different posture visible.
    """
    result = plugin._execute_code({"code": "print(6 * 7)"})
    assert result["execution_boundary"]["boundary"] == BOUNDARY_AUDIT
    assert "42" in result["output"]
    info = plugin._backends["subprocess"].list_notebooks()[0]
    assert info.boundary_kind == BOUNDARY_AUDIT


def test_the_announcement_is_spent_once_per_kernel():
    """Told once, not on every cell — and told again after a respawn.

    ``_spawn`` re-arms the latch, so re-announcing is exactly the respawn
    case; repeating it per cell would be per-turn context for a fact that
    does not change.
    """
    class _FakeKernel:
        announce_boundary = True
        info = NotebookInfo(notebook_id="n", name="n", backend="subprocess",
                            boundary_kind=BOUNDARY_AUDIT)

    kernel = _FakeKernel()
    take = SubprocessKernelBackend._consume_boundary_announcement
    assert take(kernel) == BOUNDARY_AUDIT
    assert take(kernel) is None


# ---------------------------------------------------------------------------
# Reversions: put each mechanism back the way it was, and name the one test
# that must notice.  See test_every_guard_detects_its_own_reversion.
# ---------------------------------------------------------------------------

from shared.tests.reversion import (  # noqa: E402
    Reversion,
)

_NB = "jaato-server/shared/plugins/notebook"

REVERSIONS = [
    Reversion(
        target=f"{_NB}/kernel_sandbox.py",
        find='    return _BOUNDARY_NOTICES.get(kind or "", ())',
        replace='    return _BOUNDARY_NOTICES.get(\n'
                '        kind or "", _BOUNDARY_NOTICES[BOUNDARY_AUDIT])',
        because="an unknown tier must produce no claim, not a plausible one",
        test="test_an_unnamed_tier_produces_no_claim",
    ),
    Reversion(
        target=f"{_NB}/kernel_sandbox.py",
        find='"IN A CELL. That is the boundary refusing, NOT a missing package: "',
        replace='"in a cell. "',
        because="the notice must separate containment from availability",
        test="test_the_audit_notice_separates_containment_from_availability",
    ),
    Reversion(
        target=f"{_NB}/kernel_sandbox.py",
        find='"A subprocess is NOT audited. The same import succeeds through the "',
        replace='"Packages are resolved from the interpreter installation. "',
        because="the unaudited subprocess is the fact that ends the incident",
        test="test_the_audit_notice_says_a_subprocess_is_unaudited",
    ),
    Reversion(
        target=f"{_NB}/kernel_sandbox.py",
        find='        "`import ctypes`, numpy, pandas and the rest of the scientific stack "\n'
             '        "import normally.",',
        replace='        "`import ctypes` is REFUSED here too.",',
        because="the AppArmor notice must not inherit the audit tier's cost",
        test="test_the_apparmor_notice_does_not_carry_the_audit_tier_s_cost",
    ),
    Reversion(
        target=f"{_NB}/kernel_sandbox.py",
        find='            f"{REFUSAL_PREFIX}: ctypes.dlopen(None) is refused. That handle "',
        replace='            "notebook containment: ctypes.dlopen(None) is refused \\u2014 loading "',
        because="the old wording licensed 'the sandbox forbids native code'",
        test="test_the_dlopen_refusal_denies_the_over_generalisation",
    ),
    Reversion(
        target=f"{_NB}/plugin.py",
        find="        boundary_info = self._boundary_instruction_block()",
        replace='        boundary_info = ""',
        because="the standing statement must reach the system prompt",
        test="test_the_active_tier_reaches_the_system_prompt",
    ),
    Reversion(
        target=f"{_NB}/backends/subprocess_kernel.py",
        find='                info.boundary_kind = ready.get("boundary") or None',
        replace="                pass  # the READY frame's tier, discarded",
        because="the kernel's own tier must survive the handshake",
        test="test_the_first_result_from_a_real_kernel_reports_its_own_tier",
    ),
    Reversion(
        target=f"{_NB}/backends/subprocess_kernel.py",
        find="        kernel.announce_boundary = False\n"
             "        return kernel.info.boundary_kind",
        replace="        return kernel.info.boundary_kind",
        because="an announcement that repeats is per-cell noise",
        test="test_the_announcement_is_spent_once_per_kernel",
    ),
]
