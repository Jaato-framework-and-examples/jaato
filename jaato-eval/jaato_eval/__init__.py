"""jaato-eval — a benchmark spine over jaato's session and profile primitives.

The framework already carries most of what an agent benchmark needs: the
environment (workspace + read-only ``config_root`` + AppArmor +
``config_root``), the harness under test (profiles, and profile *sets*
selected by ``JAATO_PROFILE_SET``), the input axis (``agent_params``), a
typed output boundary (``completion_payload_schema``), graders (completion
processors, judge sessions), and the metrics (``UsageBreakdown.cost_usd``,
OTel spans).

This package is the missing spine: a task manifest, hermetic fixture
materialisation, three grader adapters over one verdict type, a sweep
driver, and a result store.

It depends on ``jaato-sdk`` only — never ``shared.*``.  If something here
cannot be built on the SDK, that is an SDK gap to be fixed in the SDK.

See ``docs/design/eval-environments-layer.md`` in the jaato repo.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .arm import ArmResult, ArmSpec
from .manifest import ManifestError, TaskManifest, discover_tasks, load_manifest
from .results import ResultStore
from .verdict import BLOCKED, FAIL, PASS, Report, Verdict

__all__ = [
    "__version__",
    "ArmResult", "ArmSpec",
    "TaskManifest", "ManifestError", "load_manifest", "discover_tasks",
    "ResultStore",
    "Verdict", "Report", "PASS", "FAIL", "BLOCKED",
]
