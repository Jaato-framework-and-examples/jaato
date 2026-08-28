"""Task manifest — ``task.yaml`` loading and validation.

A task is *an input, an environment, and graders*.  This module is the
parser for that triple.  It deliberately has no defaults that could hide
an authoring mistake: an absent required key is an error, not an
inferred value.  Per project policy, no fallback heuristics.

The manifest names existing artefacts (a fixture tree, a profile, a
processor script, a judge profile) rather than inventing a grading
language.  Everything it points at is something the framework already
executes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

#: Grader kinds this engine knows how to run.  See ``graders/``.
GRADER_KINDS = ("script", "processor", "judge")


class ManifestError(ValueError):
    """Raised when a ``task.yaml`` is missing, malformed, or inconsistent.

    Carries the offending path so a sweep over many tasks can report
    *which* manifest is wrong without the caller re-deriving it.
    """

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class EnvironmentSpec:
    """Where the agent runs.

    Attributes:
        fixture: Directory copied fresh into a scratch workspace for every
            arm.  Relative to the task directory.  The agent mutates the
            copy; graders inspect it; the original is never touched.
        config_root: The read-only ``.jaato/`` tree (profiles, agents,
            completion schemas, permissions).  Kept *separate* from the
            workspace so the task definition cannot be edited by the agent
            under test.  Relative to the task directory.

    This block holds only what a PROFILE CANNOT EXPRESS.  It used to carry
    ``apparmor`` and ``runtime_limits`` as well; both are ``SubagentProfile``
    fields, so a task declares them in its own profile like every other
    session property.

    ``runtime_limits`` was worse than duplication — it was structurally
    dead.  ``runner_spawn.py`` reads ``getattr(profile, "runtime_limits")``
    and there is no session-kwarg vehicle, so a value here could not have
    reached the runner however it was plumbed, while this docstring claimed
    it was "forwarded to the profile layer" and the shipped example
    declared a ``tool_timeout_seconds`` that did nothing.

    ``apparmor`` did work, via ``ClientConfigRequest.apparmor``.  It went
    anyway: two writers for one setting means a precedence rule, and the
    framework defines none — so the rule would have been this engine's
    invention, applied to confinement.
    """

    fixture: Path
    config_root: Path


@dataclass(frozen=True)
class InputSpec:
    """What the agent is asked to do.

    Attributes:
        prompt: The instruction text.  Required — a task with no input is
            not a task.
        agent: Persona name (``.jaato/agents/<name>.md``), optional when
            the profile carries its own.
        agent_params: ``{{param}}`` substitutions for the persona.  This is
            the parameterisation axis: one persona, many task instances.
    """

    prompt: str
    agent: Optional[str] = None
    agent_params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HarnessSpec:
    """The configuration under test.

    Attributes:
        profile: Profile name resolved within the active profile set.
        profile_set: Default set (``<config_root>/profiles/<set>/``).  A
            sweep overrides this per arm — it is the model/provider axis,
            and swapping it is the whole "can I use a cheaper model?"
            experiment.
    """

    profile: str
    profile_set: Optional[str] = None


@dataclass(frozen=True)
class GraderSpec:
    """One grader declaration.

    ``kind`` selects the adapter; the remaining keys are that adapter's
    own configuration and are validated by the adapter, not here — this
    keeps the manifest parser from having to know every grader's schema.

    Attributes:
        kind: One of :data:`GRADER_KINDS`.
        config: Adapter-specific keys, verbatim from the manifest.
        weight: Relative contribution to a weighted score.  Reporting uses
            it only for the weighted column; pass-rate ignores it.
    """

    kind: str
    config: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    @property
    def identifier(self) -> str:
        """Short label distinguishing this grader in a report."""
        for key in ("script", "run", "profile"):
            if key in self.config:
                return str(self.config[key])
        return self.kind


@dataclass(frozen=True)
class BudgetSpec:
    """The task's CASCADE POOL — an aggregate over all of its arms.

    This is **not** the per-arm ceiling.  jaato has two independent budget
    gates and this block drives only the second:

    - **per-arm ceiling** — ``budget_control:`` in the arm's own profile,
      under the task's ``config_root``.  A session carrying one is on its
      own books: never clamped to a pool's remainder, never depleting it.
      That is what an arm needs, since an arm whose ceiling moved with
      what earlier arms spent would not be a reproducible measurement.
      The engine does nothing for this gate — the daemon enforces what the
      profile declares.
    - **this pool** — shared by the task's arms (repeats × profile sets),
      so ``repeats: 20`` cannot run away and no task can starve another.
      Arms drawing on it are clamped at spawn, degraded mid-flight at a
      rung, and refused once it is empty.

    A session declaring its own ``budget_control`` does not draw here.  So
    a task whose profiles all carry ceilings will leave its pool untouched
    — correct, and worth knowing before reading an untouched pool as
    evidence that nothing ran.

    ``limits`` dimensions mirror ``budget_control.limits``: usd, tokens,
    seconds, tool_calls, turns.  ``degrade`` is the optional rung ladder,
    same grammar as a profile's — each entry an ``at:`` percentage plus
    either ``model_tiers:`` (brownout) or ``action: abort``.

    An arm stopped by either gate is BLOCKED, not FAIL: it produced no
    signal about the thing under test.
    """

    limits: Dict[str, float] = field(default_factory=dict)
    degrade: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class TaskManifest:
    """A parsed ``task.yaml``.

    Attributes:
        task_id: Stable identifier, used as the results key.  Must be
            unique across the dataset.
        path: The manifest file this was parsed from.
        root: Directory containing the manifest; all relative paths in the
            manifest resolve against it.
    """

    task_id: str
    path: Path
    root: Path
    description: str
    environment: EnvironmentSpec
    input: InputSpec
    harness: HarnessSpec
    graders: List[GraderSpec]
    budget: BudgetSpec
    repeats: int = 1

    def resolved_fixture(self) -> Path:
        return (self.root / self.environment.fixture).resolve()

    def resolved_config_root(self) -> Path:
        return (self.root / self.environment.config_root).resolve()


def _require(data: Dict[str, Any], key: str, path: Path, where: str) -> Any:
    """Fetch a required key, rejecting both absence and an explicit null.

    ``prompt:`` with nothing after it parses to ``None``, and a bare
    ``str(None)`` turns that into the literal string ``"None"`` — a task
    that runs and asks the agent to do "None".  Absent and empty must not
    share a representation here either.
    """
    if key not in data:
        raise ManifestError(path, f"{where}: missing required key {key!r}")
    if data[key] is None:
        raise ManifestError(path, f"{where}: key {key!r} is present but null")
    return data[key]


def _mapping(value: Any, path: Path, where: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ManifestError(path, f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def load_manifest(path: Path) -> TaskManifest:
    """Parse and validate one ``task.yaml``.

    Raises:
        ManifestError: on a missing file, non-mapping document, missing
            required key, unknown grader kind, or a fixture/config_root
            that does not exist on disk.  Existence is checked here rather
            than at run time so a malformed dataset fails before any
            provider tokens are spent.
    """
    if not path.is_file():
        raise ManifestError(path, "no such manifest")

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ManifestError(path, f"not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ManifestError(path, f"expected a mapping at the top level, got {type(raw).__name__}")

    root = path.parent
    task_id = str(_require(raw, "id", path, "top level"))

    env_raw = _mapping(_require(raw, "environment", path, "top level"), path, "environment")
    environment = EnvironmentSpec(
        fixture=Path(str(_require(env_raw, "fixture", path, "environment"))),
        config_root=Path(str(_require(env_raw, "config_root", path, "environment"))),
    )

    for label, resolved in (("fixture", root / environment.fixture),
                            ("config_root", root / environment.config_root)):
        if not resolved.is_dir():
            raise ManifestError(path, f"environment.{label} does not exist: {resolved}")

    in_raw = _mapping(_require(raw, "input", path, "top level"), path, "input")
    prompt = str(_require(in_raw, "prompt", path, "input")).strip()
    if not prompt:
        raise ManifestError(path, "input.prompt is empty")
    task_input = InputSpec(
        prompt=prompt,
        agent=in_raw.get("agent"),
        agent_params=_mapping(in_raw.get("agent_params"), path, "input.agent_params"),
    )

    h_raw = _mapping(_require(raw, "harness", path, "top level"), path, "harness")
    harness = HarnessSpec(
        profile=str(_require(h_raw, "profile", path, "harness")),
        profile_set=h_raw.get("profile_set"),
    )

    graders_raw = _require(raw, "graders", path, "top level")
    if not isinstance(graders_raw, list) or not graders_raw:
        raise ManifestError(path, "graders must be a non-empty list")
    graders: List[GraderSpec] = []
    for i, g in enumerate(graders_raw):
        g = _mapping(g, path, f"graders[{i}]")
        kind = str(_require(g, "kind", path, f"graders[{i}]"))
        if kind not in GRADER_KINDS:
            raise ManifestError(
                path, f"graders[{i}]: unknown kind {kind!r}; expected one of {GRADER_KINDS}")
        config = {k: v for k, v in g.items() if k not in ("kind", "weight")}
        graders.append(GraderSpec(kind=kind, config=config,
                                  weight=float(g.get("weight", 1.0))))

    raw_budget = dict(_mapping(raw.get("budget"), path, "budget"))
    raw_degrade = raw_budget.pop("degrade", None) or []
    if not isinstance(raw_degrade, list):
        raise ManifestError(
            path, f"budget.degrade must be a list of rungs, got "
                  f"{type(raw_degrade).__name__}")
    try:
        budget_limits = {k: float(v) for k, v in raw_budget.items()}
    except (TypeError, ValueError) as exc:
        raise ManifestError(
            path, f"budget limits must be numbers: {exc}") from exc
    budget = BudgetSpec(limits=budget_limits, degrade=list(raw_degrade))

    repeats = int(raw.get("repeats", 1))
    if repeats < 1:
        raise ManifestError(path, f"repeats must be >= 1, got {repeats}")

    return TaskManifest(
        task_id=task_id, path=path, root=root,
        description=str(raw.get("description", "")).strip(),
        environment=environment, input=task_input, harness=harness,
        graders=graders, budget=budget, repeats=repeats,
    )


def discover_tasks(root: Path) -> List[TaskManifest]:
    """Load every ``task.yaml`` under ``root``, sorted by task id.

    Raises:
        ManifestError: on the first malformed manifest, or on a duplicate
            task id — two tasks sharing an id would silently overwrite each
            other in the results pivot.
    """
    manifests = [load_manifest(p) for p in sorted(root.rglob("task.yaml"))]
    seen: Dict[str, Path] = {}
    for m in manifests:
        if m.task_id in seen:
            raise ManifestError(m.path, f"duplicate task id {m.task_id!r} (also in {seen[m.task_id]})")
        seen[m.task_id] = m.path
    return sorted(manifests, key=lambda m: m.task_id)
