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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from .params import param_env

#: Grader kinds this engine knows how to run.  See ``graders/``.
GRADER_KINDS = ("script", "processor", "judge")

#: Harness kinds — what one arm IS (jaato #1110).
#:
#: ``session``: one jaato session, opened by the engine from
#: ``harness.profile`` and sent ``input.prompt``.  The default when
#: ``harness.kind`` is absent, and the only shape that existed before.
#:
#: ``driver``: a PROCESS the engine starts from ``harness.run``, which
#: opens as many sessions as it likes — a backtest cell of ten stages in a
#: fixed order, with market data as host tools in the driver's own process.
#: The engine hands it a contract as environment (:mod:`jaato_eval.driver`)
#: and reads its exit code; ``input.params`` is its input.
#:
#: One discriminator rather than mutually exclusive keys, the shape
#: graders already have (``kind: script | processor | judge``), so the
#: parser can name the variant a key does not belong to.
HARNESS_KINDS = ("session", "driver")


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
    """What the agent — or the driver — is asked to do.

    Which half is populated follows ``harness.kind``, and the parser
    refuses a key that belongs to the other variant rather than ignoring
    it: a ``prompt`` on a driver arm reaches nothing, and a task author
    who wrote one believes it was sent.

    Attributes:
        prompt: The instruction text.  Required and non-empty under
            ``kind: session`` — a task with no input is not a task.
            ``None`` under ``kind: driver``, whose input is ``params``.
        agent: Persona name (``.jaato/agents/<name>.md``), optional when
            the profile carries its own.  Session arms only.
        agent_params: ``{{param}}`` substitutions for the persona.  This is
            the parameterisation axis: one persona, many task instances.
            Session arms only.
        params: The driver's input, under ``kind: driver`` — exported to
            the driver process as environment, one ``JAATO_EVAL_PARAM_<KEY>``
            variable each, exactly as a ``script`` grader receives
            ``agent_params`` (:mod:`jaato_eval.params`).  A driver arm's
            graders receive this mapping AS their ``agent_params``, so the
            scorer of a ``(ticker, date)`` cell reads the same two
            variables the driver was given.
    """

    prompt: Optional[str] = None
    agent: Optional[str] = None
    agent_params: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def grader_params(self) -> Dict[str, Any]:
        """The inputs a grader is handed as ``agent_params``.

        ``agent_params`` for a session arm and ``params`` for a driver
        arm — whichever the variant populated, and never both, so a
        grader sees the mapping the arm actually ran with.
        """
        return dict(self.params) if self.params else dict(self.agent_params)


@dataclass(frozen=True)
class HarnessSpec:
    """The configuration under test.

    Attributes:
        kind: One of :data:`HARNESS_KINDS`.  ``session`` (the default when
            absent) opens one session; ``driver`` runs a process.
        profile: Profile name resolved within the active profile set.
            Required under ``kind: session``; refused under ``kind:
            driver``, where the driver names its own profiles per stage
            and a value here would bind nothing.
        profile_set: Default set (``<config_root>/profiles/<set>/``).  A
            sweep overrides this per arm — it is the model/provider axis,
            and swapping it is the whole "can I use a cheaper model?"
            experiment.  Both kinds: it reaches a session through the
            engine's ``create_session``, and a driver through the
            ``JAATO_PROFILE_SET`` the engine writes into the workspace
            ``.env`` (:mod:`jaato_eval.fixture`).
        run: The driver command line, run through the shell with the
            workspace as its working directory.  Required under ``kind:
            driver``; refused under ``kind: session``.
    """

    kind: str = "session"
    profile: Optional[str] = None
    profile_set: Optional[str] = None
    run: Optional[str] = None

    @property
    def is_driver(self) -> bool:
        return self.kind == "driver"


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
    gates and this block drives only the second.

    THREE WALL-CLOCK GATES, and only two of them are budget gates (#724).
    An author who sets ``seconds`` here and watches an arm cut at 900s is
    meeting the third, which lives in the harness rather than in any
    budget and was previously named in no manifest documentation at all:

    * ``budget.seconds`` — declared in THIS block.  Bounds the POOL:
      every arm of the task together.  Reconciled when a session ENDS.
    * ``budget_control.limits.seconds`` — declared in the arm's PROFILE.
      Bounds ONE session, enforced daemon-side.
    * the per-arm ceiling, default 900s, set with ``--arm-timeout``
      (``0`` disables) — declared on the HARNESS command line, and
      nowhere in any manifest.  Bounds ONE arm's wall clock.  This is
      the gate that produces ``arm exceeded the per-arm ceiling``.

    The third exists because neither budget gate can do its job: a pool's
    ``seconds`` is reconciled when a session ends, so a session that never
    ends never consumes it and the pool cannot abort it.  See
    :data:`jaato_eval.runner.DEFAULT_ARM_TIMEOUT_SECONDS`.

    Feeding ``budget.seconds`` into the arm ceiling would be wrong, not
    merely unimplemented: an arm whose ceiling moved with what earlier
    arms spent is not a reproducible measurement.  They stay separate, and
    ``jaato-eval run`` warns before spending anything when a ``seconds``
    here is larger than the arm ceiling, because no single arm can then
    reach the allowance.

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


def _parse_harness_and_input(raw: Dict[str, Any], path: Path
                             ) -> Tuple[HarnessSpec, InputSpec]:
    """Parse ``harness`` and ``input`` together, per ``harness.kind``.

    Together because the two blocks share one discriminator: which
    ``input`` keys are meaningful is decided by ``harness.kind``, and a
    parser that validated them apart could only refuse the wrong keys by
    guessing at the variant.
    """
    h_raw = _mapping(_require(raw, "harness", path, "top level"), path, "harness")
    kind = str(_require(h_raw, "kind", path, "harness")) if "kind" in h_raw else "session"
    if kind not in HARNESS_KINDS:
        raise ManifestError(
            path, f"harness.kind: unknown kind {kind!r}; expected one of {HARNESS_KINDS}")
    if kind == "driver":
        return _driver_variant(h_raw, raw, path)
    return _session_variant(h_raw, raw, path)


def _refuse_keys(block: Dict[str, Any], keys: Sequence[str], path: Path,
                 where: str, kind: str) -> None:
    """Refuse a key the other variant reads.

    A key nothing reads is the silent-ignore shape the manifest parser
    exists to prevent: ``input.prompt`` on a driver arm reaches no
    session, ``harness.run`` on a session arm starts no process, and in
    both cases the author believes otherwise.  The error names the
    variant so the fix is to change ``kind`` or drop the key, not to
    guess.
    """
    for key in keys:
        if key in block:
            raise ManifestError(
                path, f"{where}.{key} is not read under harness.kind: {kind}; "
                      f"drop it, or change the kind")


def _session_variant(h_raw: Dict[str, Any], raw: Dict[str, Any],
                     path: Path) -> Tuple[HarnessSpec, InputSpec]:
    _refuse_keys(h_raw, ("run",), path, "harness", "session")
    in_raw = _mapping(_require(raw, "input", path, "top level"), path, "input")
    _refuse_keys(in_raw, ("params",), path, "input", "session")
    prompt = str(_require(in_raw, "prompt", path, "input")).strip()
    if not prompt:
        raise ManifestError(path, "input.prompt is empty")
    harness = HarnessSpec(
        kind="session",
        profile=str(_require(h_raw, "profile", path, "harness")),
        profile_set=h_raw.get("profile_set"),
    )
    task_input = InputSpec(
        prompt=prompt,
        agent=in_raw.get("agent"),
        agent_params=_mapping(in_raw.get("agent_params"), path, "input.agent_params"),
    )
    return harness, task_input


def _driver_variant(h_raw: Dict[str, Any], raw: Dict[str, Any],
                    path: Path) -> Tuple[HarnessSpec, InputSpec]:
    _refuse_keys(h_raw, ("profile",), path, "harness", "driver")
    run = str(_require(h_raw, "run", path, "harness")).strip()
    if not run:
        raise ManifestError(path, "harness.run is empty")
    # ``input`` is optional here: a driver whose whole input is its command
    # line has nothing to put in it, and forcing an empty block would be a
    # default that hides nothing and helps nobody.
    in_raw = _mapping(raw.get("input"), path, "input")
    _refuse_keys(in_raw, ("prompt", "agent", "agent_params"), path, "input", "driver")
    params = _mapping(in_raw.get("params"), path, "input.params")
    # Refused HERE, before any arm is materialised: the export is what the
    # driver receives and what its graders read, so a collision is an
    # authoring error in the task, not a runtime condition of one arm.
    _, collision = param_env(params)
    if collision:
        raise ManifestError(path, f"input.params: {collision}")
    harness = HarnessSpec(kind="driver", run=run, profile_set=h_raw.get("profile_set"))
    return harness, InputSpec(params=params)


def _parse_graders(raw: Dict[str, Any], path: Path) -> List[GraderSpec]:
    """The ``graders`` list: non-empty, each of a known kind."""
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
    return graders


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

    harness, task_input = _parse_harness_and_input(raw, path)

    graders = _parse_graders(raw, path)

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
