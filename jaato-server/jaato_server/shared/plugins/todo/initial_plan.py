"""A predefined plan a profile names (#1195): resolve, read, validate.

``plugin_configs.todo.initial_plan_name: <plan_id>`` names a plan authored
at ``<config_root>/plans/<plan_id>.yaml`` — an asset beside ``profiles/``
and ``agents/``, write-denied to a confined runner and committable by the
``.gitignore`` block ``jaato-scaffold new`` writes.  The session that
declares it gets that plan as its active plan before turn 1.

This module is the ONE answer to "which file, and is it a plan", read by
three callers that must agree:

* ``JaatoSession`` at ``configure()``, which loads the plan for the session
  that declared the knob (and fails the session when it cannot);
* ``TodoPlugin.preload_plan``, which installs the loaded plan;
* ``jaato-scaffold validate``, which reports ``initial_plan_missing`` /
  ``initial_plan_invalid`` before any session exists.

It imports nothing from the framework beyond the plan model, so the
validator can call it without a runtime.  It never writes: the authored
file is read with ``yaml.safe_load`` and left exactly as it was — progress
is kept by the plugin's ordinary storage, on a COPY with its own id.

The file's shape mirrors :meth:`TodoPlan.to_dict` (see the plugin README),
so a saved plan is a valid authored plan and vice versa.  Only ``title``
and ``steps[].description`` are required; everything else defaults.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from jaato_sdk.plugins.todo.models import TodoPlan

#: The profile knob, under ``plugin_configs.todo``.
INITIAL_PLAN_KNOB = "initial_plan_name"

#: The directory under the config root that holds authored plans.
PLANS_DIRNAME = "plans"

#: The authored file's extension.  YAML only — the todo plugin persists
#: nothing as JSON.
PLAN_SUFFIX = ".yaml"

#: The hint a session carrying a loaded plan gets in its system prompt,
#: verbatim.  Contributed only when the plan actually loaded.
PRELOADED_PLAN_HINT = (
    "You were invoked with a predefined plan. Use the available TODO tools "
    "to read it and organize your work to comply with it."
)

#: The one tool a session with a predefined plan does not see.
GATED_TOOL = "createPlan"


class InitialPlanError(ValueError):
    """A declared ``initial_plan_name`` that cannot become a plan.

    Raised by :func:`load_initial_plan` and :func:`check_plan_name`, and
    allowed to propagate out of ``JaatoSession.configure()`` so the session
    is REFUSED rather than started without the plan its profile promised.
    ``code`` is the validator's finding code for the same defect.
    """

    def __init__(self, message: str, code: str = "initial_plan_invalid"):
        super().__init__(message)
        self.code = code


def check_plan_name(name: Any) -> str:
    """Return *name* if it is a plan id, else raise :class:`InitialPlanError`.

    An id, not a path: exactly one path component, so the file it names can
    only be ``<config_root>/plans/<name>.yaml``.  Refused: a non-string, an
    empty or whitespace name, an absolute path, anything containing ``/`` or
    ``\\``, and ``.`` / ``..``.  A trailing ``.yaml`` is refused too — the
    suffix is the framework's to add, and accepting it would make
    ``onboarding`` and ``onboarding.yaml`` two spellings of one plan.
    """
    if not isinstance(name, str) or not name.strip():
        raise InitialPlanError(
            f"{INITIAL_PLAN_KNOB} must be a non-empty plan id, got {name!r}",
            code="initial_plan_name_invalid")
    if (name != name.strip() or "/" in name or "\\" in name
            or name in (".", "..") or Path(name).is_absolute()
            or "\x00" in name):
        raise InitialPlanError(
            f"{INITIAL_PLAN_KNOB} {name!r} is not a plan id — it must be one "
            f"path component naming <config_root>/{PLANS_DIRNAME}/<id>"
            f"{PLAN_SUFFIX}, never a path",
            code="initial_plan_name_invalid")
    if name.endswith(PLAN_SUFFIX) or name.endswith(".yml"):
        raise InitialPlanError(
            f"{INITIAL_PLAN_KNOB} {name!r} carries a suffix — name the plan "
            f"id alone ({name.rsplit('.', 1)[0]!r}); the framework adds "
            f"{PLAN_SUFFIX}",
            code="initial_plan_name_invalid")
    return name


def plans_dir(config_root: Optional[str],
              workspace_path: Optional[str]) -> Optional[Path]:
    """The directory authored plans live in, or ``None`` if unknowable.

    ``<config_root>/plans`` when a config root is set, else
    ``<workspace>/.jaato/plans`` — the same primary tier every other
    authored asset resolves against.  There is no user-tier fallback: a
    predefined plan belongs to the workspace that ships the profile.
    """
    if config_root:
        return Path(config_root).expanduser() / PLANS_DIRNAME
    if workspace_path:
        return Path(workspace_path).expanduser() / ".jaato" / PLANS_DIRNAME
    return None


def resolve_plan_path(name: Any, config_root: Optional[str],
                      workspace_path: Optional[str]) -> Path:
    """The file *name* names.  Raises :class:`InitialPlanError` when the name
    is not an id or there is no config root to resolve it against.  Does
    not check that the file exists."""
    plan_id = check_plan_name(name)
    root = plans_dir(config_root, workspace_path)
    if root is None:
        raise InitialPlanError(
            f"{INITIAL_PLAN_KNOB} {plan_id!r} cannot be resolved: the session "
            f"has neither a config_root nor a workspace",
            code="initial_plan_missing")
    return root / f"{plan_id}{PLAN_SUFFIX}"


#: The authored plan file's fields, for the authoring surface to render
#: beneath the ``initial_plan_name`` knob (#1222).  This DESCRIBES what
#: :func:`parse_plan_document` accepts and is co-located with it so the two
#: cannot drift: ``required`` here is exactly the set the ``_require`` calls
#: below enforce, and ``test_plan_file_schema_matches_the_parser`` drives
#: :func:`parse_plan_document` with each ``required`` field absent to prove
#: it.  ``jaato-scaffold explain plugin todo`` reads this so an author learns
#: the file's layout without opening the README; the full field table
#: (``sequence`` / ``validation_required`` / ``depends_on`` / …) lives in the
#: plugin README's "Plan file schema" section, and this is the minimum an
#: author must know to write a file the loader will accept.
#:
#: Each entry is ``(field, required, meaning)``.
PLAN_FILE_FIELDS = (
    ("title", True,
     "Plan summary — a non-empty string."),
    ("steps", True,
     "A non-empty list; each entry is a mapping."),
    ("steps[].description", True,
     "What the step does — a non-empty string."),
    ("started", False,
     "Defaults to true: the authored file is the approval startPlan "
     "otherwise asks for.  Write started: false to make the agent confirm "
     "the plan first."),
    ("context", False,
     "Free-form mapping; initial_plan_name is recorded in it."),
)


def _require(cond: bool, path: Path, what: str) -> None:
    if not cond:
        raise InitialPlanError(f"{path}: {what}", code="initial_plan_invalid")


def parse_plan_document(data: Any, path: Path) -> TodoPlan:
    """Validate a ``yaml.safe_load`` result and build a fresh :class:`TodoPlan`.

    The result is a COPY for one session, never the file's identity:

    * ``plan_id`` and ``created_at`` are minted here whatever the file says,
      so two sessions preloading one file never share (and overwrite) one
      stored plan;
    * each step keeps an authored ``step_id`` (plan-scoped, so an author may
      name steps) and otherwise gets a fresh one; ``sequence`` defaults to
      the step's position;
    * ``started`` defaults to **true** — the authored file is the approval
      ``startPlan`` otherwise asks for.  Write ``started: false`` to make the
      agent confirm it;
    * ``context.initial_plan_name`` records where the plan came from.

    Raises :class:`InitialPlanError` for a document that is not a mapping, a
    missing/empty ``title``, a missing/empty ``steps`` list, or a step that
    is not a mapping with a non-empty string ``description``.
    """
    _require(isinstance(data, dict), path,
             "a plan must be a YAML mapping with 'title' and 'steps'")
    title = data.get("title")
    _require(isinstance(title, str) and bool(title.strip()), path,
             "'title' must be a non-empty string")
    steps = data.get("steps")
    _require(isinstance(steps, list) and bool(steps), path,
             "'steps' must be a non-empty list")
    for i, step in enumerate(steps, start=1):
        _require(isinstance(step, dict), path,
                 f"steps[{i - 1}] must be a mapping with a 'description'")
        desc = step.get("description")
        _require(isinstance(desc, str) and bool(desc.strip()), path,
                 f"steps[{i - 1}].description must be a non-empty string")

    doc = dict(data)
    doc["plan_id"] = str(uuid.uuid4())
    doc["created_at"] = datetime.now(timezone.utc).isoformat() + "Z"
    doc["steps"] = [{"sequence": i, **s} for i, s in enumerate(steps, start=1)]
    if "started" not in doc:
        doc["started"] = True
        doc.setdefault("started_at", doc["created_at"])
    context = dict(doc.get("context") or {})
    context.setdefault(INITIAL_PLAN_KNOB, path.stem)
    doc["context"] = context
    try:
        return TodoPlan.from_dict(doc)
    except (TypeError, ValueError, AttributeError) as exc:
        raise InitialPlanError(f"{path}: not a plan: {exc}",
                               code="initial_plan_invalid") from exc


def load_initial_plan(name: Any, config_root: Optional[str],
                      workspace_path: Optional[str]) -> TodoPlan:
    """Resolve, read and build the plan *name* names.  Read-only.

    Raises :class:`InitialPlanError` with ``code`` ``initial_plan_name_invalid``
    (not an id), ``initial_plan_missing`` (no such file) or
    ``initial_plan_invalid`` (unreadable or not a plan).
    """
    path = resolve_plan_path(name, config_root, workspace_path)
    if not path.is_file():
        raise InitialPlanError(
            f"{INITIAL_PLAN_KNOB} {name!r} names {path}, which does not exist",
            code="initial_plan_missing")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise InitialPlanError(f"{path}: not readable YAML: {exc}",
                               code="initial_plan_invalid") from exc
    return parse_plan_document(data, path)
