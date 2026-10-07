"""What the pre-warm template imports beyond plugin discovery.

The pool template (``python -m jaato_server.server.runner --template-mode``)
exists so a pool slot forked from it inherits a warm interpreter instead of
importing the runner stack per session.  It warmed only half of that:
``registry.discover(tier_filter="runner")`` walks the plugin packages'
``__init__.py`` files, and the modules ``session.bootstrap`` actually spends
its time importing are reached from elsewhere — ``JaatoRuntime`` /
``JaatoSession`` / ``retry_utils`` (and through them ``anthropic`` and its
pydantic models), the MCP plugin's thread (``mcp``), and a handful of
helpers the bootstrap pulls in lazily.  Every one of those was imported
AFTER the fork, so every slot paid for it twice: once in time, and once in
memory, because a module imported after ``fork()`` lives in that slot's
private pages instead of in pages shared copy-on-write with the template.

Measured on one daemon (echo provider, unconfined, 4 CPUs), before and
after this list:

==============================================  ==========  ===========
                                                before      after
==============================================  ==========  ===========
private memory of a slot serving a session      114 MB      25.5 MB
``session.new``, one session, warm slot         ~2.0 s      ~0.45 s
``session.new``, three at once                  ~3.9 s      ~1.4 s
in-process ``bootstrap_session``                1.96 s      0.07 s
==============================================  ==========  ===========

**What may be on this list.** A module imported here runs in the TEMPLATE,
which carries the daemon's environment and no session's.  So an entry must
not read anything session-scoped at import time — ``os.environ``, the home
directory, the temp directory, the cwd — because the value it captures would
be the daemon's and every slot would inherit it (#1171 is the shape: a
module-scope ``tempfile.gettempdir()`` resolved in the template).  Every
jaato module below was audited by recording those reads during the import;
none makes one.  The third-party packages they pull in read only their own
diagnostic knobs at import (``ANTHROPIC_LOG``, ``OPENAI_LOG``, ``OTEL_*``,
``WEBSOCKETS_*``, locale variables), and ``openai`` reads the three Azure
variables into module defaults for its global client, which no jaato
provider uses — every provider builds an explicit client.  The stated cost
is that a session's ``.env`` can no longer change those import-time knobs;
the daemon's environment decides them.

**What else it must not do.** Start a thread.  The template ``fork()``s
slots, and a thread alive at fork time is not copied while any lock it held
is — :func:`preload` reports the thread count and the guard
(``test_template_preload_covers_bootstrap.py``) asserts it stays one.

**Best effort, never fatal.** An entry that is not installed (a provider
SDK behind an extra) is skipped and logged at DEBUG; one that raises
anything else is skipped and logged at WARNING.  A preload failure costs
the per-session import it would have saved and nothing more — the session
imports the module itself, exactly as before this list existed — so it
must never take the template, and with it the pool, down.

**Who declares what.** Two sources, merged by :func:`plan`:

* :data:`CORE_MODULES`, here: what the FRAMEWORK imports for every session
  (the session stack, the runner's own bootstrap helpers).  No plugin owns
  these, so no plugin could declare them.
* ``PLUGIN_PRELOAD``, a literal tuple of module names in a plugin's or a
  model provider's package ``__init__.py``: what THAT package imports
  lazily.  It sits beside the import it describes, so its owner keeps it
  current.  It is read without importing the package (an AST read of the
  literal when the package is not yet imported), which is what lets a
  provider declare without every provider being imported to find out.

A declaration from outside the built-in package is honoured only when its
distribution is listed in ``JAATO_PLUGIN_ALLOW_PRELOAD``: preloading runs
that distribution's import-time code in the template, and its author may
not know the rule above.  An ignored or malformed declaration is logged,
never fatal.

**Keeping it complete.** The guard runs a real ``bootstrap_session`` after
:func:`preload` and fails when the bootstrap imports any module the
template did not, naming it, so a module that becomes a bootstrap import
later is declared (here, or in its owning package) rather than silently
moving back into every slot.
"""

from __future__ import annotations

import ast
import gc
import importlib
import importlib.metadata
import importlib.util
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("jaato_server.server.runner.template")

#: The package-level name a plugin or provider declares its preloads under.
PLUGIN_PRELOAD_ATTR = "PLUGIN_PRELOAD"

#: The package holding the built-in model providers.
MODEL_PROVIDER_PACKAGE = "jaato_server.shared.plugins.model_provider"

#: Modules the framework imports for every ``session.bootstrap``.  Ordered
#: roughly by cost so the log's timing reads top-down; order does not
#: affect correctness.  A module a plugin or provider imports belongs in
#: that package's ``PLUGIN_PRELOAD`` instead.
CORE_MODULES: Tuple[str, ...] = (
    # The session stack: JaatoRuntime -> JaatoSession -> retry_utils ->
    # anthropic (+ pydantic model construction) and the model_provider
    # packages it imports for error classification.
    "jaato_server.server.runner.session",
    "jaato_server.shared.jaato_client",
    "jaato_server.shared.jaato_runtime",
    "jaato_server.shared.jaato_session",
    "anthropic",
    # Imported lazily from bootstrap steps and session construction.
    "jaato_server.server.confinement.apparmor",
    "jaato_server.server.runner.lsm_confine",
    "jaato_server.server.runner.slot_plugins",
    "jaato_server.shared.ai_disclosure",
    # Imported by runner/session.py's plugin step.  It used to arrive
    # through server/core.py, until #1549 made jaato_server.server lazy.
    "jaato_server.shared.bootstrap_timing",
    "jaato_server.shared.capability_drop",
    "jaato_server.shared.completion_schema_loader",
    "jaato_server.shared.event_bus_tools",
    "jaato_server.shared.jaato_self_shadowing",
    "jaato_server.shared.lifecycle_tools",
    "jaato_server.shared.plugins.session.serializer",
    "jaato_server.shared.repo_guidance",
    "jaato_server.shared.seccomp_filter",
)


@dataclass
class PreloadPlan:
    """What the template will import, and the declarations it did not take.

    ``modules`` is :data:`CORE_MODULES` followed by every honoured
    declaration, de-duplicated in first-seen order.  ``declared`` maps an
    owning package to the modules it declared and was honoured for;
    ``ignored`` maps an owning package to why its declaration was not.
    """

    modules: Tuple[str, ...] = ()
    declared: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    ignored: Dict[str, str] = field(default_factory=dict)


def _literal_declaration(package: str) -> Any:
    """``PLUGIN_PRELOAD`` from *package*'s source, without importing it.

    Returns ``None`` when the package or the assignment cannot be found.
    Raises ``ValueError`` when the assignment is not a literal, which is
    the contract: a declaration must be readable without running code.
    """
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    try:
        tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == PLUGIN_PRELOAD_ATTR):
            return ast.literal_eval(node.value)
    return None


def read_declaration(package: str) -> Optional[Tuple[str, ...]]:
    """The validated ``PLUGIN_PRELOAD`` of *package*, or ``None``.

    An already-imported package is asked directly; any other is read from
    source.  Raises ``ValueError`` for a declaration that is not a tuple
    (or list) of non-empty strings.
    """
    module = sys.modules.get(package)
    if module is not None:
        raw = getattr(module, PLUGIN_PRELOAD_ATTR, None)
    else:
        raw = _literal_declaration(package)
    if raw is None:
        return None
    if not isinstance(raw, (tuple, list)) or not all(
            isinstance(m, str) and m for m in raw):
        raise ValueError(
            f"{PLUGIN_PRELOAD_ATTR} must be a tuple of module names, "
            f"got {raw!r}")
    return tuple(raw)


def _declaring_package(module: str) -> Optional[str]:
    """The package that holds a plugin module's declaration, if any.

    A plugin's factory usually lives in ``<package>.plugin``; the
    declaration lives on ``<package>`` (where ``PLUGIN_TIER`` lives too).
    """
    candidates = [module]
    if "." in module:
        candidates.append(module.rsplit(".", 1)[0])
    for candidate in candidates:
        mod = sys.modules.get(candidate)
        if mod is not None and hasattr(mod, PLUGIN_PRELOAD_ATTR):
            return candidate
    return None


def _owners_from_registry(registry: Any) -> List[Tuple[str, Optional[str], bool]]:
    """``(package, distribution, builtin)`` for each discovered plugin."""
    from jaato_server.shared.plugins.entry_point_trust import is_builtin_module

    owners = []
    for origin in registry.get_plugin_sources().values():
        package = _declaring_package(origin.module or "")
        if package is not None:
            owners.append(
                (package, origin.distribution, is_builtin_module(package)))
    return owners


def _owners_from_providers() -> List[Tuple[str, Optional[str], bool]]:
    """``(package, distribution, builtin)`` for each model provider.

    In-tree providers are found by directory and out-of-tree ones by their
    ``jaato.model_providers`` entry point.  Nothing here imports a
    provider package: :func:`read_declaration` reads the literal.
    """
    from jaato_server.shared.plugins.entry_point_trust import (
        entry_point_distribution, entry_point_module, is_builtin_module,
    )
    from jaato_server.shared.plugins.model_provider import (
        MODEL_PROVIDER_ENTRY_POINT,
    )

    owners = []
    spec = importlib.util.find_spec(MODEL_PROVIDER_PACKAGE)
    for location in (spec.submodule_search_locations or []) if spec else []:
        for item in sorted(Path(location).iterdir()):
            if (item.is_dir() and not item.name.startswith(("_", "."))
                    and item.name != "tests"
                    and (item / "__init__.py").is_file()):
                owners.append(
                    (f"{MODEL_PROVIDER_PACKAGE}.{item.name}", None, True))
    try:
        eps = importlib.metadata.entry_points(group=MODEL_PROVIDER_ENTRY_POINT)
    except Exception:  # noqa: BLE001 — metadata is best effort here
        eps = []
    for ep in eps:
        module = entry_point_module(ep)
        if not module or is_builtin_module(module):
            continue
        package = module.rsplit(".", 1)[0] if "." in module else module
        owners.append((package, entry_point_distribution(ep), False))
    return owners


def plan(registry: Any) -> PreloadPlan:
    """Merge :data:`CORE_MODULES` with the declarations the template honours.

    *registry* is the template's own, after runner-tier discovery.  An
    out-of-tree owner is honoured only when its distribution is listed in
    ``JAATO_PLUGIN_ALLOW_PRELOAD``; it is consulted BEFORE the declaration
    is read, because reading an out-of-tree provider's literal locates its
    package, which imports that package's parents.  Never raises.
    """
    from jaato_server.shared.plugins.entry_point_trust import (
        ENV_ALLOW_PRELOAD, normalize_distribution, preload_opt_ins,
    )

    result = PreloadPlan()
    modules = list(CORE_MODULES)
    allowed = preload_opt_ins()
    seen = set()
    for package, dist, builtin in (
            _owners_from_registry(registry) + _owners_from_providers()):
        if package in seen:
            continue
        seen.add(package)
        if not builtin and normalize_distribution(dist or "") not in allowed:
            if _safe_has_declaration(package):
                result.ignored[package] = (
                    f"out-of-tree ({dist or 'unknown distribution'}); "
                    f"not listed in {ENV_ALLOW_PRELOAD}")
            continue
        try:
            declared = read_declaration(package)
        except Exception as exc:  # noqa: BLE001 — never fatal
            result.ignored[package] = f"malformed: {exc}"
            continue
        if declared:
            result.declared[package] = declared
            modules.extend(declared)
    result.modules = tuple(dict.fromkeys(modules))
    return result


def _safe_has_declaration(package: str) -> bool:
    """Whether an (unhonoured) out-of-tree owner declares anything.

    Asked only of packages discovery already imported, so the answer never
    imports code: an out-of-tree provider that is not imported reads as
    "no declaration", which only costs the log line saying it was ignored.
    """
    module = sys.modules.get(package)
    return module is not None and hasattr(module, PLUGIN_PRELOAD_ATTR)


@dataclass
class PreloadReport:
    """What :func:`preload` did, for the template's startup log and tests.

    ``loaded`` names entries that imported; ``missing`` names entries that
    are not installed (``ModuleNotFoundError`` for the entry itself);
    ``failed`` maps an entry to the error it raised.  ``new_modules`` is
    how many modules the whole preload added to ``sys.modules``, and
    ``threads`` the live thread count afterwards, which must be one for the
    template to fork safely.
    """

    loaded: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)
    new_modules: int = 0
    threads: int = 1
    elapsed_ms: float = 0.0


def _import_one(name: str, report: PreloadReport) -> None:
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as exc:
        # Only "this entry is not installed" is a quiet skip.  A missing
        # DEPENDENCY of an installed entry is a broken install, which the
        # session would hit too, so it is reported like any other failure.
        if exc.name is not None and (name == exc.name or name.startswith(exc.name + ".")):
            report.missing.append(name)
            log.debug("template preload: %s not installed; skipped", name)
            return
        report.failed[name] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 — a preload must never be fatal
        report.failed[name] = f"{type(exc).__name__}: {exc}"
    else:
        report.loaded.append(name)
        return
    log.warning(
        "template preload: %s failed (%s); sessions will import it "
        "themselves", name, report.failed[name],
    )


def preload(modules: Iterable[str]) -> PreloadReport:
    """Import *modules* into this (template) process, then freeze the heap.

    Called once by the template after plugin discovery and before it
    serves any fork request.  Never raises.

    After the imports, ``gc.freeze()`` moves every tracked object into
    the permanent generation.  Without it, the first collection in each
    forked slot writes to the GC header of every object it inherited,
    copying those pages into the slot's private memory and undoing most of
    what the preload shares.  Freezing does not leak: the frozen objects
    are the imported modules' own, alive for the life of the process
    anyway.
    """
    report = PreloadReport()
    started = time.perf_counter()
    before = len(sys.modules)
    for name in modules:
        _import_one(name, report)
    gc.collect()
    gc.freeze()
    report.new_modules = len(sys.modules) - before
    report.threads = threading.active_count()
    report.elapsed_ms = (time.perf_counter() - started) * 1000
    return report


def log_report(report: PreloadReport, preload_plan: PreloadPlan) -> None:
    """Write the template's preload lines, and the thread warning."""
    for package, reason in sorted(preload_plan.ignored.items()):
        log.info("template preload: %s declaration ignored: %s",
                 package, reason)
    log.info(
        "runner template preload: %d/%d modules imported in %.1fms "
        "(%d declared by packages; +%d in sys.modules; "
        "not installed: %s; failed: %s)",
        len(report.loaded),
        len(report.loaded) + len(report.missing) + len(report.failed),
        report.elapsed_ms,
        len(preload_plan.declared),
        report.new_modules,
        ", ".join(report.missing) or "none",
        ", ".join(sorted(report.failed)) or "none",
    )
    if report.threads != 1:
        log.error(
            "runner template preload left %d threads alive (%s); pool "
            "slots fork from this process, and a thread alive at fork "
            "time is not copied while any lock it holds is",
            report.threads,
            ", ".join(t.name for t in threading.enumerate()),
        )
