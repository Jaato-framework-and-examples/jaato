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

**Keeping it complete.** The guard runs a real ``bootstrap_session`` after
:func:`preload` and fails when the bootstrap imports any module the
template did not, naming it, so a module that becomes a bootstrap import
later is added here rather than silently moving back into every slot.
"""

from __future__ import annotations

import gc
import importlib
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

log = logging.getLogger("jaato_server.server.runner.template")


#: Modules every ``session.bootstrap`` imports.  Ordered roughly by cost so
#: the log's timing reads top-down; order does not affect correctness.
BOOTSTRAP_MODULES: Tuple[str, ...] = (
    # The session stack: JaatoRuntime -> JaatoSession -> retry_utils ->
    # anthropic (+ pydantic model construction) and the model_provider
    # packages it imports for error classification.
    "jaato_server.server.runner.session",
    "jaato_server.shared.jaato_client",
    "jaato_server.shared.jaato_runtime",
    "jaato_server.shared.jaato_session",
    "anthropic",
    # The MCP plugin imports these on its own thread at initialize().
    "mcp",
    "mcp.client.stdio",
    "jaato_server.shared.mcp_context_manager",
    # Imported lazily from bootstrap steps and plugin initialize().
    "jaato_server.server.confinement.apparmor",
    "jaato_server.server.runner.lsm_confine",
    "jaato_server.server.runner.slot_plugins",
    "jaato_server.shared.ai_disclosure",
    "jaato_server.shared.capability_drop",
    "jaato_server.shared.completion_schema_loader",
    "jaato_server.shared.event_bus_tools",
    "jaato_server.shared.jaato_self_shadowing",
    "jaato_server.shared.lifecycle_tools",
    "jaato_server.shared.plugins.notebook.backends.subprocess_kernel",
    "jaato_server.shared.plugins.notebook.kernel_protocol",
    "jaato_server.shared.plugins.references.entry_handler",
    "jaato_server.shared.plugins.session.serializer",
    "jaato_server.shared.plugins.subagent.entry_handler",
    "jaato_server.shared.repo_guidance",
    "jaato_server.shared.seccomp_filter",
)

#: Provider SDKs and the provider packages built on them.  Imported at the
#: session's first provider creation, so whichever one a session uses would
#: otherwise land in its slot's private memory.  Installed only through
#: extras, hence best effort: absent ones are skipped.
PROVIDER_MODULES: Tuple[str, ...] = (
    "openai",
    "jaato_server.shared.plugins.model_provider._openai_compat",
    "jaato_server.shared.plugins.model_provider.openrouter",
    "google.genai",
    "jaato_server.shared.plugins.model_provider.google_genai",
)

PRELOAD_MODULES: Tuple[str, ...] = BOOTSTRAP_MODULES + PROVIDER_MODULES


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


def preload(modules: Iterable[str] = PRELOAD_MODULES) -> PreloadReport:
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


def log_report(report: PreloadReport) -> None:
    """Write the template's one preload line, and the thread warning."""
    log.info(
        "runner template preload: %d/%d modules imported in %.1fms "
        "(+%d in sys.modules; not installed: %s; failed: %s)",
        len(report.loaded),
        len(report.loaded) + len(report.missing) + len(report.failed),
        report.elapsed_ms,
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
