"""Bootstrap timing instrumentation for profiling session startup.

Provides a lightweight timer hierarchy that records wall-clock durations
for each stage and sub-stage of the bootstrap process.  The results can
be printed as a human-readable report or returned as structured data for
programmatic consumption.

Usage in JaatoServer.initialize()::

    from shared.bootstrap_timing import BootstrapTimer

    timer = BootstrapTimer()
    with timer.stage("load_config"):
        ...
    with timer.stage("connect_provider"):
        ...
    timer.report()  # writes to provided file or sys.stderr

Sub-stages::

    with timer.stage("load_plugins") as stage:
        with stage.sub("discover"):
            registry.discover()
        with stage.sub("expose_all"):
            registry.expose_all(...)

The timer is intentionally low-overhead (uses ``time.perf_counter``)
and has no external dependencies.
"""

import contextlib
import io
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TextIO

logger = logging.getLogger(__name__)


@dataclass
class TimingEntry:
    """A single timed operation (stage or sub-stage).

    Attributes:
        name: Human-readable name of the operation.
        start: ``perf_counter`` timestamp when the operation began.
        end: ``perf_counter`` timestamp when the operation finished,
            or ``None`` if still running.
        children: Ordered list of sub-stage entries.
    """

    name: str
    start: float = 0.0
    end: Optional[float] = None
    children: List["TimingEntry"] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        """Wall-clock seconds for this entry.  Returns 0.0 if not finished."""
        if self.end is None:
            return 0.0
        return self.end - self.start

    @property
    def self_time(self) -> float:
        """Time spent in this entry excluding children."""
        child_total = sum(c.elapsed for c in self.children)
        return max(0.0, self.elapsed - child_total)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict."""
        d: dict = {
            "name": self.name,
            "elapsed_ms": round(self.elapsed * 1000, 2),
            "self_ms": round(self.self_time * 1000, 2),
        }
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


class _StageContext:
    """Context manager returned by ``BootstrapTimer.stage()``.

    Provides a ``.sub()`` method so callers can nest sub-stages.
    """

    def __init__(self, entry: TimingEntry) -> None:
        self._entry = entry

    def __enter__(self) -> "_StageContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._entry.end = time.perf_counter()
        return None

    @contextlib.contextmanager
    def sub(self, name: str):
        """Record a sub-stage within this stage."""
        child = TimingEntry(name=name, start=time.perf_counter())
        self._entry.children.append(child)
        try:
            yield child
        finally:
            child.end = time.perf_counter()

    def mark(self, name: str) -> None:
        """Record a zero-duration milestone (e.g. a checkpoint)."""
        now = time.perf_counter()
        self._entry.children.append(TimingEntry(name=name, start=now, end=now))


class BootstrapTimer:
    """Hierarchical timer for bootstrap profiling.

    Attributes:
        stages: Ordered list of top-level timing entries.
    """

    def __init__(self) -> None:
        self.stages: List[TimingEntry] = []
        self._overall_start: Optional[float] = None
        self._overall_end: Optional[float] = None

    def stage(self, name: str) -> _StageContext:
        """Start timing a top-level stage.

        Args:
            name: Human-readable stage name (e.g. ``"load_plugins"``).

        Returns:
            A context manager that stops the timer on exit and provides
            ``.sub()`` for nested sub-stages.
        """
        entry = TimingEntry(name=name, start=time.perf_counter())
        if self._overall_start is None:
            self._overall_start = entry.start
        self.stages.append(entry)
        return _StageContext(entry)

    def finish(self) -> None:
        """Mark the overall bootstrap as complete."""
        self._overall_end = time.perf_counter()

    @property
    def total_elapsed(self) -> float:
        """Total wall-clock seconds from first stage start to finish()."""
        if self._overall_start is None:
            return 0.0
        end = self._overall_end or time.perf_counter()
        return end - self._overall_start

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, file: Optional[TextIO] = None, threshold_ms: float = 1.0) -> str:
        """Print a human-readable timing report.

        Args:
            file: Output stream (defaults to ``sys.stderr``).
            threshold_ms: Only show sub-stages slower than this.

        Returns:
            The report string (also written to *file*).
        """
        out = io.StringIO()
        total = self.total_elapsed
        out.write("\n")
        out.write("=" * 70 + "\n")
        out.write(f"  BOOTSTRAP TIMING REPORT  (total: {total*1000:.0f} ms)\n")
        out.write("=" * 70 + "\n\n")

        for entry in self.stages:
            pct = (entry.elapsed / total * 100) if total > 0 else 0
            bar = _bar(pct)
            out.write(f"  {entry.name:<35} {entry.elapsed*1000:>8.1f} ms  {pct:>5.1f}%  {bar}\n")

            # Show children sorted by elapsed (slowest first)
            children_sorted = sorted(entry.children, key=lambda c: c.elapsed, reverse=True)
            for child in children_sorted:
                if child.elapsed * 1000 < threshold_ms:
                    continue
                child_pct = (child.elapsed / total * 100) if total > 0 else 0
                out.write(f"    {child.name:<33} {child.elapsed*1000:>8.1f} ms  {child_pct:>5.1f}%\n")

                # Show grandchildren if any (for per-plugin detail)
                gc_sorted = sorted(child.children, key=lambda c: c.elapsed, reverse=True)
                for gc in gc_sorted[:10]:  # Top 10 grandchildren
                    if gc.elapsed * 1000 < threshold_ms:
                        continue
                    out.write(f"      {gc.name:<31} {gc.elapsed*1000:>8.1f} ms\n")

            out.write("\n")

        # Summary: top 5 slowest components across all levels
        all_entries = []
        for s in self.stages:
            for c in s.children:
                all_entries.append((f"{s.name}/{c.name}", c.elapsed))
                for gc in c.children:
                    all_entries.append((f"{s.name}/{c.name}/{gc.name}", gc.elapsed))

        if all_entries:
            out.write("-" * 70 + "\n")
            out.write("  TOP 5 SLOWEST COMPONENTS:\n")
            out.write("-" * 70 + "\n")
            for path, elapsed in sorted(all_entries, key=lambda x: x[1], reverse=True)[:5]:
                out.write(f"    {path:<50} {elapsed*1000:>8.1f} ms\n")
            out.write("\n")

        out.write("=" * 70 + "\n")

        report_text = out.getvalue()
        dest = file or sys.stderr
        dest.write(report_text)
        return report_text

    def to_dict(self) -> dict:
        """Return structured timing data as a dict."""
        return {
            "total_ms": round(self.total_elapsed * 1000, 2),
            "stages": [s.to_dict() for s in self.stages],
        }


def _bar(pct: float, width: int = 20) -> str:
    """Render a simple ASCII percentage bar."""
    filled = int(pct / 100 * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"
