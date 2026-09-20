"""The :class:`Reversion` declaration, and nothing that costs anything.

WHY THIS IS ITS OWN MODULE.  ``Reversion`` used to live in
``test_every_guard_detects_its_own_reversion``, and 157 guard modules
import it.  That module builds its case list at import time::

    _CASES = [
        (name, mod, rev)
        for name, mod in _guard_modules()      # imports all 157 modules
        for rev in mod.REVERSIONS
    ]

so importing ONE guard module imported every other one, plus whatever
each of those pulls in -- ``shared.plugins.references`` reaching
``requests`` was in the measured chain.  Nothing in a guard's own test
run needs any of that.

It was paid 358 times per CI run, because the meta-suite proves each
guard by spawning ``pytest <guard_module>::<test>`` as a subprocess, and
every one of those subprocesses imported the corpus to get a dataclass.
Measured on one guard module, with ``python -X importtime``::

    import pytest                            0.14 s
    pytest --collect-only on ONE file        3.62 s
      of which, this one import              2.60 s

At ~6.6 s per case that is over a third of every case spent importing
modules the case never runs -- roughly 15 minutes of a 40-minute job,
which was 67% of the runner-minutes of a CI run and effectively all of
its wall clock, since every other job finished inside 4 minutes.

So the dataclass moved here and the eager discovery stayed where it
belongs: in the one process that actually walks the corpus.

WHAT THIS MODULE MAY IMPORT: the standard library, and preferably not
much of it.  Its whole purpose is to be cheap enough that 157 modules
can import it without thinking about it, so anything added here is paid
by every guard subprocess.  A helper that needs the framework belongs in
the meta-guard, which already imports the world on purpose.

The rule is enforced rather than remembered:
``test_reversion_module_stays_cheap`` in the meta-guard fails if this
module grows a non-stdlib import, and
``test_no_guard_imports_reversion_from_the_meta_guard`` fails if a guard
module goes back to the slow path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reversion:
    """One way to put a guarded defect back."""
    target: str      #: repo-relative file to edit
    find: str        #: the FIXED text (must be present, exactly once)
    replace: str     #: the BROKEN text it becomes
    because: str     #: what the guard is supposed to notice
    test: str        #: the ONE test that must fail. See below.
    #
    # NAMING THE TEST IS NOT PEDANTRY.  The first version of this asserted
    # only that the MODULE failed, and a module is many tests: neutering
    # one assertion was masked by a sibling in the same file, and the
    # meta-guard reported the decorative guard as working.  That is the
    # over-broad match -- the anchor matched something, just not the thing
    # under test.  Caught only by sabotaging this suite itself.
