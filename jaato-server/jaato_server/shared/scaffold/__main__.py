"""``python -m jaato_server.shared.scaffold`` — and the import path this CLI had before #1267.

The ``jaato-scaffold`` shell moved to jaato-sdk (:mod:`jaato_sdk.scaffold.cli`)
and jaato-server's four introspection verbs to :mod:`.introspection_verbs`.
Run as a module, this starts the shell with those verbs; imported, it IS
:mod:`.introspection_verbs` (the same module object), so every existing
``from jaato_server.shared.scaffold.__main__ import ...`` and every monkeypatch
of its names keeps meaning what it meant.
"""

import sys

from . import introspection_verbs as _verbs

if __name__ == "__main__":
    raise SystemExit(_verbs.main())

sys.modules[__name__] = _verbs
