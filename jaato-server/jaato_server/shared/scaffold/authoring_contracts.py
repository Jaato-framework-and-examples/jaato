"""Moved to :mod:`jaato_sdk.scaffold.authoring_contracts` (#1267); an alias of it.

The module object is the SDK's, so imports, attribute reads and monkeypatches
through this path reach the one implementation.  Kept runnable because the
snapshot is regenerated from jaato-server's live tree, and this is the command
the docs, the pre-commit hook and the guard name::

    python -m jaato_server.shared.scaffold.authoring_contracts --write
"""

import sys

from jaato_sdk.scaffold import authoring_contracts as _moved

if __name__ == "__main__":
    raise SystemExit(_moved.main())

sys.modules[__name__] = _moved
