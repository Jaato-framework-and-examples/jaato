"""Moved to :mod:`jaato_sdk.scaffold.gitignore` (#1267); this name is an alias of it.

The module object is the SDK's, so imports, attribute reads and monkeypatches
through this path reach the one implementation.
"""

import sys

from jaato_sdk.scaffold import gitignore as _moved

sys.modules[__name__] = _moved
