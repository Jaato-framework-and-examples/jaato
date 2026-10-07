"""Moved to :mod:`jaato_sdk.gitignore_parser` (#1267); this name is an alias of it.

``GitignoreParser`` is read by the daemon (the workspace monitor, the
filesystem-query plugin) and by ``jaato-scaffold`` in the SDK, so it lives in
the SDK and the daemon imports it from there.  The module object is the SDK's.
"""

import sys

from jaato_sdk import gitignore_parser as _moved

sys.modules[__name__] = _moved
