"""Daemon entry point: ``python -m jaato_server``.

This is the renamed successor of ``python -m server`` (which no longer
resolves). It delegates to the daemon ``main`` in
:mod:`jaato_server.server.__main__`; the ``jaato-server`` console script
calls the same function. See that module's docstring for the flags.
"""

from jaato_server.server.__main__ import main

if __name__ == "__main__":
    main()
