"""jaato_server — the jaato orchestration runtime and daemon.

This distribution ships two importable subpackages under one namespace:

- :mod:`jaato_server.server` — the daemon, session manager, IPC/WebSocket
  servers, runner subprocess, and the ``jaato-server`` / ``jaato-scaffold``
  console-script entry points.
- :mod:`jaato_server.shared` — the UI-agnostic core: runtime, sessions,
  plugins, model providers, token accounting, scaffold.

The daemon is started with ``python -m jaato_server`` (or the unchanged
``jaato-server`` console script); ``jaato_server.__main__`` delegates to the
daemon entry point. Before the 1.0 namespace rename these lived as top-level
``server`` / ``shared`` packages; those names no longer resolve — see the
``jaato-doctor`` package-layout check.
"""
