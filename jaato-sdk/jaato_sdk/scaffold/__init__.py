"""The authoring half of ``jaato-scaffold``, shipped with the SDK (#1267).

``new`` (scaffold clients, profile-sets, gates, a ``.gitignore`` block),
``integration`` (install the jaato-sdk skill where another tool looks for
skills) and the CLI shell (:mod:`.cli`) that owns the ``jaato-scaffold``
console script.  The introspection verbs (``explain``, ``validate``,
``dependencies``, ``releases``) are contributed by jaato-server when it is
installed.

Nothing in this package imports ``jaato_server`` at module level.  Where an
authoring path still needs it (the provider/env-var live scan, profile
resolution, the re-validation of an emitted profile-set, the gate self-probe)
it is imported inside the function, guarded, and either degrades to the
checked-in snapshot or refuses by name.
"""
