"""Shared bundle infrastructure (domain-agnostic).

This package is *not* a plugin — it deliberately omits ``PLUGIN_KIND``
so the registry's directory walker skips it during plugin discovery.
It exists to host the domain-agnostic pieces of bundle management so
that every plugin (references, agents, tasks, profiles, services) can
reuse the same machinery the references plugin pioneered:

* :class:`Bundle` — the on-disk container abstraction: a ``name``, a
  ``directory``, and the ``tier`` it was discovered under.
* :func:`parse_bundle_ref`, :func:`find_bundle`,
  :func:`AmbiguousBundleRefError` — the ``[<scope>:]<name>`` reference
  syntax shared across every domain.
* :func:`resolve_bundle_roots`, :func:`discover_bundles` — two-tier
  (workspace + user) discovery with workspace-shadows-user precedence.
* :func:`write_bundle_manifest` — declare a directory a bundle.
* :mod:`.pack` / :mod:`.unpack` — the archive format, parametric over
  :class:`~.handler.BundleEntryHandler`.

Nothing here knows what a bundle CONTAINS. A domain that needs more
about its own bundles subclasses :class:`Bundle` in its own package;
:class:`shared.plugins.references.bundle.ReferenceBundle` is the worked
example, carrying the embedding sidecar fields, the reconcile mode and
the live matcher that used to sit on the generic dataclass. The only
references-shaped thing left here is a *filename*: the legacy
``embedding_config.json`` is still recognised as a bundle marker (see
:data:`.bundle.LEGACY_BUNDLE_MARKER_FILENAMES`) so every bundle already
on disk keeps loading, and its body is read by the references plugin
and by nothing in this package.
"""

# Intentionally no PLUGIN_KIND — see module docstring.
