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
the live matcher that used to sit on the generic dataclass.

Nothing references-shaped is left here, filenames included. A directory
is a bundle because it carries ``bundle.json`` -- a domain's claim --
never because of what else it happens to contain. The references
plugin's ``embedding_config.json`` sits beside that manifest, describes
a vector index, and marks nothing; a bundle and an index are
independent, which is the whole of #1130.
"""

# Intentionally no PLUGIN_KIND — see module docstring.
