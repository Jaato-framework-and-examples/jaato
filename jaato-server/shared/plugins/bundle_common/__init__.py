"""Shared bundle infrastructure (domain-agnostic).

This package is *not* a plugin — it deliberately omits ``PLUGIN_KIND``
so the registry's directory walker skips it during plugin discovery.
It exists to host the domain-agnostic pieces of bundle management so
that future plugins (agents, tasks, profiles, services) can reuse the
same machinery the references plugin pioneered:

* :class:`Bundle` — the on-disk container abstraction (manifest +
  optional sidecar + entries).
* :func:`parse_bundle_ref`, :func:`find_bundle`,
  :func:`AmbiguousBundleRefError` — the ``[<scope>:]<name>`` reference
  syntax shared across every domain.
* :func:`resolve_bundle_roots`, :func:`discover_bundles` — two-tier
  (workspace + user) discovery with workspace-shadows-user precedence.

References-specific concepts (embedding sidecars, drift detection,
``metadata_hash``) remain in :mod:`shared.plugins.references.bundle` and
are *not* re-exported here. ``Bundle``'s embedding fields are populated
only when a references plugin uses them; other domains can leave them
as opaque structural metadata.

The pack / unpack archive format also lives in references today;
moving it here is a follow-up once the per-domain
``BundleEntryHandler`` interface is fleshed out.
"""

# Intentionally no PLUGIN_KIND — see module docstring.
