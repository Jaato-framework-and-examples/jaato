"""output_marker -- a provenance sidecar beside an AI-generated file.

The one in-tree implementation of
:data:`~jaato_sdk.plugins.base.TRAIT_OUTPUT_MARKER`, the Article 50(2)
marking hook (Regulation (EU) 2024/1689; see
``docs/design/eu-ai-act.md`` §4.3).  ``ToolOutputEvent.generated_by``
stamps model media on the WIRE; this plugin is what makes the fact
survive the bytes being written down, by putting a manifest beside the
file: ``<file>.provenance.json``.

**A sidecar rather than an embedded marking**, for the reason the design
doc gives: embedding needs a library per format (PNG chunks, PDF
metadata, ID3, BWF), and the point of the exercise is that a file can be
checked by someone who does not have jaato.  A JSON document with a
stated shape is the interoperable floor.

**It is not a signed C2PA manifest, and says so.**  It carries the IPTC
``trainedAlgorithmicMedia`` source token and a C2PA-style actions
assertion so a reader who knows C2PA recognises it, and it carries
``"conformance": "c2pa-shaped-unsigned"`` and a ``signature`` of
``None`` so nothing can read it as one.  Signing needs a certificate the
framework has no way to hold for you; a document that looked signed and
was not would be worse than no document, because the value of a
provenance record is exactly that it is hard to forge.

**What it does NOT mark, and why that is the plugin's answer rather than
a gap:**

* *bytes with no filesystem destination* -- model audio streaming to a
  client.  A sidecar marker has nowhere to put a sidecar, so it DECLINES
  (``marked=False`` with a reason), which is a different fact from no
  marker being configured and is recorded as one.  Watermarking those
  bytes is the out-of-tree marker's job (§4.10: there is no
  dependency-free implementation and the state of the art moves faster
  than a release).
* *text* -- ``docs/design/eu-ai-act.md`` §4.3 records the posture:
  nothing is marked until the Art. 50(7) code of practice names a
  detection standard for text.
* *anything a tool merely relayed* -- the framework refuses to invoke
  any marker on a payload with no ``generated_by`` stamp.  A fetched
  image is not AI-generated because an agent fetched it.

Enable it in a profile's ``plugins:`` list; it exposes no tools.
"""

# Plugin kind identifier for registry discovery.  "enrichment": it
# provides no tools, only the marking hook.
PLUGIN_KIND = "enrichment"

# Runner-tier: it writes beside a file a runner-tier tool produced, and
# holds no daemon-only state.
PLUGIN_TIER = "runner"

from .plugin import OutputMarkerPlugin, create_plugin

__all__ = [
    "OutputMarkerPlugin",
    "create_plugin",
    "PLUGIN_KIND",
    "PLUGIN_TIER",
]
