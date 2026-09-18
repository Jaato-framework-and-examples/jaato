"""Marking AI-generated output -- Regulation (EU) 2024/1689, Art. 50(2).

Article 50(2) obliges the provider of a generative AI system to ensure
its outputs are "marked in a machine-readable format and detectable as
artificially generated or manipulated", with techniques that are
"effective, interoperable, robust and reliable as far as this is
technically feasible".  In force since 2 August 2026 for systems put
into service after that date, and from 2 December 2026 for ones already
on the market.

jaato had the machine-readable half and only that half.
``ToolOutputEvent.generated_by`` (protocol 1.14) stamps the model's own
media on the wire -- and the wire is where it stops.  The moment a
client writes those bytes to a file, the fact that a model produced them
is gone, and the Article is about the OUTPUT, not about the event that
carried it.

This module is the seam between the two.  It owns:

* :class:`OutputPayload` -- what a marker is handed.  One shape for
  bytes on the wire and bytes landing on disk, because the difference
  between them is exactly one optional field (``path``) and a marker
  that writes a sidecar is entitled to decline the first.
* :class:`MarkResult` -- what a marker answers.  Never just the payload:
  a caller has to be able to record that marking was ATTEMPTED and
  declined, which is a different fact from no marker being configured.
* :func:`provenance_manifest` -- the in-tree marker's document, kept
  here rather than in the plugin so the shape is readable from the SDK
  by whoever has to verify one.

Stdlib only, and in the SDK rather than in ``shared`` so an out-of-tree
plugin can implement the contract without depending on jaato-server
(#917's rule: the plugin contract is SDK-shaped).

**What the in-tree manifest is not.**  It is C2PA-*shaped* -- it carries
the IPTC ``trainedAlgorithmicMedia`` digital-source token and a
C2PA-style actions assertion, so a reader who knows C2PA recognises it
-- and it is NOT a signed C2PA manifest.  Signing needs a certificate
and a library; a document that looked signed and was not would be worse
than one that plainly says it is not, because the whole value of a
provenance record is that it cannot be forged.  ``signature`` is
therefore present and ``None``, and the file is named
``<file>.provenance.json`` rather than ``<file>.c2pa.json`` so the name
cannot be read as a conformance claim either.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

#: The IPTC digital-source-type code for content created by a generative
#: model.  The machine-readable token regulators and the C2PA ecosystem
#: actually read; carrying it is most of what makes a manifest
#: "interoperable" in the Article's sense.
IPTC_TRAINED_ALGORITHMIC_MEDIA = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
)

#: Bumped when the manifest's SHAPE changes in a way a reader must notice.
MANIFEST_VERSION = "1"

#: The suffix appended to the subject file's name.  Deliberately not
#: ``.c2pa.json``: see the module docstring.
SIDECAR_SUFFIX = ".provenance.json"


@dataclass(frozen=True)
class OutputPayload:
    """What a :data:`~jaato_sdk.plugins.base.TRAIT_OUTPUT_MARKER` plugin is handed.

    Attributes:
        data: The bytes themselves.  A marker that rewrites them (an
            audio watermarker) returns new bytes on the result; one that
            writes a sidecar leaves them alone.
        mime_type: What the bytes are.  A marker decides from this
            whether it applies at all.
        generated_by: The provenance stamp
            (:func:`jaato_sdk.events.ai_generated_by`).  **Never
            ``None`` at a marker's door** -- the framework refuses to
            invoke a marker on an unstamped payload, because a relayed
            file is not AI-generated because an agent relayed it.
        path: Where the bytes are landing, when they are landing
            anywhere.  ``None`` for a payload that exists only on the
            wire, which a sidecar marker cannot do anything with and
            must decline rather than fail on.
        display_name: A human name for the payload, when one is known.
    """

    data: bytes
    mime_type: str
    generated_by: Dict[str, Any]
    path: Optional[str] = None
    display_name: Optional[str] = None


@dataclass(frozen=True)
class MarkResult:
    """What a marker answers.

    Three states, and keeping them apart is the point.  ``marked=True``
    says a marking exists.  ``marked=False`` with a ``detail`` says a
    marker looked and declined -- an audio watermarker handed a PDF, a
    sidecar marker handed bytes with nowhere to put a sidecar.  A caller
    that got no ``MarkResult`` at all is in the third state: no marker
    was configured, and nothing was even attempted.

    Collapsing the second into the third is how a deployment ends up
    believing its output is marked because a marker is installed.

    Attributes:
        data: The payload's bytes, rewritten by the marker or unchanged.
        marked: Whether a marking now exists for this payload.
        marker: The name of the marker that answered.
        detail: One line saying what it did, or why it declined.
        sidecar_path: Where a sidecar was written, when one was.
    """

    data: bytes
    marked: bool
    marker: str
    detail: str = ""
    sidecar_path: Optional[str] = None


def provenance_manifest(
    payload: OutputPayload,
    framework_version: str,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """The in-tree marker's manifest for one payload.

    Args:
        payload: What is being marked.
        framework_version: The jaato version that did the marking.
        now: The instant to record.  A parameter rather than a call to
            ``utcnow()`` so a test can state the instant it means
            instead of betting on a clock (#996), and so two markings
            of one batch can share a timestamp.

    Returns:
        A JSON-serialisable dict.  ``signature`` is present and ``None``
        deliberately: see the module docstring.
    """
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return {
        "jaato_output_marker": MANIFEST_VERSION,
        "conformance": "c2pa-shaped-unsigned",
        "marked_at": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "subject": {
            "name": payload.display_name or "",
            "media_type": payload.mime_type,
            "size": len(payload.data),
            "sha256": hashlib.sha256(payload.data).hexdigest(),
        },
        "generated_by": dict(payload.generated_by),
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {"actions": [{
                    "action": "c2pa.created",
                    "digitalSourceType": IPTC_TRAINED_ALGORITHMIC_MEDIA,
                }]},
            },
        ],
        "framework": {"name": "jaato", "version": framework_version},
        "signature": None,
    }


def manifest_bytes(manifest: Dict[str, Any]) -> bytes:
    """Serialise a manifest to the bytes that go on disk.

    Sorted keys and a fixed separator set, so two markings of the same
    payload produce the same file -- which is what lets a verifier
    compare one against a recomputed one, and what stops a regenerated
    sidecar showing up as a diff.
    """
    return json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=False,
    ).encode("utf-8") + b"\n"


def sidecar_path_for(path: str) -> str:
    """Where the sidecar for ``path`` goes.

    Appended rather than substituted for the extension, so
    ``chart.png`` and ``chart.pdf`` in one directory do not collide on
    one manifest -- and so the subject is recoverable from the sidecar's
    own name.
    """
    return path + SIDECAR_SUFFIX
