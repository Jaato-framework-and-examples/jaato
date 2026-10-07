"""The provenance-sidecar marker (EU AI Act, Art. 50(2); #1117).

See the package docstring for the design.  This module implements one
method -- ``mark_output`` -- and the small amount of policy around it:
which mime types this marker claims, what it does when it has nowhere
to write, and how a failure is reported without losing the payload.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from jaato_sdk.output_marking import (
    MarkResult,
    OutputPayload,
    manifest_bytes,
    provenance_manifest,
    sidecar_path_for,
)
from jaato_sdk.plugins.base import TRAIT_OUTPUT_MARKER

logger = logging.getLogger(__name__)

#: What this marker claims.  Image and PDF, per ``docs/design/eu-ai-act.md``
#: §4.3 -- the two kinds for which a sidecar is the recognised idiom and
#: for which no in-tree producer exists yet, so the contract is what
#: forces the next one to stamp.  Audio is deliberately absent: a sidecar
#: beside a streamed utterance marks nothing anybody will read, and
#: watermarking it is the out-of-tree marker's job.
DEFAULT_MARKED_MIMES = ("image/", "application/pdf")


def _framework_version() -> str:
    """The installed jaato-server version, or ``"unknown"``.

    Read from distribution metadata rather than a constant, the rule
    :func:`shared.scaffold.releases.installed_jaato_dists` follows:
    a hardcoded version in a provenance record is a fact that goes stale
    silently, which is the one thing a provenance record may not do.
    """
    try:
        from importlib.metadata import version
        return version("jaato-server")
    except Exception:  # noqa: BLE001 -- an uninstalled tree is not an error
        return "unknown"


class OutputMarkerPlugin:
    """Writes ``<file>.provenance.json`` beside an AI-generated file."""

    plugin_traits = frozenset({TRAIT_OUTPUT_MARKER})

    def __init__(self) -> None:
        self._initialized = False
        self._marked_mimes = tuple(DEFAULT_MARKED_MIMES)

    @property
    def name(self) -> str:
        return "output_marker"

    @property
    def description(self) -> str:
        return ("Writes a provenance sidecar beside files an AI generated "
                "(EU AI Act Art. 50(2))")

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        if self._initialized:
            return
        cfg = config or {}
        mimes = cfg.get("marked_mimes")
        if mimes:
            if not isinstance(mimes, (list, tuple)) or not all(
                    isinstance(m, str) and m for m in mimes):
                raise ValueError(
                    "output_marker.marked_mimes must be a list of non-empty "
                    f"mime prefixes, got {mimes!r}")
            self._marked_mimes = tuple(mimes)
        self._initialized = True

    def shutdown(self) -> None:
        self._initialized = False

    def reset_for_next_session(self) -> None:
        """Nothing to clear: this plugin holds no per-session state.

        Not a reason to declare ``TRAIT_SLOT_SCOPED``.  The trait means
        "everything I hold is deliberately cross-session", and what this
        holds is its configured mime set -- which belongs to the PROFILE
        that enabled it, so carrying the instance to the next session
        would carry one profile's marking policy into another's.
        """

    # -- the enrichment contract: this plugin subscribes to nothing ------
    #
    # ``EnrichmentPlugin`` asks for at least one subscription.  This one
    # has none: its hook is ``mark_output``, invoked by the session at
    # the delivery seams rather than by the enrichment pipeline.  Saying
    # so explicitly beats leaving the methods off and having the registry
    # report a plugin that subscribes to nothing as a mistake.

    def subscribes_to_prompt_enrichment(self) -> bool:
        return False

    def subscribes_to_system_instruction_enrichment(self) -> bool:
        return False

    def subscribes_to_tool_result_enrichment(self) -> bool:
        return False

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "marked_mimes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Mime prefixes this marker claims. Default "
                        f"{list(DEFAULT_MARKED_MIMES)} -- the kinds for which "
                        "a sidecar is the recognised idiom. Audio is absent "
                        "on purpose: a sidecar beside a streamed utterance "
                        "marks nothing anybody reads, and watermarking it "
                        "needs a dependency this tree does not carry."),
                },
            },
        }

    # ------------------------------------------------------- the hook

    def mark_output(self, payload: OutputPayload) -> MarkResult:
        """Write a provenance sidecar for ``payload``, or say why not.

        Never raises.  A marker that lost the payload it was asked to
        mark would be a stronger posture than Article 50(2) asks for and
        would take down every session that produced a file on a
        transient disk error -- so a failure is a ``MarkResult`` with
        ``marked=False`` and a reason, and the caller delivers the bytes
        unmarked.
        """
        if not payload.path:
            return MarkResult(
                data=payload.data, marked=False, marker=self.name,
                detail=("no filesystem destination -- this marker writes a "
                        "sidecar, and bytes on the wire have nowhere to put "
                        "one"),
            )
        mime = (payload.mime_type or "").split(";", 1)[0].strip().lower()
        if not any(mime.startswith(pref) for pref in self._marked_mimes):
            return MarkResult(
                data=payload.data, marked=False, marker=self.name,
                detail=(f"mime {mime!r} is outside this marker's set "
                        f"{list(self._marked_mimes)}"),
            )

        target = sidecar_path_for(payload.path)
        try:
            body = manifest_bytes(provenance_manifest(
                payload, _framework_version(),
                now=datetime.now(timezone.utc)))
            directory = os.path.dirname(target)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(target, "wb") as handle:
                handle.write(body)
        except Exception as exc:  # noqa: BLE001 -- see the docstring
            logger.warning(
                "output_marker: could not write the Art. 50(2) provenance "
                "sidecar for %s (%s); the file is delivered UNMARKED",
                payload.path, exc,
            )
            return MarkResult(
                data=payload.data, marked=False, marker=self.name,
                detail=f"sidecar write failed: {exc}",
            )

        return MarkResult(
            data=payload.data, marked=True, marker=self.name,
            detail=f"provenance sidecar written to {target}",
            sidecar_path=target,
        )


def create_plugin() -> OutputMarkerPlugin:
    """Factory used by ``PluginRegistry.discover()``."""
    return OutputMarkerPlugin()
