"""Grader adapters — three existing mechanisms, one Verdict out.

Nothing here invents a way of judging an agent.  Each adapter wraps
something the framework or the surrounding repos already execute:

``script``     a command run against the mutated workspace — the shape
               ``jaato-cascade-based-prototype`` uses when it settles a
               codegen iteration with ``mvn clean compile``.
``processor``  a completion processor, the framework's own
               ``validate(payload, context) -> list[str]`` contract, run
               post-hoc as a grader instead of in-band as a retry gate.
``judge``      a jaato session whose profile declares a rubric
               ``completion_payload_schema`` — the prototype's
               ``build_judge`` stage, generalised.
"""
from __future__ import annotations

from typing import Dict, Type

from .base import Grader, GraderContext
from .judge import JudgeGrader
from .processor import ProcessorGrader
from .script import ScriptGrader

#: Manifest ``kind`` -> adapter.  ``manifest.GRADER_KINDS`` must agree
#: with these keys; ``tests/test_graders.py`` executes that comparison so
#: the two copies cannot drift silently.
REGISTRY: Dict[str, Type[Grader]] = {
    "script": ScriptGrader,
    "processor": ProcessorGrader,
    "judge": JudgeGrader,
}

__all__ = ["Grader", "GraderContext", "REGISTRY",
           "ScriptGrader", "ProcessorGrader", "JudgeGrader"]
