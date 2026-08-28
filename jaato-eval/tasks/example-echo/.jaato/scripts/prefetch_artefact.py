"""Read the artefact under test and inject it into the judge's prompt.

WHY THIS EXISTS.  Reading a file is a FACT, not a judgement, and routing a
fact through a model's discretion makes it unreliable — measured, roughly
one run in four the judge simply did not call the tool and said so:
"I did not open answer.txt."  Not a tool failure; the call was never made.
No amount of instruction fixes that reliably, because the instruction is
competing with the model's own sense of whether the step is needed.

So the harness reads it.  ``{{!py:}}`` (no ``?``) is MANDATORY: a failure
here raises DynamicInstructionsError and aborts session-prep, so the judge
cannot start without the artefact rather than starting and guessing.  The
model's remaining job is the part that genuinely needs a model — deciding
whether these bytes satisfy the rubric.

Args: one or more workspace-relative paths.  Each is read and fenced with
its own path, so a rubric can name several artefacts.
"""
from pathlib import Path

#: Refuse to inline more than this per file.  A judge does not need a
#: whole build log, and a prompt that swallows one has stopped being a
#: rubric input and become a context-window problem.
_MAX_BYTES = 20_000


def render(context, args) -> str:
    paths = args or ["answer.txt"]
    root = Path(context.workspace_path)
    out = []
    for rel in paths:
        target = root / rel
        if not target.is_file():
            # Stated, not raised: "the artefact is absent" is a legitimate
            # thing for a judge to score, and is exactly what a failing
            # agent produces.  Only an unreadable-for-other-reasons file
            # is an error, and that surfaces as the exception below.
            # OUT OF BAND, and visibly so.  A prose marker inside the
            # section reads as file content: measured, a judge scored an
            # absent file correctly but its reasoning quoted
            # "(absent — no such file...)" as though those were the bytes.
            # Right answer, wrong reading — and a reading that would go
            # wrong on a rubric that cared what the bytes SAID.
            out.append(
                f"### `{rel}`\n"
                f"> HARNESS NOTE — this file does not exist in the "
                f"workspace. There are no bytes; nothing below is content."
            )
            continue
        raw = target.read_bytes()
        truncated = len(raw) > _MAX_BYTES
        text = raw[:_MAX_BYTES].decode("utf-8", errors="replace")
        note = f"\n[truncated at {_MAX_BYTES} bytes of {len(raw)}]" if truncated else ""
        out.append(
            f"### `{rel}` ({len(raw)} bytes)\n"
            f"The fenced block below is the file verbatim.\n"
            f"```\n{text}\n```{note}"
        )
    return "\n\n".join(out)
