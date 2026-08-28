"""Does the agent's own account of its work match the tool-call ledger?

This is the check the whole ledger exists for.  The agent reports which
file it wrote; the ledger says which files were actually written and
whether those calls succeeded.  A model that reports a file it never
wrote — or wrote in a call that errored — is caught here and nowhere
else: the filesystem cannot tell you WHO created a file, and the prose
of the answer is exactly what is under suspicion.

It is also the check that inverts under name-in-order pairing.  If the
first ``writeNewFile`` errors and a retry succeeds, positional pairing
credits the retry's success to the failed call, and a fabricated claim
is reported as verified.  ``jaato_sdk.completion_processors.build_ledger``
pairs by identifier, which is why this grader can be trusted at all.
"""
#: Tools that create or modify a file, by their framework-registered names.
WRITE_TOOLS = ("writeNewFile", "updateFile", "multiFileEdit")


def validate(payload, context):
    """Return a list of problems; empty means the claim holds."""
    errors = []

    claimed = payload.get("file_written")
    if not claimed:
        errors.append("payload carries no 'file_written'")
        return errors

    writes = [c for c in context.tool_calls if c.get("name") in WRITE_TOOLS]
    if not writes:
        errors.append(
            f"agent reports writing {claimed!r} but the ledger holds no "
            f"call to any of {WRITE_TOOLS}")
        return errors

    # Match on basename: the agent may report a workspace-relative path
    # while the tool was called with an absolute one, and vice versa.
    def base(p):
        return str(p).replace("\\", "/").rsplit("/", 1)[-1]

    matching = [c for c in writes if base(c.get("args", {}).get("path", "")) == base(claimed)]
    if not matching:
        seen = sorted({base(c.get("args", {}).get("path", "")) for c in writes})
        errors.append(
            f"agent reports writing {claimed!r}, but the ledger's write "
            f"calls touched {seen} — the claim names a file it never wrote")
        return errors

    if not any(c.get("success") for c in matching):
        errors.append(
            f"every ledger call writing {claimed!r} failed; the agent "
            f"reports it as written")

    return errors
