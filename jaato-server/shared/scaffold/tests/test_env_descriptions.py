"""`explain env` descriptions come from `# env: ...` comments at the read site."""

from shared.scaffold import introspect


def test_env_doc_comment_parsed_at_read_line():
    src = (
        "import os\n"
        'x = os.environ.get("JAATO_FOO")  # env: turns the foo knob on\n'
        'y = os.environ.get("JAATO_BAR")  # not a doc comment\n'
    )
    docs = introspect._env_doc_comments(src)
    assert docs.get(2) == "turns the foo knob on"
    assert 3 not in docs                      # plain comment ignored


def test_env_doc_variants():
    for line, expect in [
        ("# env: hi", "hi"),
        ("#env:hi", "hi"),
        ("#   env:   spaced   ", "spaced"),
        ("# not env: nope", None),
    ]:
        docs = introspect._env_doc_comments(f"x = 1  {line}\n")
        assert docs.get(1) == expect, line


def test_seeded_var_has_description_end_to_end():
    # The egress flag was annotated with an `# env:` comment in wireup.py;
    # env_vars() must surface that as its description.
    ev = introspect.env_vars().get("JAATO_EGRESS_NFT_ENFORCE")
    assert ev is not None and ev.description
    assert "egress" in ev.description.lower()


def test_undocumented_var_has_none_description():
    # A var with no `# env:` comment stays undocumented (graceful, no fallback).
    evs = introspect.env_vars()
    # PATH is read somewhere but never `# env:`-annotated.
    if "PATH" in evs:
        assert evs["PATH"].description is None
