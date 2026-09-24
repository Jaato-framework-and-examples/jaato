"""A model's own prose drawn as a code block, because a fence did not close.

Reported with a screenshot from the web client: the second half of an
ordinary reply -- the model's headings, its questions to the user, its
closing line -- rendered monospaced, line-numbered and syntax-coloured
as one ``<j-code>`` block running to the end of the message.  Every
client renders a ``<j-code>`` block faithfully, so the defect is where
the block is decided: ``code_block_formatter``.

Its opener was ``\\`\\`\\`(\\w*)\\n`` and its closer any line STARTING with
three backticks.  So an opener the pattern did not recognise was passed
through as text and ITS closer opened a block with nothing after it to
close it:

* an info string that is not one bare word -- ``c++``,
  ``shell-session``, ``text`` followed by a space;
* a four-backtick fence quoting a three-backtick one -- the inner fence
  closed the outer, and the parity of every later fence flipped;
* a closer indented inside a list item -- never seen at all.

The rule is CommonMark's now: an opener at a line start takes any info
string, a closer is a line holding only a run of the SAME character at
least as long as the opener's, and both may be indented.  Every positive
case is paired with a control, and every fixture is also streamed one
character at a time and in random pieces, because the formatter's whole
job is to decide this without seeing the rest of the reply.
"""
from __future__ import annotations

import random
import re
from typing import List

import pytest

from jaato_server.shared.plugins.formatter_pipeline.registry import create_default_pipeline
from jaato_server.shared.tests.reversion import Reversion

_CODE = "jaato-server/jaato_server/shared/plugins/code_block_formatter/plugin.py"

REVERSIONS = [
    Reversion(
        target=_CODE,
        find="""        return run[0] == self.char and len(run) >= self.length""",
        replace="""        return run[0] == self.char""",
        test="test_a_long_fence_quotes_a_shorter_one",
        because=(
            "a three-backtick line inside a four-backtick fence closes it, "
            "and every fence after that is read with the wrong parity"
        ),
    ),
    Reversion(
        target=_CODE,
        find="""_LINE_START_OPEN = re.compile(r'^(?P<indent>[ \\t]*)(?P<fence>`{3,}|~{3,})(?P<info>.*)$')""",
        replace="""_LINE_START_OPEN = re.compile(r'^(?P<indent>[ \\t]*)(?P<fence>`{3,}|~{3,})(?P<info>\\w*)$')""",
        test="test_an_opener_with_any_info_string_opens",
        because=(
            "an opener whose info string is not one bare word is passed "
            "through as text, and its closer opens a block that swallows "
            "the rest of the reply -- the reported screenshot"
        ),
    ),
    Reversion(
        target=_CODE,
        find="""_CLOSE = re.compile(r'^[ \\t]*(?P<fence>`{3,}|~{3,})[ \\t]*$')""",
        replace="""_CLOSE = re.compile(r'^(?P<fence>`{3,}|~{3,})[ \\t]*$')""",
        test="test_a_closer_indented_in_a_list_closes",
        because=(
            "a fence inside a list item is indented, and an unindented-only "
            "closer never sees it"
        ),
    ),
]

PROSE = "Once you answer, I will post the comment."


def _format(chunks: List[str]) -> str:
    """Run ``chunks`` through the daemon's default pipeline, as it streams."""
    pipeline = create_default_pipeline()
    out = "".join(piece for chunk in chunks for piece in pipeline.process_chunk(chunk))
    return out + "".join(pipeline.flush())


def _outside_code(out: str) -> str:
    """The output with every ``<j-code>`` block removed."""
    return re.sub(r"<j-code\b.*?</j-code>", "", out, flags=re.S)


def _text(out: str) -> str:
    """The output as a reader sees it: tags removed, entities kept."""
    return re.sub(r"<[^>]+>", "", out)


def _assert_prose_is_prose(text: str) -> str:
    out = _format([text])
    assert PROSE in _outside_code(out), out
    return out


@pytest.mark.parametrize("info", ["c++", "shell-session", "text ", "python title=\"x.py\""])
def test_an_opener_with_any_info_string_opens(info):
    text = f"Draft:\n```{info}\nint x;\n```\n\n{PROSE}\n"
    out = _assert_prose_is_prose(text)
    # And the code is still code, with the language's first word.
    assert f'<j-code language="{info.split()[0]}">' in out, out


def test_a_long_fence_quotes_a_shorter_one():
    text = ("Here is the comment I will post:\n"
            "````markdown\n## Proposal\n```yaml\na: 1\n```\nmore draft\n````\n\n"
            f"{PROSE}\n")
    out = _assert_prose_is_prose(text)
    # One block, holding the quoted fence as its content.
    assert out.count("<j-code") == 1, out
    assert "more draft" in _text(out) and "more draft" not in _outside_code(out)


def test_a_closer_indented_in_a_list_closes():
    text = f"1. Run it:\n   ```bash\n   ls -la\n   ```\n2. {PROSE}\n"
    out = _assert_prose_is_prose(text)
    # The list's indent is not part of the code.
    code = _text(re.search(r"<j-code\b.*?</j-code>", out, re.S).group(0))
    assert "ls -la" in code and "   ls" not in code, out


def test_a_line_that_merely_starts_with_a_fence_does_not_close():
    """``` ```python`` inside a block is content, not its closer."""
    text = f"````\n```python\nx = 1\n````\n{PROSE}\n"
    out = _assert_prose_is_prose(text)
    assert out.count("<j-code") == 1


def test_the_controls_still_behave_as_they_did():
    """A plain block, the mid-line opener, and a block left open."""
    plain = _format([f"Look:\n```py\nx = 1\n```\n{PROSE}\n"])
    assert '<j-code language="py">' in plain and PROSE in _outside_code(plain)

    mid = _format([f"pre ```py\nx = 1\n```\n{PROSE}\n"])
    assert '<j-code language="py">' in mid and PROSE in _outside_code(mid)

    # A reply that forgets its closer still loses nothing.
    unclosed = _format(["```py\nx = 1\n"])
    assert "x = 1" in _text(unclosed) and "<j-code" in unclosed

    # A closer with no newline after it closes at flush, not as code.
    tail = _format(["```py\nx = 1\n```"])
    assert "```" not in tail, tail


FIXTURES = [
    f"Draft:\n```c++\nint x;\n```\n\n{PROSE}\n",
    f"Here:\n````markdown\n## P\n```yaml\na: 1\n```\nmore\n````\n\n{PROSE}\n",
    f"1. Run it:\n   ```bash\n   ls -la\n   ```\n2. {PROSE}\n",
    f"pre ```py\nx = '|a|b|'\n```\n{PROSE} `inline` and ~~struck~~\n",
    f"~~~text\nbody\n~~~\n{PROSE}\n",
]


@pytest.mark.parametrize("text", FIXTURES)
def test_any_chunking_formats_exactly_like_the_whole_text(text):
    whole = _format([text])
    assert PROSE in _outside_code(whole)
    assert _format(list(text)) == whole
    for seed in range(20):
        rnd = random.Random(seed)
        chunks, i = [], 0
        while i < len(text):
            n = rnd.randint(1, 7)
            chunks.append(text[i:i + n])
            i += n
        assert _format(chunks) == whole, (seed, chunks)
