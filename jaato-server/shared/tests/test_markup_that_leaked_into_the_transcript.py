"""Server markup that reached a client as literal tags (#1191, #1193).

Reported as two web-client defects -- ``<j-table>`` tags and ``<nb-row>``
tags "leaking" into the chat as raw text.  Driving the daemon's own
formatter pipeline located two of the three causes UPSTREAM of every
client, which is why the TUI showed the same thing:

**#1191 -- a table inside a code fence.**  ``table_formatter`` runs at
priority 25, before ``code_block_formatter`` (40), and knew nothing about
fences.  A markdown table quoted in a fence -- a ````markdown`` example,
a notebook cell's fenced output -- was rewritten into ``<j-table>``
markup, and ``code_block_formatter`` then escaped that markup into a
``<j-code>`` block.  Every client renders a ``<j-code>`` block
faithfully: monospace, line-numbered, and the literal
``&lt;j-table&gt;`` / ``&lt;j-td&gt;`` text the report's screenshot
shows.  The fix is to leave the inside of a fence alone, deciding "the
inside of a fence" with ``code_block_formatter``'s OWN two patterns, so
the two formatters cannot disagree about which lines are code.

**#1193, the server half -- an error cell with no execution count.**
Every early exit on the notebook's streaming path (no code, a cell the
containment boundary refused, a notebook that could not be created or
does not exist) emits ``<notebook-cell type="error">`` with no ``exec``
attribute, and ``notebook_output_formatter`` required one.  Unmatched,
the raw marker went to every client verbatim -- and the refusals are the
messages a person most needs to read.

The client half of #1193 -- the web client had no ``<nb-row>`` renderer
at all -- is ``jaato-web-coder-ui/src/protocol/nbmarkup.ts`` and its
tests.

Every positive case here is paired with the neighbouring input that
already worked, because a formatter that emits nothing at all would pass
a test that only asserts an absence.
"""
from __future__ import annotations

import ast
import random
import re
from pathlib import Path
from typing import List

import pytest

from shared.plugins.code_block_formatter import plugin as code_block
from shared.plugins.formatter_pipeline.registry import create_default_pipeline
from shared.plugins.notebook_output_formatter import plugin as notebook_fmt
from shared.plugins.table_formatter import plugin as table_fmt
from shared.tests.reversion import Reversion

_TABLE = "jaato-server/shared/plugins/table_formatter/plugin.py"
_NB_FMT = "jaato-server/shared/plugins/notebook_output_formatter/plugin.py"

REVERSIONS = [
    Reversion(
        target=_TABLE,
        find="""            if self._fence_transition(whole_line):""",
        replace="""            if False:""",
        test="test_a_table_quoted_in_a_fence_stays_code",
        because=(
            "the reported defect: a table inside a fence is rewritten into "
            "<j-table> before the code-block formatter escapes the fence, so "
            "every client shows literal, line-numbered markup tags"
        ),
    ),
    Reversion(
        target=_TABLE,
        find="""            whole_line = self._fence_line_prefix + line""",
        replace="""            whole_line = line""",
        test="test_a_fence_opener_split_from_its_newline_still_opens",
        because=(
            "a streamed ```lang with no newline yet is passed straight "
            "through for latency; judging the completed line without that "
            "head means the fence never opens and the table inside leaks"
        ),
    ),
    Reversion(
        target=_TABLE,
        find="""from shared.plugins.code_block_formatter.plugin import Fence, open_fence
""",
        replace="""from shared.plugins.code_block_formatter.plugin import Fence
from shared.plugins.code_block_formatter.plugin import open_fence as _shared_open_fence


def open_fence(line, *, at_line_start=True):
    return _shared_open_fence(line, at_line_start=at_line_start)
""",
        test="test_both_formatters_decide_fences_with_one_definition",
        because=(
            "a second definition of the fence rule agrees with the first only "
            "until somebody edits one of them -- which is how two formatters "
            "start disagreeing about which lines are code"
        ),
    ),
    Reversion(
        target=_NB_FMT,
        find="""(?:\\s+exec="(\\d+)")?>""",
        replace="""\\s+exec="(\\d+)">""",
        test="test_an_error_cell_with_no_execution_count_is_formatted",
        because=(
            "every early exit on the notebook's streaming path emits an error "
            "cell with no exec attribute; requiring one sends the raw "
            "<notebook-cell> marker to every client"
        ),
    ),
]


def _format(chunks: List[str]) -> str:
    """Run ``chunks`` through the daemon's default pipeline, as it streams."""
    pipeline = create_default_pipeline()
    out = "".join(piece for chunk in chunks for piece in pipeline.process_chunk(chunk))
    return out + "".join(pipeline.flush())


FENCED_TABLE = "Here:\n```markdown\n| a | b |\n|---|---|\n| 1 | 2 |\n```\nDone.\n"
BARE_TABLE = "Here:\n| a | b |\n|---|---|\n| 1 | 2 |\nDone.\n"


# ------------------------------------------------------------------ #1191

def test_a_table_quoted_in_a_fence_stays_code():
    out = _format([FENCED_TABLE])
    assert "&lt;j-table" not in out, out
    assert "<j-table>" not in out, out
    # The table is still THERE -- as the code the author quoted.
    assert '<j-code language="markdown">' in out
    assert "| a | b |</j-line>" in out


def test_a_bare_table_still_becomes_a_table():
    """The control: fence-awareness must not switch table rendering off."""
    out = _format([BARE_TABLE])
    assert "<j-table>" in out
    assert "<j-th>a</j-th>" in out


def test_a_table_after_a_closed_fence_is_a_table_again():
    text = "```py\nx = 1\n```\n| c | d |\n|---|---|\n| 3 | 4 |\nend\n"
    out = _format([text])
    assert "<j-th>c</j-th>" in out
    assert '<j-code language="py">' in out


def test_a_fence_opener_split_from_its_newline_still_opens():
    """The head of a line is yielded before its newline arrives."""
    chunks = ["Here:\n```markdown", "\n| a | b |\n|---|---|\n| 1 | 2 |\n```\nDone.\n"]
    assert _format(chunks) == _format([FENCED_TABLE])
    assert "&lt;j-table" not in _format(chunks)


@pytest.mark.parametrize("text", [FENCED_TABLE, BARE_TABLE,
    "pre ```py\nx = '|a|b|'\n|---|---|\n```\n| c | d |\n|---|---|\n| 3 | 4 |\nend\n"])
def test_any_chunking_formats_exactly_like_the_whole_text(text):
    """What a stream is cut into must not change what the client is sent."""
    whole = _format([text])
    assert _format(list(text)) == whole
    for seed in range(20):
        rnd = random.Random(seed)
        chunks, i = [], 0
        while i < len(text):
            n = rnd.randint(1, 7)
            chunks.append(text[i:i + n])
            i += n
        assert _format(chunks) == whole, (seed, chunks)


def test_both_formatters_decide_fences_with_one_definition():
    """The table formatter IMPORTS the fence rule; it does not restate it.

    Asserted on the source, not on behaviour: a local copy agrees with
    the original on every input right up until one of them is edited,
    so no input can tell them apart today.  (The rule was two regexes
    until the fence rewrite; it is ``open_fence`` / ``Fence`` now.)
    """
    tree = ast.parse(Path(table_fmt.__file__).read_text())
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == code_block.__name__
        for alias in node.names
    }
    assert {"open_fence", "Fence"} <= imported, imported
    defined = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert not {"open_fence", "Fence"} & defined, defined
    assert table_fmt.open_fence is code_block.open_fence


# ------------------------------------------------------------------ #1193

def test_an_error_cell_with_no_execution_count_is_formatted():
    out = _format(['<notebook-cell type="error">No code provided</notebook-cell>'])
    assert "<notebook-cell" not in out, out
    assert '<nb-row type="error" label="Err:">' in out
    assert "No code provided" in out


def test_an_error_cell_with_a_count_keeps_its_label():
    """The control: the numbered shape is unchanged."""
    out = _format(['<notebook-cell type="error" exec="3">Traceback x</notebook-cell>\n'])
    assert '<nb-row type="error" label="Err [3]:">' in out


def _non_emitting_string_ids(tree: ast.AST) -> set:
    """Ids of string nodes that are part of the source but emit nothing.

    A docstring that MENTIONS the marker documents it; and the literal
    fragments INSIDE an f-string are visited on their own by ``ast.walk`` --
    the f-string is the emission, a fragment of it is not.
    """
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    docstrings = {
        id(owner.body[0].value)
        for owner in ast.walk(tree)
        if isinstance(owner, owners) and owner.body
        and isinstance(owner.body[0], ast.Expr)
        and isinstance(owner.body[0].value, ast.Constant)
    }
    fragments = {
        id(part)
        for joined in ast.walk(tree) if isinstance(joined, ast.JoinedStr)
        for part in joined.values
    }
    return docstrings | fragments


def _string_text(node: ast.AST):
    """The text a string node emits, an f-string's placeholders stood in."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value if isinstance(part, ast.Constant) else "placeholder"
            for part in node.values
        )
    return None


def _notebook_cell_literals() -> List[str]:
    """Every ``<notebook-cell`` string the notebook plugin can emit.

    Read from the plugin's AST rather than listed here: the defect was an
    emitter shape nobody checked against the formatter, and a list written
    in this file covers only the shapes its author thought of.
    """
    source = Path(__file__).resolve().parents[1] / "plugins" / "notebook" / "plugin.py"
    tree = ast.parse(source.read_text())
    skip = _non_emitting_string_ids(tree)
    found: List[str] = []
    for node in ast.walk(tree):
        text = None if id(node) in skip else _string_text(node)
        if text and re.search(r'<notebook-cell\s+type="', text):
            found.append(text)
    return found


def test_every_notebook_cell_the_plugin_emits_is_one_the_formatter_accepts():
    literals = _notebook_cell_literals()
    assert literals, "found no <notebook-cell emitters -- the scan is broken"
    for literal in literals:
        out = _format([literal])
        assert "<notebook-cell" not in out, (literal, out)
        assert "<nb-row" in out, (literal, out)
