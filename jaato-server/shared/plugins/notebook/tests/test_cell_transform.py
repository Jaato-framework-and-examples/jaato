"""Tests for IPython-style `!shell` line support in notebook cells."""

import ast
import io
import sys

from shared.plugins.notebook.cell_transform import (
    transform_cell_source,
    strip_shell_lines,
    inject_shell_helper,
    _run_shell,
    SHELL_HELPER_NAME,
)
from shared.plugins.notebook.code_analyzer import CodeAnalyzer


# ---- transform_cell_source --------------------------------------------------

def test_transform_rewrites_per_line_preserving_indent():
    src = "x = 1\n!echo hi\nif x:\n    !ls -a\n"
    out = transform_cell_source(src)
    assert f"{SHELL_HELPER_NAME}('echo hi')" in out
    assert f"    {SHELL_HELPER_NAME}('ls -a')" in out   # indent preserved
    ast.parse(out)                                       # valid Python
    assert len(out.split("\n")) == len(src.split("\n"))  # line count preserved


def test_non_shell_lines_untouched():
    src = "y = 5 != 3\nprint(y)"
    assert transform_cell_source(src) == src


def test_triple_quoted_string_not_transformed():
    src = 'doc = """\n!not a shell line\n"""\n'
    out = transform_cell_source(src)
    assert SHELL_HELPER_NAME not in out
    ast.parse(out)


# ---- strip_shell_lines (analyzer feed) --------------------------------------

def test_strip_returns_parseable_code_and_line_numbers():
    src = "x = 1\n!echo hi\nif x:\n    !ls -a\n"   # ! lines on 2 and 4
    stripped, cmds = strip_shell_lines(src)
    ast.parse(stripped)                       # rest of the cell still parses
    assert cmds == [(2, "echo hi"), (4, "ls -a")]


def test_strip_noop_for_shell_free_code():
    src = "a = 1\nb = a + 2\n"
    assert strip_shell_lines(src) == (src, [])


# ---- runtime helper ---------------------------------------------------------

def test_run_shell_streams_stdout():
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        rv = _run_shell("echo hello_from_shell")
    finally:
        sys.stdout = old
    assert rv is None
    assert "hello_from_shell" in buf.getvalue()


def test_transformed_cell_execs_and_runs_shell():
    ns = {}
    inject_shell_helper(ns)
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        exec(compile(transform_cell_source("!echo via_exec"), "<cell>", "exec"), ns)
    finally:
        sys.stdout = old
    assert "via_exec" in buf.getvalue()


def test_run_shell_reports_nonzero_exit():
    buf = io.StringIO()
    old = sys.stderr
    sys.stderr = buf
    try:
        _run_shell("exit 3")
    finally:
        sys.stderr = old
    assert "status 3" in buf.getvalue()


# ---- analyzer: per-line ! detection -----------------------------------------

def test_analyzer_flags_each_shell_line_and_analyzes_rest():
    analyzer = CodeAnalyzer()
    result = analyzer.analyze("import os\n!pip install x\nos.getcwd()\n!ls\n")
    shell = [r for r in result.risks if r.category == "subprocess"
             and r.description == "Shell command execution"]
    assert {r.line_number for r in shell} == {2, 4}      # both ! lines flagged
    # the non-! lines were still parsed (analysis didn't bail on SyntaxError)
    assert not any(r.category == "syntax" for r in result.risks)
