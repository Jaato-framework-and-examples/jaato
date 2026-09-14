"""A model-supplied ``extra_paths`` cannot choose the binary (issue #697).

The defect was three facts that only mattered together:

1. ``PermissionPolicy._build_signature`` reads ``command`` and ``args`` and
   nothing else, so the whitelist, the blacklist, every session approval and
   the #951/#968 DECISION trace line are all keyed on the command TEXT.
2. Every execution site read ``extra_paths`` out of the model's tool-call
   arguments — the two ``CliPlugin`` paths (with the operator's configured
   list as a mere *fallback*) and ``server.runner.cli_runner`` (with no
   operator tier at all).
3. It reached ``shutil.which``.

So an approval for ``deploy --prod`` was an approval for whatever ``deploy``
resolved to under a ``PATH`` the model supplied in the same call, and the two
DECISION lines were byte-identical apart from ``call_id``.

The fix answers the issue's first question — *may a model-supplied path
component change binary resolution at all?* — with **no**, at every site
(``shared.cli_path_policy``). That is what makes the issue's second question
moot rather than unanswered: a value the executor refuses cannot vary between
approval and execution, so there is nothing for the signature to key on, and
keying it there anyway would only prompt a human to authorize a call that is
going to be refused.

Four things are asserted, and the first two are the ones that would let the
defect back:

- :func:`test_no_execution_site_reads_extra_paths_from_args` — an AST scan, so
  a *new* site cannot reintroduce the read by copying an old one.
- the three behavioural refusals, one per live execution path.
- :func:`test_the_signature_is_still_blind_to_extra_paths` — the property that
  made the bug possible is recorded as fact rather than assumed to be gone.
  It is why the refusal has to hold; a future change to the signature does not
  make removing the refusal safe.
"""
from __future__ import annotations

import ast
import os
import stat
from pathlib import Path
from typing import Any, Dict, List

import pytest

from server.runner.cli_runner import execute_cli_based_tool
from shared.cli_path_policy import (
    MODEL_REFUSED_ARG,
    REFUSAL_ERROR,
    refuse_model_extra_paths,
)
from shared.plugins.cli.plugin import CLIToolPlugin
from shared.plugins.permission.policy import PermissionPolicy
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


_PLUGIN = "jaato-server/shared/plugins/cli/plugin.py"
_RUNNER = "jaato-server/server/runner/cli_runner.py"

#: Every module that turns ``cli_based_tool`` arguments into a subprocess.
#: A fourth one is exactly what the AST guard exists to catch.
_EXECUTION_SITES = (_PLUGIN, _RUNNER)

ROOT = Path(__file__).resolve().parents[3]


REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find="""            refusal = precheck_cli_args(args, surface='cli plugin')
            if refusal is not None:
                self._trace(f"execute: refused — {refusal['error']}")
                return refusal

            command = args.get('command')
            arg_list = args.get('args')
            extra_paths = self._extra_paths""",
        replace="""            command = args.get('command')
            arg_list = args.get('args')
            extra_paths = args.get('extra_paths', self._extra_paths)

            if not command:
                return {'error': 'cli_based_tool: command must be provided'}""",
        test="test_plugin_execute_refuses_caller_supplied_extra_paths",
        because="the cli plugin honours a model-supplied PATH extension again",
    ),
    Reversion(
        target=_RUNNER,
        find="""    refusal = precheck_cli_args(args, surface="runner cli executor")
    if refusal is not None:
        return False, refusal

    command = args.get("command")""",
        replace="""    command = args.get("command")

    if not command:
        return False, {"error": "cli_based_tool: command must be provided"}""",
        test="test_runner_executor_refuses_caller_supplied_extra_paths",
        because=("the runner-tier cli executor stops refusing a "
                 "caller-supplied extra_paths"),
    ),
]


# --------------------------------------------------------------- AST guard


def _args_get_extra_paths_calls(path: Path) -> List[str]:
    """Every ``<args>.get('extra_paths', ...)`` expression in a module.

    Matches on the attribute call rather than the receiver's name, so a site
    that renames the tool-call dict (``kwargs``, ``payload``) is still found.
    A read of the plugin's OWN configured list (``self._extra_paths``, or
    ``config['extra_paths']`` in ``initialize``) is an attribute/subscript,
    never a ``.get(...)`` call, so it is not matched and not reported.

    Args:
        path: Absolute path to the module to scan.

    Returns:
        One ``"<file>:<line>"`` string per offending call, for a failure
        message that names where to look.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "get":
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value == MODEL_REFUSED_ARG:
            hits.append(f"{path.name}:{node.lineno}")
    return hits


def test_no_execution_site_reads_extra_paths_from_args() -> None:
    """No cli execution path may take ``extra_paths`` from the call's args.

    Structural rather than behavioural on purpose: the behavioural tests
    below cover the paths that exist today, and this is what stops a path
    added later from reintroducing the shape by copying one of them.  PATH
    extension is operator-configured (``plugin_configs.cli.extra_paths``),
    and the operator's list is read off ``self``, never off the model's
    arguments.
    """
    offenders: List[str] = []
    for rel in _EXECUTION_SITES:
        offenders.extend(_args_get_extra_paths_calls(ROOT / rel))

    assert not offenders, (
        f"{MODEL_REFUSED_ARG!r} is read from the tool-call arguments at "
        f"{offenders}. Extending PATH decides which binary a command name "
        f"resolves to, and the permission decision is keyed on the command "
        f"TEXT alone, so a caller-supplied value puts executable resolution "
        f"outside the value that was approved (#697). Read the operator's "
        f"configured list instead, and refuse the argument with "
        f"shared.cli_path_policy.refuse_model_extra_paths."
    )


# ------------------------------------------------------- behavioural: policy


def test_refuse_model_extra_paths_passes_a_clean_call_through() -> None:
    """The common case costs nothing: no key, no refusal."""
    assert refuse_model_extra_paths({"command": "ls -la"}, surface="t") is None


@pytest.mark.parametrize("value", [["/tmp/evil/bin"], [], None, "/tmp/evil"])
def test_refuse_model_extra_paths_refuses_on_presence_not_truth(
    value: Any,
) -> None:
    """Presence is refused, whatever the value.

    An empty list is inert only because of how the consuming code happens to
    be written today, and "this argument has no effect when falsy" is the
    kind of invariant a later refactor breaks silently.  The rule a reader
    has to hold is the simple one: the key never comes from the caller.
    """
    refusal = refuse_model_extra_paths(
        {"command": "ls", MODEL_REFUSED_ARG: value}, surface="t")

    assert refusal is not None
    assert refusal["error"] == REFUSAL_ERROR
    assert "operator" in refusal["hint"]


# ------------------------------------------------- behavioural: the plugin


def _plugin(tmp_path: Path, **config: Any) -> CLIToolPlugin:
    """A cli plugin rooted at *tmp_path*, with the operator's own config."""
    plugin = CLIToolPlugin()
    plugin.initialize({"workspace_root": str(tmp_path), **config})
    return plugin


def _planted_command(directory: Path) -> str:
    """Write an executable whose name is absent from the base PATH.

    Returns:
        The bare command name, resolvable ONLY through a PATH that includes
        *directory* — the exploit window #697 describes, since ``extra_paths``
        are appended and so can never shadow an existing binary.
    """
    directory.mkdir(parents=True, exist_ok=True)
    exe = directory / "jaato-697-planted"
    exe.write_text("#!/bin/sh\necho PLANTED\n", encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe.name


def test_plugin_execute_refuses_caller_supplied_extra_paths(
    tmp_path: Path,
) -> None:
    """The foreground path refuses, and spawns nothing."""
    planted = tmp_path / "planted"
    command = _planted_command(planted)
    plugin = _plugin(tmp_path)

    result = plugin._execute(
        {"command": command, "extra_paths": [str(planted)]})

    assert result["error"] == REFUSAL_ERROR
    assert "stdout" not in result, "the refusal must precede the spawn"


def test_plugin_execute_streaming_refuses_caller_supplied_extra_paths(
    tmp_path: Path,
) -> None:
    """The streaming path refuses too, through its own callbacks.

    Both loops are live (``_get_effective_output_callback`` decides between
    them per call), so a fix applied to one of them leaves the other open.
    """
    planted = tmp_path / "planted"
    command = _planted_command(planted)
    plugin = _plugin(tmp_path)
    stdout: List[bytes] = []
    stderr: List[bytes] = []
    codes: List[int] = []

    result = plugin._execute_streaming(
        {"command": command, "extra_paths": [str(planted)]},
        on_stdout=stdout.append,
        on_stderr=stderr.append,
        on_returncode=codes.append,
    )

    assert result["error"] == REFUSAL_ERROR
    assert stdout == [], "the refusal must precede the spawn"
    assert codes == [1]
    assert b"extra_paths" in b"".join(stderr)


def test_the_operator_configured_list_still_resolves_a_command(
    tmp_path: Path,
) -> None:
    """The operator's own ``extra_paths`` keeps working, and is the point.

    Without this the fix could be read as "PATH extension was removed",
    which would be a regression rather than a narrowing.  The knob lives in
    ``plugin_configs.cli.extra_paths`` and is unchanged.
    """
    planted = tmp_path / "planted"
    command = _planted_command(planted)
    plugin = _plugin(tmp_path, extra_paths=[str(planted)])

    result = plugin._execute({"command": command})

    assert result.get("returncode") == 0, result
    assert "PLANTED" in result["stdout"]


# ------------------------------------------------- behavioural: the runner


def test_runner_executor_refuses_caller_supplied_extra_paths(
    tmp_path: Path,
) -> None:
    """The Phase-2 runner executor refuses, where nothing else could.

    This surface never had an operator tier: ``extra_paths`` came straight
    out of the model's arguments, so before the fix this same call returned
    ``ok=True`` with the planted binary's output.
    """
    planted = tmp_path / "planted"
    command = _planted_command(planted)

    ok, result = execute_cli_based_tool(
        {"command": command, "extra_paths": [str(planted)]},
        workspace_root=str(tmp_path),
    )

    assert ok is False
    assert result["error"] == REFUSAL_ERROR
    assert "stdout" not in result, "the refusal must precede the spawn"


def test_runner_executor_still_runs_a_clean_command(tmp_path: Path) -> None:
    """A call carrying no ``extra_paths`` is untouched by any of this."""
    ok, result = execute_cli_based_tool(
        {"command": "echo hello"}, workspace_root=str(tmp_path))

    assert ok is True
    assert "hello" in result["stdout"]


# ------------------------------------------------ the property being relied on


def test_the_signature_is_still_blind_to_extra_paths() -> None:
    """Recorded as fact: the signature does NOT cover ``extra_paths``.

    This is the property the whole issue rests on, and it is deliberately
    unchanged — the fix removes the argument's effect rather than widening
    the value the decision is made over, because a refused argument cannot
    vary between approval and execution.

    It is asserted rather than assumed so the reasoning stays visible: if a
    later change DOES put the argument in the signature, this test fails and
    the author is made to re-read why the refusal exists before deciding
    that the signature now carries the weight.
    """
    policy = PermissionPolicy(whitelist_patterns=["deploy *"],
                              default_policy="ask")
    plain: Dict[str, Any] = {"command": "deploy --prod"}
    tainted: Dict[str, Any] = {"command": "deploy --prod",
                               "extra_paths": ["/tmp/evil/bin"]}

    assert (policy._build_signature("cli_based_tool", plain)
            == policy._build_signature("cli_based_tool", tainted))


def test_the_schema_does_not_advertise_extra_paths() -> None:
    """The model-facing schema names ``command`` and ``args`` only.

    A tool schema is the contract a provider's function-calling validator
    enforces, so keeping the key out of it is the cheapest half of the fix —
    the refusal is what covers a model that sends it anyway, since nothing
    here declares ``additionalProperties: false``.
    """
    schema, = CLIToolPlugin().get_tool_schemas()
    properties = schema.parameters["properties"]

    assert MODEL_REFUSED_ARG not in properties
    assert set(properties) == {"command", "args"}
