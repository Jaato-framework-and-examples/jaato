"""An executable named by path gets the verdict its bare name gets (#1202).

``cli``'s path containment classified the EXECUTABLE POSITION as a data
path, so one program got two verdicts::

    git --version            -> allowed  (bare name, nothing path-like)
    /usr/bin/git --version   -> refused  (``/usr/bin/git`` read as data)

and the refusal was disguised as ``git: /usr/bin/git: No such file or
directory``.  A session concluded binaries were vanishing from the host and
filed two misdiagnosed issues (#1202, #1204) plus three duplicates.

The rule: **an executable named by path is allowed iff its directory is an
entry of the PATH its bare name would be resolved against** -- the PATH
``CLIToolPlugin._build_subprocess_env`` builds (``os.environ`` + configured
``extra_paths`` appended + workspace venv bin prepended), computed ONCE and
used both to judge and to run.  It widens nothing: anything in such a
directory already runs by bare name.  What it must not do, and what these
tests pin:

- a binary read as DATA (``cat <bin>/tool``) is still refused -- including
  when the same path is also the executable (``<bin>/tool <bin>/tool``);
- only the operator-configured PATH authorizes.  A PATH the command sets
  for itself (``PATH=...``, ``env PATH=...``, ``export PATH=...``) never
  does;
- ``..`` cannot normalise its way into a PATH directory, because the kernel
  resolves ``..`` through symlinks and ``normpath`` does not;
- a wrapped program (``sudo <bin>/tool``, ``xargs -a <bin>/f cat``) is not
  exempted -- the wrapper model does not know flag arity, so the word after
  a wrapper may be a data file.

And the refusal is honest: it says it is a containment refusal, names the
boundary, and names ``plugin_configs.cli.extra_paths``.

Every directory here is a temp directory this module controls; the host's
own ``/usr/bin`` is never relied on.  ``/tmp`` is normally an allowed region
for ``cli``, and pytest's ``tmp_path`` lives under it, so the temp allowance
is switched off for these tests -- otherwise every "outside" directory would
be allowed as data and nothing here would discriminate.
"""

import os
import sys
from pathlib import Path

import pytest

from jaato_server.shared.plugins import sandbox_utils
from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin
from jaato_server.shared.plugins.command_containment import (
    EXEC_MODE,
    classify_command_paths,
    executable_on_search_path,
)
from jaato_server.shared.tests.reversion import Reversion

_CONTAINMENT = "jaato-server/jaato_server/shared/plugins/command_containment.py"

REVERSIONS = [
    Reversion(
        target=_CONTAINMENT,
        find="""    exe_index = (
        executable_word_index(segment.words) if mark_executables else None
    )""",
        replace="""    exe_index = None""",
        test="test_executable_by_absolute_path_in_a_path_directory_runs",
        because="the executable position is classified as a data read again",
    ),
    Reversion(
        target=_CONTAINMENT,
        find="""            elif _MODE_RANK[mode] > _MODE_RANK[modes[path]]:
                modes[path] = mode""",
        replace="""            elif mode == 'write':
                modes[path] = 'write'""",
        test="test_the_executable_itself_as_an_argument_is_still_data",
        because="a data use of the executable's own path keeps the exec exemption",
    ),
    Reversion(
        target=_CONTAINMENT,
        find="""    if '..' in re.split(r'[\\\\/]', candidate):
        return False
""",
        replace="",
        test="test_dotdot_through_a_symlink_cannot_pose_as_a_path_directory",
        because="a lexically-normalised '..' is trusted though the kernel resolves it differently",
    ),
]

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX executables and PATH semantics"
)

_PREFIX = "cli containment (workspace boundary):"


def _script(path: Path, marker: Path) -> Path:
    """An executable that proves it ran by printing AND touching *marker*."""
    path.write_text(f'#!/bin/sh\necho RAN-{path.name}\ntouch "{marker}"\n')
    path.chmod(0o755)
    return path


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """A workspace, a directory ON PATH, and one that is not.

    ``bin`` is prepended to the process PATH, which is what the plugin's
    environment builder copies.  ``outside`` is on no PATH.  The temp-dir
    allowance is disabled so both read as outside the workspace.
    """
    monkeypatch.setattr(sandbox_utils, "SYSTEM_TEMP_PATHS", [])
    ws = tmp_path / "ws"
    bindir = tmp_path / "bin"
    outside = tmp_path / "outside"
    for d in (ws, bindir, outside):
        d.mkdir()
    marker = tmp_path / "ran"
    _script(bindir / "tool", marker)
    _script(outside / "evil", marker)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    return {"ws": ws, "bin": bindir, "outside": outside, "marker": marker}


def _plugin(layout, **config):
    plugin = CLIToolPlugin()
    plugin.initialize({
        "workspace_root": str(layout["ws"]),
        "scrub_secret_env": "default",
        **config,
    })
    return plugin


def _run(plugin, command, streaming):
    """Drive one of the two execution paths; both must agree."""
    if not streaming:
        return plugin._execute({"command": command})
    return plugin._execute_streaming(
        {"command": command}, lambda _b: None, lambda _b: None, lambda _rc: None,
    )


def _assert_refused_honestly(result, layout, path):
    assert result.get("returncode") == 1, result
    stderr = result["stderr"]
    assert stderr.startswith(_PREFIX), stderr
    assert "No such file or directory" not in stderr
    assert "plugin_configs.cli.extra_paths" in stderr
    assert str(path) in stderr
    assert not layout["marker"].exists(), "the refused command ran"


# --- the rule ---------------------------------------------------------------


@pytest.mark.parametrize("streaming", [False, True], ids=["foreground", "streaming"])
def test_executable_by_absolute_path_in_a_path_directory_runs(layout, streaming):
    """``<bin>/tool`` runs because bare ``tool`` would."""
    plugin = _plugin(layout)

    bare = _run(plugin, "tool", streaming)
    assert bare.get("returncode") == 0, bare
    layout["marker"].unlink()

    by_path = _run(plugin, f"{layout['bin']}/tool", streaming)
    assert by_path.get("returncode") == 0, by_path
    assert "RAN-tool" in by_path["stdout"]
    assert layout["marker"].exists()


@pytest.mark.parametrize("streaming", [False, True], ids=["foreground", "streaming"])
def test_executable_outside_path_is_refused_with_the_honest_message(layout, streaming):
    evil = layout["outside"] / "evil"
    result = _run(_plugin(layout), str(evil), streaming)

    _assert_refused_honestly(result, layout, evil)
    assert "refused to run" in result["stderr"]
    assert "not a missing file" in result["stderr"]


@pytest.mark.parametrize("streaming", [False, True], ids=["foreground", "streaming"])
def test_extra_paths_directory_becomes_runnable_by_path(layout, streaming):
    """The operator's ``extra_paths`` is the supported way in -- and the ONE
    PATH that judges is the one that runs, on both execution paths."""
    plugin = _plugin(layout, extra_paths=[str(layout["outside"])])

    result = _run(plugin, f"{layout['outside']}/evil", streaming)

    assert result.get("returncode") == 0, result
    assert "RAN-evil" in result["stdout"]


# --- data stays data --------------------------------------------------------


def test_the_same_binary_as_data_is_still_refused(layout):
    tool = layout["bin"] / "tool"
    result = _plugin(layout)._execute({"command": f"cat {tool}"})

    _assert_refused_honestly(result, layout, tool)
    assert "refused read access" in result["stderr"]


@pytest.mark.parametrize("template", [
    "head -c 4 {tool}",
    "cp {tool} ./copy",
])
def test_reading_a_path_directory_binary_is_refused(layout, template):
    tool = layout["bin"] / "tool"
    result = _plugin(layout)._execute({"command": template.format(tool=tool)})
    _assert_refused_honestly(result, layout, tool)


def test_the_executable_itself_as_an_argument_is_still_data(layout):
    """``<bin>/tool <bin>/tool``: one occurrence is a program, the other a
    read.  The strongest role wins, so the exemption never covers data."""
    tool = layout["bin"] / "tool"
    result = _plugin(layout)._execute({"command": f"{tool} {tool}"})
    _assert_refused_honestly(result, layout, tool)


# --- only the operator's PATH authorizes -----------------------------------


@pytest.mark.parametrize("template", [
    "PATH={out} {out}/evil",
    "env PATH={out} {out}/evil",
    "export PATH={out}; {out}/evil",
    "PATH={out}:$PATH {out}/evil",
])
def test_a_path_the_command_sets_for_itself_authorizes_nothing(layout, template):
    out = layout["outside"]
    result = _plugin(layout)._execute({"command": template.format(out=out)})
    _assert_refused_honestly(result, layout, out / "evil")


def test_dotdot_through_a_symlink_cannot_pose_as_a_path_directory(layout):
    """``<bin>/link/../evil`` normalises to ``<bin>/evil`` but the kernel
    follows ``link`` first and runs ``<outside>/evil``."""
    sub = layout["outside"] / "sub"
    sub.mkdir()
    (layout["bin"] / "link").symlink_to(sub)
    word = f"{layout['bin']}/link/../evil"

    result = _plugin(layout)._execute({"command": word})

    assert result.get("returncode") == 1, result
    assert result["stderr"].startswith(_PREFIX)
    assert not layout["marker"].exists(), "the out-of-PATH binary ran"


@pytest.mark.parametrize("template", [
    "sudo {tool}",
    "xargs -a {tool} cat",
])
def test_a_wrapped_program_is_not_exempted(layout, template):
    """The wrapper model cannot tell a wrapped program from a flag's file
    argument (``xargs -a FILE``), so nothing after a wrapper is exempted."""
    tool = layout["bin"] / "tool"
    refusal = _plugin(layout)._validate_command_paths(
        template.format(tool=tool), search_path=os.environ["PATH"],
    )
    assert refusal is not None and refusal["stderr"].startswith(_PREFIX)


# --- the predicate and the classifier, directly -----------------------------


@pytest.mark.parametrize("exe, search, allowed", [
    ("/opt/b/tool", "/opt/b", True),
    ("/opt/b/tool", "/opt/b/", True),
    ("/opt//b/tool", "/x:/opt/b", True),
    ("/opt/b/sub/tool", "/opt/b", False),        # not a prefix match
    ("/opt/bin2/tool", "/opt/b", False),
    ("/opt/b/../c/tool", "/opt/c", False),       # '..' disqualifies
    ("/opt/b/tool", None, False),                # no PATH authorizes nothing
    ("/ws/rel/tool", "rel", True),               # relative entry vs cwd
    ("/ws/tool", "/x::/y", True),                # empty entry is cwd
    ("/ws/tool", "/x:/y", False),
    ("/opt/b/", "/opt", False),                  # a directory is no program
])
def test_executable_on_search_path(exe, search, allowed):
    assert executable_on_search_path(exe, search, "/ws") is allowed


def test_classification_is_unchanged_unless_executables_are_asked_for():
    """``interactive_shell`` asks without ``mark_executables`` and must see
    the pre-#1202 classification; ``cli`` asks with it."""
    assert classify_command_paths("/usr/bin/git --version") == [
        ("/usr/bin/git", "read"),
    ]
    assert classify_command_paths(
        "FOO=1 /usr/bin/git --version && cat /usr/bin/ls",
        mark_executables=True,
    ) == [("/usr/bin/git", EXEC_MODE), ("/usr/bin/ls", "read")]
