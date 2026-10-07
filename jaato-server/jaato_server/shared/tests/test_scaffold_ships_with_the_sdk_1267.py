"""``jaato-scaffold`` ships with jaato-sdk, and works there without jaato-server (#1267, tier 1).

``jaato-server`` (the daemon) is deployed once; ``jaato-sdk`` is embedded in
many applications, and it is their developers, who install only the SDK, that
need ``jaato-scaffold new`` and ``integration``.  Tier 1 moved the shell, the
authoring verbs and their payloads into ``jaato_sdk.scaffold``; jaato-server
contributes ``explain`` / ``validate`` / ``dependencies`` / ``releases``
through the ``jaato.scaffold_verbs`` entry-point group.  What the move must
keep true, each attached to how it goes wrong:

* **SDK-only runs.**  ``new client``, ``new gitignore`` and ``integration
  claude-code`` succeed with ``jaato_server`` unimportable, and the four
  introspection verbs and ``new dossier`` (the one server-only archetype
  since tier 2) answer with a refusal naming the fix, never an
  ``ImportError``.  Driven in a subprocess whose
  meta path refuses ``jaato_server``, because this process already has it.
* **No module-level server import** anywhere under the SDK location, by AST
  scan: a behavioural run only covers the paths it drives.
* **Reserved names.**  Only code under ``jaato_server`` may answer the four
  introspection verbs, so an installed package cannot replace the validator.
* **One owner** for the console script, and the verbs it reserves are the
  ones jaato-server registers.
* **The payloads ship** in the SDK wheel (package-data covers every file).
* **Provenance.**  The snapshot names the jaato-server it came from, the
  integration stamp names the SDK that ships the payload, and a snapshot read
  beside a different installed jaato-server says so, once.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import subprocess
import sys
import textwrap
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

import jaato_sdk
import jaato_sdk.scaffold as sdk_scaffold
from jaato_sdk.scaffold import authoring_contracts as contracts
from jaato_sdk.scaffold import cli
from jaato_sdk.scaffold import integrations
from jaato_server.shared.tests.reversion import Reversion

_CLI = "jaato-sdk/jaato_sdk/scaffold/cli.py"
_BUILD = "jaato-sdk/jaato_sdk/scaffold/build.py"
_CONTRACTS = "jaato-sdk/jaato_sdk/scaffold/authoring_contracts.py"
_INTEGRATIONS = "jaato-sdk/jaato_sdk/scaffold/integrations.py"
_SERVER_PYPROJECT = "jaato-server/pyproject.toml"

REVERSIONS = [
    Reversion(
        target=_BUILD,
        find="from . import authoring_facts as _facts\n",
        replace=("from . import authoring_facts as _facts\n"
                 "from jaato_server.shared.scaffold import validate as _v\n"),
        because="a module-level jaato_server import makes the SDK's `new` "
                "unimportable in the environment it was moved for",
        test="test_no_sdk_scaffold_module_imports_jaato_server_at_module_level",
    ),
    Reversion(
        target=_CLI,
        find="        else:\n            _add_server_refusal(sub, name)\n",
        replace="        else:\n            pass\n",
        because="without the refusal an SDK-only `validate` is an argparse "
                "`invalid choice`, which names no fix",
        test="test_sdk_only_introspection_verbs_refuse_with_the_fix",
    ),
    Reversion(
        target=_CLI,
        find="    return module == SERVER_PACKAGE or module.startswith(SERVER_PACKAGE + \".\")\n",
        replace="    return True\n",
        because="any installed package could then answer `validate`, replacing "
                "the framework's own validator",
        test="test_a_reserved_verb_from_another_package_is_refused",
    ),
    Reversion(
        target=_BUILD,
        find="    refusal = _refuse_without_server(_server_need(args, archetype))\n",
        replace="    refusal = None\n",
        because="`new dossier` without jaato-server would import the "
                "introspection pages it renders and die with an ImportError, "
                "naming no fix",
        test="test_sdk_only_server_archetypes_refuse_before_writing",
    ),
    Reversion(
        target=_INTEGRATIONS,
        find='PAYLOAD_DIST = "jaato-sdk"\n',
        replace='PAYLOAD_DIST = "jaato-server"\n',
        because="the stamp names a distribution that no longer ships the "
                "payload, so every copy reads stale whenever the two are "
                "released apart",
        test="test_the_integration_stamp_names_the_sdk",
    ),
    Reversion(
        target=_CONTRACTS,
        find="        _SNAPSHOT_CACHE = data\n        _warn_on_version_skew(data)\n",
        replace="        _SNAPSHOT_CACHE = data\n",
        because="an SDK wheel's snapshot read beside a different jaato-server "
                "would answer for the wrong server, silently",
        test="test_a_snapshot_beside_another_server_version_warns_once",
    ),
    Reversion(
        target=_SERVER_PYPROJECT,
        find='jaato-server = "jaato_server.server.__main__:main"\n',
        replace=('jaato-server = "jaato_server.server.__main__:main"\n'
                 'jaato-scaffold = "jaato_server.shared.scaffold.__main__:main"\n'),
        because="two distributions declaring one console script fight over "
                "one file in bin/, and the last installed wins",
        test="test_the_console_script_has_one_owner",
    ),
]


# ------------------------------------------------------------------ helpers

_BLOCKER = textwrap.dedent('''
    import sys
    import importlib.metadata as _md

    class _NoServer:
        def find_spec(self, name, path=None, target=None):
            if name == "jaato_server" or name.startswith("jaato_server."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    sys.meta_path.insert(0, _NoServer())
    _eps = _md.entry_points
    _md.entry_points = lambda **kw: [e for e in _eps(**kw)
                                     if not e.value.startswith("jaato_server")]
    from jaato_sdk.scaffold.cli import main
    rc = main(sys.argv[1:])
    leaked = sorted(m for m in sys.modules if m.startswith("jaato_server"))
    print("RESULT", rc, leaked, file=sys.stderr)
''')


def _sdk_only(*argv: str, cwd: Path):
    """Run ``jaato-scaffold argv`` where ``jaato_server`` cannot be imported.

    The blocker also drops jaato-server's entry points, since its metadata is
    installed here even though the package is refused.  Returns
    ``(rc, stdout, stderr, leaked_modules)``.
    """
    proc = subprocess.run([sys.executable, "-c", _BLOCKER, *argv],
                          capture_output=True, text=True, cwd=str(cwd))
    line = [ln for ln in proc.stderr.splitlines() if ln.startswith("RESULT ")]
    assert line, f"the shell crashed:\n{proc.stdout}\n{proc.stderr}"
    _, rc, leaked = line[-1].split(" ", 2)
    return int(rc), proc.stdout, proc.stderr, leaked


def _sdk_root() -> Path:
    return Path(sdk_scaffold.__file__).resolve().parent


def _pyproject(dist: str) -> dict:
    # Located from the installed packages' own files so the reversion
    # meta-guard's sandboxed copy is the one read.
    if dist == "jaato-sdk":
        root = Path(jaato_sdk.__file__).resolve().parents[1]
    else:
        import jaato_server
        root = Path(jaato_server.__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


# --------------------------------------------------------------- SDK-only


def test_sdk_only_authoring_commands_run(tmp_path):
    ws = tmp_path / "ws"
    rc, out, err, leaked = _sdk_only(
        "new", "client", "--workspace", str(ws), "--provider", "anthropic",
        "--model", "m", cwd=tmp_path)
    assert rc == 0, out + err
    assert (ws / "run_client.py").is_file() and (ws / ".env").is_file()
    assert leaked == "[]"

    rc, out, err, leaked = _sdk_only("new", "gitignore", "--workspace", str(ws),
                                     cwd=tmp_path)
    assert rc == 0, out + err
    assert ".jaato/*" in (ws / ".gitignore").read_text()
    assert leaked == "[]"

    rc, out, err, leaked = _sdk_only("integration", "claude-code",
                                     "--workspace", str(ws), cwd=tmp_path)
    assert rc == 0, out + err
    assert (ws / ".claude" / "skills" / "jaato-sdk" / "SKILL.md").is_file()
    assert leaked == "[]"


#: What each verb is asked here.  Bare ``explain`` is answered from the
#: snapshot shipped with jaato-sdk (test_explain_snapshot_mirrors_the_server),
#: so it is asked a topic the snapshot cannot answer, which still refuses.
_REFUSED_ARGV = {"explain": ("explain", "releases")}


@pytest.mark.parametrize("verb", cli.SERVER_VERBS)
def test_sdk_only_introspection_verbs_refuse_with_the_fix(tmp_path, verb):
    rc, out, err, leaked = _sdk_only(*_REFUSED_ARGV.get(verb, (verb,)),
                                     cwd=tmp_path)
    assert rc == 2
    assert "pip install jaato-server" in err, err
    assert "Traceback" not in err and "ImportError" not in err
    assert leaked == "[]"


@pytest.mark.parametrize("argv", [
    ("new", "dossier", "--component"),
    ("new", "dossier", "--profile", "worker"),
])
def test_sdk_only_server_archetypes_refuse_before_writing(tmp_path, argv):
    """Only ``new dossier`` is refused since tier 2; the rest emit (see
    ``test_scaffold_tier2_snapshot_1267.py``)."""
    ws = tmp_path / "ws"
    rc, out, err, leaked = _sdk_only(*argv, "--workspace", str(ws), cwd=tmp_path)
    assert rc == 2, out + err
    assert "needs jaato-server" in out and "pip install jaato-server" in out
    assert not ws.exists() or not any(ws.rglob("*")), "wrote before refusing"
    assert leaked == "[]"


def test_a_sweep_without_its_gate_needs_no_server(tmp_path):
    ws = tmp_path / "ws"
    rc, out, err, _ = _sdk_only("new", "sweep", "--no-gate", "--workspace",
                                str(ws), "--provider", "anthropic", "--model",
                                "m", cwd=tmp_path)
    assert rc == 0, out + err


# ------------------------------------------------------------ import scan


def _module_level_imports(tree: ast.Module):
    """Import nodes executed at import time: module body, if/try/with, class bodies."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        else:
            stack.extend(ast.iter_child_nodes(node))


def test_no_sdk_scaffold_module_imports_jaato_server_at_module_level():
    files = sorted(_sdk_root().rglob("*.py"))
    files.append(Path(jaato_sdk.__file__).resolve().parent / "gitignore_parser.py")
    assert len(files) >= 12, files
    offenders = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _module_level_imports(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            if any(n == "jaato_server" or n.startswith("jaato_server.")
                   for n in names):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        f"module-level jaato_server imports in the SDK: {offenders}.  Reach "
        "jaato-server inside the function that needs it, guarded, as "
        "jaato_sdk/doctor.py does (#1267).")


# ---------------------------------------------------------- reserved names


def test_a_reserved_verb_from_another_package_is_refused(monkeypatch, capsys):
    class Impostor:
        name = "validate"
        help = "not the framework's validator"

        def configure(self, parser):
            parser.add_argument("rest", nargs="*")

        def run(self, args):
            return 99

    class EP:
        name = "validate"
        value = "someone_else.verbs:Impostor"

        def load(self):
            return Impostor

    monkeypatch.setattr("importlib.metadata.entry_points",
                        lambda *a, **k: [EP()])
    rc = cli.main(["validate", "."])
    assert rc == 2, "an impostor answered a name reserved for jaato-server"
    assert "pip install jaato-server" in capsys.readouterr().err


def test_a_server_verb_handed_in_directly_is_mounted(monkeypatch, capsys):
    """``python -m jaato_server.shared.scaffold`` works without entry points."""
    from jaato_server.shared.scaffold import introspection_verbs as iv

    monkeypatch.setattr("importlib.metadata.entry_points", lambda *a, **k: [])
    assert iv.main(["explain", "paths"]) == 0
    assert "paths" in capsys.readouterr().out


# ------------------------------------------------------------- packaging


def test_the_console_script_has_one_owner():
    sdk = _pyproject("jaato-sdk")["project"]
    server = _pyproject("jaato-server")["project"]
    assert sdk["scripts"].get("jaato-scaffold") == "jaato_sdk.scaffold.cli:main"
    assert "jaato-scaffold" not in server.get("scripts", {}), (
        "jaato-scaffold is declared by jaato-sdk only (#1267)")
    verbs = server["entry-points"]["jaato.scaffold_verbs"]
    assert set(verbs) == set(cli.SERVER_VERBS)
    for name, target in verbs.items():
        assert target.startswith("jaato_server.shared.scaffold.introspection_verbs:"), target


def test_the_reserved_names_are_the_ones_the_server_implements():
    from jaato_server.shared.scaffold import introspection_verbs as iv

    assert [v.name for v in iv.server_verbs()] == list(cli.SERVER_VERBS)
    assert not set(cli.SERVER_VERBS) & set(cli.BUILTIN_VERBS)


def test_every_payload_file_is_package_data():
    patterns = _pyproject("jaato-sdk")["tool"]["setuptools"]["package-data"][
        "jaato_sdk.scaffold"]
    root = _sdk_root()
    shipped = [p.relative_to(root).as_posix() for p in root.rglob("*")
               if p.is_file() and p.suffix != ".py" and "__pycache__" not in p.parts]
    assert "authoring_snapshot.json" in shipped
    missing = [f for f in shipped
               if not any(fnmatch.fnmatch(f, pat) for pat in patterns)]
    assert not missing, f"not in jaato-sdk's package-data, so absent from the wheel: {missing}"


# ------------------------------------------------------------- provenance


def test_the_integration_stamp_names_the_sdk():
    assert integrations.PAYLOAD_DIST == "jaato-sdk"
    assert integrations.framework_version() == version("jaato-sdk")


def test_the_snapshot_records_the_server_it_came_from():
    data = json.loads(contracts.SNAPSHOT_FILE.read_text(encoding="utf-8"))
    assert data["jaato_server_version"] == _pyproject("jaato-server")["project"]["version"]


def test_a_snapshot_beside_another_server_version_warns_once(monkeypatch, capsys):
    monkeypatch.setattr(contracts, "_FORCE_SNAPSHOT", True)
    monkeypatch.setattr(contracts, "_SNAPSHOT_CACHE", None)
    monkeypatch.setattr(contracts, "_SKEW_WARNED", False)
    monkeypatch.setattr(contracts, "installed_server_version", lambda: "0.0.1")
    recorded = json.loads(contracts.SNAPSHOT_FILE.read_text())["jaato_server_version"]

    contracts.providers()
    err = capsys.readouterr().err
    assert "0.0.1" in err and recorded in err, err

    monkeypatch.setattr(contracts, "_SNAPSHOT_CACHE", None)
    contracts.env_vars()
    assert capsys.readouterr().err == "", "warned twice"


def test_a_snapshot_beside_the_same_server_version_is_quiet(monkeypatch, capsys):
    recorded = json.loads(contracts.SNAPSHOT_FILE.read_text())["jaato_server_version"]
    monkeypatch.setattr(contracts, "_FORCE_SNAPSHOT", True)
    monkeypatch.setattr(contracts, "_SNAPSHOT_CACHE", None)
    monkeypatch.setattr(contracts, "_SKEW_WARNED", False)
    for installed in (recorded, None):
        monkeypatch.setattr(contracts, "installed_server_version",
                            lambda v=installed: v)
        monkeypatch.setattr(contracts, "_SNAPSHOT_CACHE", None)
        contracts.providers()
    assert capsys.readouterr().err == ""
