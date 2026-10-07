"""A profile's own top-level keys are checked, and the accepted set cannot drift.

``SubagentProfile`` construction is keyword-explicit — ``config.py`` says so
where the snapshot version is declared — so a key the loader does not read is
never read by anything, silently.  That is the outermost layer of the
silent-ignore family (#910 / #925 / #947 / #950) and the one none of them
closed: ``unknown_knob`` covers ``plugin_configs.<plugin>.<knob>``, ``trace:``
refuses unknown keys of its own, ``runtime_limits`` parks them in ``extra`` —
and the profile's own outermost layer had no reporter at all.

Two halves here, and the second is what keeps the first honest: the findings
behave, and :data:`PROFILE_FILE_KEYS` still describes the loader.  A constant
that drifts from the code it describes turns this check into a generator of
false reports about working profiles, which is worse than the silence it
replaced.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
from pathlib import Path

from jaato_server.shared.plugins.subagent import config as cfg
from jaato_server.shared.scaffold import validate


#: Every function that reads keys out of a raw profile FILE dict.  The five
#: block parsers are included because they are where ``cache`` / ``gc`` /
#: ``trace`` / ``regulatory`` / ``env`` are read — a scan of the builder alone would miss them
#: and report four working keys as unknown.
_LOADER_FUNCS = (
    "_parse_profile_file",
    "_scan_profiles_dir",
    "parse_cache_block",
    "parse_gc_block",
    "parse_trace_block",
    "parse_regulatory_block",
    "parse_profile_env",
)


def _is_data(node) -> bool:
    return isinstance(node, ast.Name) and node.id == "data"


def _const_str(node):
    """The node's value if it is a string constant, else ``None``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _get_key(node):
    """``data.get("x")`` → ``"x"``."""
    if not (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _is_data(node.func.value) and node.args):
        return None
    return _const_str(node.args[0])


def _subscript_key(node):
    """``data["x"]`` → ``"x"``."""
    if not (isinstance(node, ast.Subscript) and _is_data(node.value)):
        return None
    return _const_str(node.slice)


def _contains_key(node):
    """``"x" in data`` → ``"x"``."""
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], ast.In)
            and _is_data(node.comparators[0])):
        return None
    return _const_str(node.left)


_KEY_SHAPES = (_get_key, _subscript_key, _contains_key)


def _literal_keys(func_name: str) -> set:
    """Every constant string this function reads out of its ``data`` mapping.

    Recognises the three spellings the loader uses: ``data.get("x")``,
    ``data["x"]`` and ``"x" in data``.  A key read through a variable (
    ``parse_profile_env``'s ``key`` parameter) is invisible to this and is
    covered by the ghost test below instead.
    """
    src = inspect.getsource(getattr(cfg, func_name))
    tree = ast.parse(textwrap.dedent(src))
    found: set = set()
    for node in ast.walk(tree):
        for shape in _KEY_SHAPES:
            key = shape(node)
            if key is not None:
                found.add(key)
    return found


def test_every_key_the_loader_reads_is_declared():
    """The drift direction that matters: a new key must not read as unknown."""
    read: set = set()
    for fn in _LOADER_FUNCS:
        read |= _literal_keys(fn)
    assert read, "the AST scan found nothing — it has stopped seeing the loader"
    undeclared = read - set(cfg.PROFILE_FILE_KEYS)
    assert not undeclared, (
        f"the loader reads {sorted(undeclared)}, which PROFILE_FILE_KEYS does "
        f"not list — `validate` would report a working profile key as unknown")


def test_no_declared_key_is_a_ghost():
    """The other direction: a stale entry silently exempts a key from checking.

    Weaker than its sibling by construction — a key may legitimately be read
    through a variable — so this asserts only that the name appears in the
    loader module at all, which a deleted key would not.
    """
    src = Path(cfg.__file__).read_text(encoding="utf-8")
    ghosts = [k for k in cfg.PROFILE_FILE_KEYS
              if f"'{k}'" not in src and f'"{k}"' not in src]
    assert not ghosts, f"PROFILE_FILE_KEYS names keys config.py never mentions: {ghosts}"


def test_derived_fields_are_fields_and_not_file_keys():
    names = {f.name for f in dataclasses.fields(cfg.SubagentProfile)}
    for key in cfg.PROFILE_DERIVED_FIELDS:
        assert key in names, f"{key} is not a SubagentProfile field"
        assert key not in cfg.PROFILE_FILE_KEYS, (
            f"{key} is declared both derived and file-readable")


# ------------------------------------------------------------------ findings

def _codes(data, name="p"):
    return [(d.code, d.where) for d in validate._profile_key_findings(data, name)]


def test_a_key_nobody_reads_is_reported():
    # The real one: `config_root` is a session/SDK parameter, and a profile
    # carrying it looks like it configured something.
    assert ("unknown_profile_key", "config_root") in _codes(
        {"plugins": [], "config_root": ".jaato"})


def test_a_near_miss_names_the_key_it_meant():
    findings = validate._profile_key_findings(
        {"plugins": [], "plugins_configs": {}}, "p")
    assert "did you mean 'plugin_configs'" in findings[0].message


def test_a_derived_field_gets_its_own_finding():
    # `tool_scopes` IS a SubagentProfile field, so `explain profile` lists it
    # — and writing it in a file does nothing.
    assert ("derived_profile_key", "tool_scopes") in _codes(
        {"plugins": [], "tool_scopes": {"cli": ["cli_based_tool"]}})


def test_every_accepted_key_is_silent():
    data = {k: None for k in cfg.PROFILE_FILE_KEYS}
    assert _codes(data) == []


def test_findings_are_warnings():
    # An inherits base may carry a key a later version reads, and a snapshot
    # may be newer than this installation; failing those outright is the
    # worse trade.
    out = validate._profile_key_findings({"plugins": [], "nope": 1}, "p")
    assert [d.severity for d in out] == ["warn"]


def test_a_non_mapping_is_not_a_crash():
    assert validate._profile_key_findings(None, "p") == []
    assert validate._profile_key_findings([1, 2], "p") == []


# ------------------------------------------------------ the workspace walk

def _write(root: Path, rel: str, body: str):
    p = root / ".jaato" / "profiles" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body), encoding="utf-8")


def test_the_walk_reads_files_the_resolver_never_surfaces(tmp_path):
    # A profile inside an unselected set is not in the effective set, so the
    # per-profile checks never see it — and its keys are still the author's
    # to get wrong.
    _write(tmp_path, "a_set/agent.yaml", """
        name: agent
        description: d
        plugins: []
        config_root: .jaato
    """)
    out: list = []
    validate._check_profile_file_keys(str(tmp_path / ".jaato"), out)
    assert [(d.code, d.where) for d in out] == [
        ("unknown_profile_key", "config_root")]


def test_a_file_the_loader_refuses_outright_is_left_alone(tmp_path):
    # No `plugins:` — the loader refuses the file by name, so anything else
    # said about it is noise on top of a reported error.
    _write(tmp_path, "broken.yaml", """
        name: broken
        description: d
        nonsense: 1
    """)
    out: list = []
    validate._check_profile_file_keys(str(tmp_path / ".jaato"), out)
    assert out == []


def test_a_non_profile_yaml_is_not_invented_into_findings(tmp_path):
    _write(tmp_path, "notes.yaml", "just: some data\nkeys: here\n")
    out: list = []
    validate._check_profile_file_keys(str(tmp_path / ".jaato"), out)
    assert out == []


def test_a_missing_profiles_dir_is_not_a_crash(tmp_path):
    out: list = []
    validate._check_profile_file_keys(str(tmp_path / ".jaato"), out)
    assert out == []
