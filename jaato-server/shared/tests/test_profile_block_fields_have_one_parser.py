"""A profile's optional block fields are parsed in ONE place.

WHY THIS EXISTS.  Four separate functions build a ``SubagentProfile``
from a dict -- ``build_inline_profile`` (inline spec),
``_scan_profiles_dir`` (workspace + user profile dirs),
``_discover_premium_profiles``, and ``SubagentConfig.from_dict`` -- and
each carried its own copy of the ``gc:`` parse, in two spellings of the
identical guard.

Four copies is four places to forget.  A block field wired into three
ingresses and missed in the fourth is silently inert in exactly one code
path, and which path that is depends on how the profile was loaded --
so it presents as "the knob works for me and not for you".

That is not hypothetical for this area: §4 of
``docs/design/model-tier-prompt-cache.md`` documents a cache knob that
reached NO ingress, was documented as working, and was silently ignored
for months.  The next block field (the proposed ``cache:``, §7) should be
addable once rather than four times.
"""

import ast
import pathlib

from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

#: The defect, put back: one ingress parses ``gc`` inline again instead of
#: calling the shared parser.
REVERSIONS = [
    Reversion(
        target="jaato-server/shared/plugins/subagent/config.py",
        find="    gc_config = parse_gc_block(data)\n",
        replace=("    gc_config = None\n"
                 "    if data.get('gc'):\n"
                 "        gc_config = GCProfileConfig.from_dict(data['gc'])\n"),
        test="test_only_the_shared_parser_constructs_a_gc_config",
        because="a second copy of a parse that must not drift between ingresses",
    ),
]

ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIG = ROOT / "jaato-server" / "shared" / "plugins" / "subagent" / "config.py"


def _tree():
    return ast.parse(CONFIG.read_text(encoding="utf-8"))


def _enclosing_function_names(pred):
    """Names of functions containing a node matching ``pred``."""
    out = set()
    for node in ast.walk(_tree()):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if child is not node and pred(child):
                out.add(node.name)
    return out


def test_only_the_shared_parser_constructs_a_gc_config():
    """``GCProfileConfig.from_dict`` is called from exactly one function.

    Checked as an AST rather than by counting a string, so a call written
    across two lines still counts and a mention in a docstring does not.
    """
    def is_from_dict_call(n):
        return (isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "from_dict"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "GCProfileConfig")

    callers = _enclosing_function_names(is_from_dict_call)
    assert callers == {"parse_gc_block"}, (
        f"GCProfileConfig.from_dict is constructed in {sorted(callers)}; it "
        f"belongs only in parse_gc_block, or the copies drift and a new "
        f"block field gets wired into some ingresses and not others"
    )


def test_every_ingress_goes_through_the_shared_parser():
    """All four profile-building functions call it.

    The complement of the test above: one asserts there is no second
    parser, this asserts nobody quietly stopped parsing the block at all.
    A dropped call is just as silent as a duplicated one.
    """
    def is_parse_call(n):
        return (isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "parse_gc_block")

    callers = _enclosing_function_names(is_parse_call)
    expected = {
        "build_inline_profile",       # inline {"model": ...} spec
        "_scan_profiles_dir",         # workspace + user .jaato/profiles
        "_discover_premium_profiles",  # jaato-premium's bundled set
        "from_dict",                  # SubagentConfig.from_dict
    }
    missing = expected - callers
    assert not missing, (
        f"these profile ingresses no longer parse the gc block: "
        f"{sorted(missing)} — a profile loaded that way silently loses it"
    )
