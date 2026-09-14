#!/usr/bin/env python3
"""Census: which config keys a plugin READS that its schema does not DECLARE.

NOT a guard, and deliberately so.  It is the measurement that has to come
before one, because "every key a plugin reads is declared in its schema" is
not one question in this tree — it is four, and a ratchet seeded before they
are separated would freeze the ambiguity as though it were a fact.

Produced for jaato PR #1040, whose nested-knob descent made the gap matter:
an undeclared name reached by a validator that descends a declared
``properties`` tree is reported, and reporting a key the plugin demonstrably
reads is the one thing this family must never do.  The descent is
evidence-driven for that reason (``plugin_nested_config_read_sites``); this
script is how the size and shape of the gap it compensates for were
measured.

THE FOUR SURFACES, in decreasing confidence that a name belongs to
``plugin_configs.<plugin>``:

1. **Framework-injected.**  ``PluginRegistry._augment_plugin_config``
   puts ``workspace_path`` / ``config_root`` / ``session_id`` /
   ``agent_name`` into EVERY plugin's config dict before ``initialize``.
   ``setdefault`` semantics, so an author MAY override one — which is why
   they are not simply "not knobs".  They are read by ~20 plugins and
   declared by none, and they dominate any raw count.

2. **The block an author writes.**  The plugin's own knobs.  This is the
   surface ``explain plugin <name>`` publishes and the only one a
   completeness claim could sensibly be made about.

3. **A nested dict with its own owner.**  ``permission.channel_config``
   carries whatever the chosen channel accepts (``endpoint`` / ``headers``
   / ``auth_token`` for webhook, ``base_path`` / ``poll_interval`` for
   file).  Those names belong to the channel, not to the plugin, and
   ``additionalProperties`` is the honest marker.

4. **A different file the block points at.**  ``permission.config_path``
   names a permissions JSON whose own keys are ``version`` /
   ``defaultPolicy`` / ``blacklist`` / ``whitelist`` / ``channel``
   (``config_loader.validate_config``: *"Raw configuration dict loaded from
   JSON"*).  ``version`` and ``channel`` are keys of THAT file and are not
   ``plugin_configs.permission`` keys at all.

Only surface 1 is separable mechanically, by reading the registry's own
list.  2 / 3 / 4 need someone to decide, per plugin, which dict a read site
belongs to — so everything else is reported as UNCLASSIFIED, which is the
honest answer rather than a number dressed up as one.

If this ever becomes a ratchet, the thing to count is surface 2 alone, said
so in its docstring.  Surfaces 3 and 4 are then a documented exclusion
rather than an unexamined inclusion.

Usage::

    python scripts/plugin_schema_census.py                 # markdown
    python scripts/plugin_schema_census.py --json          # machine-readable
    python scripts/plugin_schema_census.py -o docs/design/plugin-schema-census.md
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "jaato-server"))
sys.path.insert(0, str(_ROOT / "jaato-sdk"))


def framework_injected_keys() -> List[str]:
    """The keys the registry adds to every plugin config, read from its source.

    AST-scanned rather than spelled, so a key added to
    ``PluginRegistry._augment_plugin_config`` joins this census without an
    edit here — the same discipline the scaffold's own derived tables use.

    Keyed on the ``framework_keys[...]`` assignment rather than on the
    enclosing method NAME, which the first draft of this script got wrong
    (it used the name the method's own docstring mentions).  The failure was
    silent and expensive: an empty result reclassifies every injected key as
    UNCLASSIFIED, turning ~20 framework reads into apparent schema gaps and
    inflating the one number this census exists to report.  The marker is
    distinctive and survives a rename; the name did neither.

    Falls back to an empty list on a parse failure, which is the same
    over-report — so :func:`render` states the count rather than assuming it,
    and a zero here against a non-empty tree is visible in the output.
    """
    src = _ROOT / "jaato-server" / "shared" / "plugins" / "registry.py"
    try:
        tree = ast.parse(src.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    found: List[str] = []
    for sub in ast.walk(tree):
        if (isinstance(sub, ast.Subscript)
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "framework_keys"
                and isinstance(sub.slice, ast.Constant)
                and isinstance(sub.slice.value, str)
                and sub.slice.value not in found):
            found.append(sub.slice.value)
    return sorted(found)


def _declared_names(settings) -> set:
    """Every name a plugin's schema declares, at any depth."""
    out: set = set()
    stack = list(settings)
    while stack:
        s = stack.pop()
        out.add(s.name)
        stack.extend(s.children or [])
    return out


def census() -> Dict[str, Any]:
    """``{framework_keys, plugins: [...], totals}`` over every in-tree plugin."""
    from shared.scaffold import introspect

    injected = framework_injected_keys()
    plugins = introspect.plugins()
    rows: List[Dict[str, Any]] = []
    for name, pi in sorted(plugins.items()):
        sites = introspect.plugin_config_read_sites(name)
        if sites is None:
            continue            # out-of-tree: not scanned, not counted
        declared = _declared_names(pi.config_settings)
        gap = sorted(set(sites) - declared)
        rows.append({
            "plugin": name,
            "declared": len(declared),
            "framework": [k for k in gap if k in injected],
            "unclassified": [k for k in gap if k not in injected],
            "sites": {k: sites[k] for k in gap},
        })
    return {
        "framework_keys": injected,
        "plugins": rows,
        "totals": {
            "plugins_scanned": len(rows),
            "plugins_with_a_gap": sum(1 for r in rows
                                      if r["framework"] or r["unclassified"]),
            "framework_injected": sum(len(r["framework"]) for r in rows),
            "unclassified": sum(len(r["unclassified"]) for r in rows),
        },
    }


def render(data: Dict[str, Any]) -> str:
    """The census as markdown, gap-bearing plugins first."""
    t = data["totals"]
    out = [
        "# Plugin schema census",
        "",
        "Generated by `python scripts/plugin_schema_census.py`. **Not a "
        "guard** — see that script's docstring for the four surfaces this "
        "counts across and why a ratchet needs them separated first.",
        "",
        f"- plugins scanned: **{t['plugins_scanned']}**",
        f"- plugins with at least one undeclared read: "
        f"**{t['plugins_with_a_gap']}**",
        f"- framework-injected reads (surface 1, mechanically separable): "
        f"**{t['framework_injected']}**",
        f"- unclassified reads (surfaces 2/3/4, need a decision): "
        f"**{t['unclassified']}**",
        "",
        "Framework-injected keys, read from "
        "`PluginRegistry._augment_plugin_config`: "
        + (", ".join(f"`{k}`" for k in data["framework_keys"])
           or "**none found — the scan failed, so every injected key below "
              "is counted as unclassified**"),
        "",
        "| plugin | declared | framework-injected | unclassified |",
        "|---|---:|---|---|",
    ]
    for r in sorted(data["plugins"],
                    key=lambda r: (-len(r["unclassified"]), r["plugin"])):
        if not (r["framework"] or r["unclassified"]):
            continue
        fw = ", ".join(f"`{k}`" for k in r["framework"]) or "—"
        un = ", ".join(f"`{k}`" for k in r["unclassified"]) or "—"
        out.append(f"| `{r['plugin']}` | {r['declared']} | {fw} | {un} |")
    clean = [r["plugin"] for r in data["plugins"]
             if not (r["framework"] or r["unclassified"])]
    if clean:
        out += ["", "Declaring everything they read: "
                + ", ".join(f"`{n}`" for n in sorted(clean)) + "."]
    out += _WORKED_EXAMPLE
    return "\n".join(out) + "\n"


#: The one plugin whose gap was resolved read-site by read-site, kept here
#: because the POINT of this census is that the remaining rows have not been.
#: Every other row's "unclassified" column is a question, not a defect.
_WORKED_EXAMPLE = [
    "",
    "## Worked example: `permission`",
    "",
    "The only row audited site by site (jaato PR #1040). It started at 16 "
    "undeclared reads and is the evidence for the four surfaces above.",
    "",
    "| was | surface | resolution |",
    "|---|---|---|",
    "| `agent_name`, `config_path`, `workspace_path`, `channel_type`, "
    "`channel_config` | 2 — the block an author writes | **declared**. Read "
    "straight off `initialize(config)` (`plugin.py:508-544`); an author "
    "configuring a webhook approval channel had no route to the shape but "
    "the source |",
    "| `cwd`, `sanitization.custom_blocked_commands`, "
    "`path_scope.resolve_symlinks` | 2, nested | **declared**. Read by "
    "`PermissionPolicy.from_config` (`policy.py:646-668`) and reported as "
    "typos by the nested descent until they were |",
    "| `endpoint`, `headers`, `auth_token`, `timeout`, `base_path`, "
    "`poll_interval` | 3 — a nested dict with its own owner | "
    "`channel_config` is marked `additionalProperties`. These names belong "
    "to the channel, and nothing in this plugin should judge them |",
    "| `version`, `channel` | 4 — a different file | keys of the permissions "
    "JSON `config_path` names, whose shape "
    "`config_loader.validate_config` owns (*\"Raw configuration dict loaded "
    "from JSON\"*). Not `plugin_configs.permission` keys at all |",
    "| `input_func`, `output_func`, `use_colors`, `skip_readline_history` | "
    "3 — console channel | constructor arguments of the console channel, "
    "reached through `channel_config` |",
    "",
    "Note what the fourth row costs a naive count: two of the sixteen were "
    "never this block's keys, so \"undeclared\" would have been wrong about "
    "them in a direction no amount of declaring could fix.",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="machine-readable")
    ap.add_argument("-o", "--output", help="write to this path instead of stdout")
    args = ap.parse_args()
    data = census()
    text = (json.dumps(data, indent=2, sort_keys=True) + "\n" if args.json
            else render(data))
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
