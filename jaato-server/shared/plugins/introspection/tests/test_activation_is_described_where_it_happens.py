"""The tool that ACTIVATES must be the tool described as activating.

``get_tool_schemas`` is not a reference lookup.  It is the only call that
runs ``session.activate_discovered_tools``, so a discoverable tool is not
callable until its id is passed there.  ``list_tools`` shows what exists
and activates nothing.

That was inverted in the prose for months: the system instructions told
the model to use ``list_tools(category_id=...)`` "to load the real tool
and its id", while ``get_tool_schemas`` was advertised as returning
"full parameter specifications, types, required/optional flags, and
descriptions" -- pure documentation.  A model reading it literally listed
a tool, believed it loaded, found it uncallable, and reached the
capability another way.  Observed 2026-08-29: a session wrote an 8-byte
file through the notebook's in-process ``tools.writeNewFile`` bridge
because it did not know ``get_tool_schemas`` would have made
``writeNewFile`` directly callable.  Its own words afterwards: "I could
have called get_tool_schemas(["writeNewFile"]) and then invoked
writeNewFile directly, without needing the notebook bridge at all."

Prose drifts from code silently, so these pin the two together: whichever
executor calls ``activate_discovered_tools`` is the one whose description
has to say so.
"""

import ast
import inspect
import textwrap

from shared.plugins.introspection.plugin import IntrospectionPlugin


def _schema(name: str):
    for s in IntrospectionPlugin().get_tool_schemas():
        if s.name == name:
            return s
    raise AssertionError(f"{name} is not among the introspection schemas")


def _methods_calling(fn_name: str) -> set:
    """Names of IntrospectionPlugin methods whose body calls ``fn_name``.

    Parsed with ``ast`` rather than matched with a regex: method bodies
    have to be bounded exactly, and a regex that mis-bounds them reports a
    caller in the wrong method -- which is a false accusation, the worst
    failure mode for a guard whose whole job is to say WHERE something
    happens.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(IntrospectionPlugin)))
    cls = tree.body[0]
    found = set()
    for node in cls.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == fn_name):
                found.add(node.name)
    return found


def test_only_get_tool_schemas_activates() -> None:
    """Pin WHERE activation happens, so the prose has a fixed target."""
    activating = _methods_calling("activate_discovered_tools")
    assert activating == {"_execute_get_tool_schemas"}, (
        f"Activation moved: {activating or 'no executor'} calls "
        f"activate_discovered_tools.  The descriptions asserted by the "
        f"other tests in this module name get_tool_schemas as THE "
        f"enabling call -- update them together or the prose inverts "
        f"again."
    )


def test_get_tool_schemas_description_says_it_enables() -> None:
    """The activating tool must advertise the side effect, not just the read."""
    desc = _schema("get_tool_schemas").description.lower()
    assert any(w in desc for w in ("enable", "activat")), (
        f"get_tool_schemas is the ONLY call that makes a discoverable tool "
        f"callable, but its description never says so:\n  {desc!r}\n"
        f"Described as a schema lookup, a model treats discovery as "
        f"read-only and routes around it."
    )


def test_list_tools_does_not_claim_to_load_tools() -> None:
    """``list_tools`` must not borrow the activating verb it does not earn."""
    desc = _schema("list_tools").description.lower()
    for verb in ("enable", "activat", "load the real tool"):
        assert verb not in desc, (
            f"list_tools description claims {verb!r}, but it never calls "
            f"activate_discovered_tools -- only get_tool_schemas does.  "
            f"This exact inversion sent a model looking for another route "
            f"to a tool it had already 'loaded'."
        )


def test_system_instructions_name_the_enabling_call() -> None:
    """The workflow prose must mark step 3 as the step that enables."""
    plugin = IntrospectionPlugin()
    text = plugin.get_system_instructions()
    if not text:
        return  # gated off (no deferred tools on this wire) — nothing to check
    assert "get_tool_schemas" in text
    lowered = text.lower()
    assert "enable" in lowered or "activat" in lowered, (
        "The discovery workflow never tells the model that "
        "get_tool_schemas is what makes a tool callable."
    )
    assert "load the real tool" not in lowered, (
        "The instructions still attach the loading verb to list_tools, "
        "which activates nothing."
    )
