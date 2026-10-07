"""The overview banner advertises what `explain` dispatches (#1006).

``jaato-scaffold explain`` with no scope prints a "drill down:" banner.  It was
the THIRD hand-typed topic list, after the ``one of:`` error and argparse's
``--help`` that #994 derived -- and it had drifted the same way: 21 topics
advertised against 23 in ``_SCOPES``, with ``env``, ``event`` and ``events``
dispatching and named nowhere.  An introspection tool under-reporting what it
offers is the one failure it must not have.

THE TRAP THIS GUARD EXISTS FOR -- and it is NOT #994's.  ``dependencies``
appears in the banner and is not a topic, so a derivation from ``_SCOPES``
alone either drops it or, worse, reintroduces it as a phantom topic the
round-trip check would then have to be weakened to tolerate.  It is a FACET:
``_take_deps_word`` strips the word from any position and dispatches to
``dependencies.render`` BEFORE ``_SCOPES`` is consulted.  So the banner renders
topics and facets from two different sources, and this guard checks both --
including the half the old prose got wrong, which was not the omissions at all:

    "append `dependencies` (or `deps`) to ANY of the above -- it is a facet of
     every scope"

Only a NAMED provider or plugin has a facet of its own.  Every other scope --
and a bare ``explain dependencies``, and ``provider`` with no name -- falls
through to the framework picture, whose own note says *"'<scope>' has no
dependency facet of its own"*.  The banner contradicted the code it described.
So the facet forms are checked BEHAVIOURALLY (call ``render``, look at what
comes back) rather than against a second list, because a list is what went
stale.
"""

import re

from jaato_server.shared.scaffold import dependencies as _deps
from jaato_server.shared.scaffold import explain as _explain
from jaato_server.shared.scaffold.__main__ import _DEPS_WORDS, _SCOPES, scope_catalog
from jaato_server.shared.tests.reversion import Reversion

_EXPLAIN = "jaato-server/jaato_server/shared/scaffold/explain.py"

_HAND_TYPED_BANNER = '''        "drill down:\\n"
        "  jaato-scaffold explain plugins\\n"
        "  jaato-scaffold explain plugin <name>\\n"
        "  jaato-scaffold explain commands\\n"
        "  jaato-scaffold explain providers\\n"
        "  jaato-scaffold explain provider <name>\\n"
        "  jaato-scaffold explain gc\\n"
        "\\n"
'''

_EVERY_SCOPE_IS_A_FACET = '''        "`dependencies` is a facet of every scope\\n"
'''


REVERSIONS = [
    Reversion(
        target=_EXPLAIN,
        find='        + "\\n".join(_topic_lines()) + "\\n"',
        replace=_HAND_TYPED_BANNER,
        test="test_every_dispatched_topic_is_advertised_in_the_banner",
        because="the banner is hand-typed prose again, so topics that "
                "dispatch (env, event, events, ...) are advertised nowhere",
    ),
    Reversion(
        target=_EXPLAIN,
        find='        + "\\n".join(_facet_lines()) + "\\n"',
        replace=_EVERY_SCOPE_IS_A_FACET,
        test="test_the_banner_claims_a_unit_facet_only_where_one_exists",
        because="the banner claims every scope has a dependency facet, while "
                "only a named provider or plugin does",
    ),
]


def _banner() -> str:
    _data, text = _explain.overview()
    return text


def _advertised_topics(banner: str) -> list:
    """The topic names the banner's `explain <topic>` lines advertise.

    Reads the rendered text rather than the catalog it was built from: a
    derivation asserted against its own source proves only that the source
    exists.

    Scoped to the DRILL-DOWN BLOCK -- the lines between ``drill down:`` and the
    blank line that ends it -- because that block is what "the topic list"
    means.  The facet block below it also spells ``jaato-scaffold explain
    ...`` lines (``explain dependencies``, ``explain provider <name> deps``),
    and those are facet FORMS, not topics.

    Two wrong ways to draw that line, both measured rather than reasoned
    about.  Skipping every line that mentions a facet word swallows a phantom
    ``dependencies`` injected into the topic list, so the sabotage that adds
    one passed.  Skipping only when the word TRAILS a topic then counts the
    facet block's own ``explain dependencies`` as a topic, so the guard failed
    on clean source.  The section boundary is the distinction that was
    actually meant.
    """
    lines = banner.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("drill down:"))
    out = []
    for line in lines[start + 1:]:
        if not line.strip():
            break
        m = re.match(r"\s*jaato-scaffold explain (\S+)", line)
        if m:
            out.append(m.group(1))
    return out


def test_every_dispatched_topic_is_advertised_in_the_banner():
    """The round trip, both directions -- #994's check, on the third list."""
    advertised = _advertised_topics(_banner())

    missing = sorted(set(_SCOPES) - set(advertised))
    assert not missing, (
        f"dispatched but absent from the overview banner: {missing}. "
        f"`explain {missing[0]}` works and the banner does not mention it -- "
        f"#1006 exactly."
    )

    phantom = sorted(set(advertised) - set(_SCOPES))
    assert not phantom, (
        f"the banner advertises topics `explain` answers with 'unknown "
        f"explain scope': {phantom}."
    )

    assert len(advertised) == len(set(advertised)), (
        f"the banner lists a topic twice: {advertised}"
    )


def test_the_banner_carries_each_topics_argument_hint():
    """A hint lives on the table entry, so the banner cannot spell it wrong."""
    banner = _banner()
    for topic in scope_catalog():
        expected = f"explain {topic['scope']} {topic['arg']}".rstrip()
        assert expected in banner, (
            f"the banner does not carry {topic['scope']}'s argument hint "
            f"{topic['arg']!r}"
        )


def test_the_banner_marks_the_topics_that_read_the_workspace():
    """``[--workspace DIR]`` on exactly the topics handed that value.

    The hand-typed banner put the hint on ``sets`` alone while ``agents``,
    ``services`` and ``profile`` read the workspace just as much -- the same
    drift as the missing topics, in a field rather than a row.
    """
    for line in _banner().splitlines():
        m = re.match(r"\s*jaato-scaffold explain (\S+)", line)
        if not m or m.group(1) not in _SCOPES:
            continue
        spec = _SCOPES[m.group(1)]
        reads = spec.kind in ("workspace", "optional_named")
        assert ("[--workspace DIR]" in line) == reads, (
            f"{m.group(1)}: banner says workspace="
            f"{'[--workspace DIR]' in line}, dispatch says {reads}"
        )


def test_the_facet_word_is_not_advertised_as_a_topic():
    """`dependencies` is stripped before the topic lookup, so it is no topic.

    The naive derivation's other failure mode: re-listing the facet among the
    topics would make ``explain <nonsense>``'s topic list and this banner
    disagree again, in the opposite direction.
    """
    for word in _DEPS_WORDS:
        assert word not in _SCOPES, (
            f"{word!r} is registered as a topic; it is consumed by "
            f"_take_deps_word before _SCOPES is consulted, so a topic of that "
            f"name could never be reached."
        )
        assert word not in _advertised_topics(_banner()), (
            f"the banner advertises {word!r} as a topic in the drill-down list"
        )


def test_the_banner_claims_a_unit_facet_only_where_one_exists():
    """The claim the old prose got wrong, checked against `render` itself.

    Every unit form the banner advertises must actually produce a facet about
    that unit, and every scope it does NOT advertise must fall through to the
    framework picture.  Behavioural on both sides: the defect was a sentence
    that no longer matched the routing.
    """
    banner = _banner()
    probes = {"provider": "openrouter", "plugin": "cli"}

    for unit, (aliases, _build) in _deps.UNIT_FACETS.items():
        assert f"explain {unit} <name> " in banner, (
            f"{unit!r} routes to a unit dependency facet and the banner does "
            f"not advertise `explain {unit} <name> <deps-word>`"
        )
        data, _text = _deps.render(unit, probes[unit])
        assert data.get("kind") == unit and not data.get("note"), (
            f"the banner advertises a {unit} facet and render() answered with "
            f"{data.get('kind')!r}"
        )
        assert aliases, f"{unit!r} declares no accepted scope words"

    advertised_units = set(_deps.UNIT_FACETS)
    for scope in _SCOPES:
        if scope in {a for al, _ in _deps.UNIT_FACETS.values() for a in al}:
            continue
        data, _text = _deps.render(scope, "anything")
        assert data.get("note"), (
            f"{scope!r} is not an advertised unit facet, so render() must "
            f"answer with the framework picture and SAY so; it returned "
            f"{data.get('kind')!r} with no note."
        )
    assert advertised_units, "no unit facets declared at all"


def test_a_unit_scope_without_a_name_is_not_a_unit_facet():
    """`explain provider deps` names no unit, so it cannot report one.

    This is why the banner's example forms carry ``<name>``: the facet needs
    something to be about, and without it the answer is the framework's.
    """
    for unit in _deps.UNIT_FACETS:
        data, _text = _deps.render(unit, None)
        assert data.get("note"), (
            f"render({unit!r}, None) returned a unit facet with no unit named"
        )


def test_the_dependency_facet_is_reachable_from_any_position():
    """The banner says "any position after `explain`"; the parser must agree."""
    from jaato_server.shared.scaffold.__main__ import _take_deps_word
    for query in [("dependencies", None, None),
                  ("provider", "openrouter", "deps"),
                  ("deps", "provider", "openrouter")]:
        _scope, _name, asked = _take_deps_word(*query)
        assert asked, f"the deps word was not recognised in {query}"
