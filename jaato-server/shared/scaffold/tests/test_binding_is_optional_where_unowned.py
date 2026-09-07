"""``--provider`` / ``--model`` are required only where the archetype OWNS
the binding.

Provider and model name what CREATES a session, and a profile is what carries
them.  Two archetypes were required to supply them anyway and neither owns the
binding: ``observer`` never creates a session or sends a message at all — the
flags produced two dead constants in a script with no provider relationship —
and ``cascade`` / ``sweep`` stages point at profiles, so the flags shaped only
a placeholder the template tells the reader to replace, plus a misleading
``.env`` default (jaato #820).

The flags remain ACCEPTED for all three: supplying them keeps the inline-spec
placeholder, which is what makes a generated cascade runnable before any
profile exists.  What changed is that omitting them is now the profile-driven
shape rather than an error.
"""

from __future__ import annotations

import argparse
import ast
import py_compile

import pytest

from shared.scaffold import archetypes, build, introspect
from shared.scaffold._client_templates import PROVIDER_OPTIONAL, TEMPLATES

_OWNS_THE_BINDING = tuple(sorted(set(TEMPLATES) - set(PROVIDER_OPTIONAL)))


def _args(**kw):
    base = dict(archetype=None, workspace=None, provider=None, model=None,
                set=None, agents=None, force=True, recoverable=False,
                json=False, transport=None, url=None, token=None, ca=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _a_provider() -> str:
    names = sorted(introspect.providers())
    assert names, "no providers installed — cannot scaffold a client"
    return names[0]


def test_the_two_lists_partition_the_archetypes():
    """Guard the guard: every archetype is in exactly one camp, so neither
    parametrised list below can quietly go empty."""
    assert set(PROVIDER_OPTIONAL) | set(_OWNS_THE_BINDING) == set(TEMPLATES)
    assert PROVIDER_OPTIONAL and _OWNS_THE_BINDING


@pytest.mark.parametrize("arch", PROVIDER_OPTIONAL)
def test_it_generates_without_a_binding(tmp_path, arch):
    ws = tmp_path / arch
    assert build.run(_args(archetype=arch, workspace=str(ws))) == 0
    py_compile.compile(str(ws / f"run_{arch}.py"), doraise=True)


@pytest.mark.parametrize("arch", _OWNS_THE_BINDING)
def test_an_archetype_that_owns_the_binding_still_requires_it(tmp_path, arch):
    """``client`` / ``fire`` / ``host-tools`` create a session from an inline
    spec and may have no profile at all, so the flags stay mandatory."""
    ws = tmp_path / arch
    assert build.run(_args(archetype=arch, workspace=str(ws))) == 2


@pytest.mark.parametrize("arch", PROVIDER_OPTIONAL)
def test_half_a_binding_is_refused(tmp_path, arch):
    """One live constant beside one placeholder reads as a working spec and
    is not one — so it is an error, not a partial fill."""
    assert build.run(_args(archetype=arch, workspace=str(tmp_path / "a"),
                           model="m")) == 2
    assert build.run(_args(archetype=arch, workspace=str(tmp_path / "b"),
                           provider=_a_provider())) == 2


@pytest.mark.parametrize("arch", PROVIDER_OPTIONAL)
def test_in_process_still_requires_it(tmp_path, arch):
    """The embedded client IS the binding — there is no daemon to resolve a
    profile against — so the flags are required whatever the archetype."""
    assert build.run(_args(archetype=arch, workspace=str(tmp_path / arch),
                           transport="in_process")) == 2


def test_observer_never_emits_model_or_provider_constants(tmp_path):
    """Even WITH the flags.  The observer is read-only: it neither creates a
    session nor sends a message, so ``MODEL`` / ``PROVIDER`` would be two
    dead variables however they were supplied."""
    ws = tmp_path / "obs"
    assert build.run(_args(archetype="observer", workspace=str(ws),
                           provider=_a_provider(), model="m")) == 0
    src = (ws / "run_observer.py").read_text()
    tree = ast.parse(src)
    assigned = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    assert "MODEL" not in assigned and "PROVIDER" not in assigned


@pytest.mark.parametrize("arch", ["cascade", "sweep"])
def test_without_a_binding_the_placeholder_is_a_profile_name(tmp_path, arch):
    """The generated placeholder teaches the shape the reader is in.

    With a binding it stays an inline spec (runnable before any profile
    exists); without one it must not invent a provider — that default reads
    as guidance rather than the throwaway it is.
    """
    ws = tmp_path / arch
    assert build.run(_args(archetype=arch, workspace=str(ws))) == 0
    src = (ws / f"run_{arch}.py").read_text()
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    assert '"model": MODEL' not in code
    assert "<profile-name>" in code or '"your-profile"' in code


def test_without_a_binding_the_env_has_no_provider_stanza(tmp_path):
    """``.env`` is the one file a reader treats as configuration; a provider
    section for a provider nobody chose is a misleading default."""
    ws = tmp_path / "casc"
    assert build.run(_args(archetype="cascade", workspace=str(ws))) == 0
    env = (ws / ".env").read_text()
    active = [ln for ln in env.splitlines()
              if ln.strip() and not ln.lstrip().startswith("#")]
    assert not active, (
        f"the .env sets {active} with no provider bound; a provider nobody "
        "chose reads as guidance rather than the throwaway it is"
    )
    assert "---- provider:" not in env
    # ...but the framework knobs, which are the same whoever serves the
    # model, are still catalogued — commented out, as knobs always are.
    assert "---- daemon knobs" in env
    assert "# MODEL_NAME=" in env


def test_the_docs_say_the_same_thing():
    """``explain archetype`` is read INSTEAD of running the generator, so a
    ``requires`` list that disagrees with the generator is worse than none."""
    for name in PROVIDER_OPTIONAL:
        req = archetypes.resolve(name).requires
        assert "--provider" not in req and "--model" not in req
    for name in _OWNS_THE_BINDING:
        req = archetypes.resolve(name).requires
        assert "--provider" in req and "--model" in req
