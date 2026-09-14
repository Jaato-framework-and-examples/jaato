"""Two findings about a profile that is correct and cannot run.

``provider_dependency_missing``
    a provider whose vendor SDK lives behind an optional extra, on a machine
    that does not have it.  The profile validates, and the session dies at
    ``connect()`` with an ImportError several layers from anything the author
    wrote.  The validator knew the provider name and never asked the
    question: it did not import :mod:`shared.scaffold.dependencies` at all,
    while both halves of the answer already lived there.

``completion_processors_without_schema``
    processors declared behind a tool that is not on the wire.  Gate 1 of
    ``LifecycleTools._should_hide_signal_completion`` hides
    ``signal_completion`` whenever no ``completion_payload_schema`` is
    declared, so the gate can never run — the agent hunts for the tool, the
    nudge budget drains, and the driver is handed ``None`` by a session that
    looked like it ran.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from shared.plugins.subagent.config import CompletionProcessor, SubagentProfile
from shared.scaffold import dependencies, introspect, validate


@pytest.fixture(scope="module")
def registry():
    return introspect.providers(), introspect.plugins()


def _run(registry, **kw):
    providers, plugins = registry
    kw.setdefault("plugins", [])
    profile = SubagentProfile(name="t", description="d", **kw)
    return validate.validate_profile(profile, providers=providers,
                                     plugins=plugins, gc_names=[])


def _codes(diags, code):
    return [d for d in diags if d.code == code]


# ------------------------------------------------------- provider SDK gaps

def test_a_missing_import_resolves_to_the_extra_that_supplies_it():
    """The mapping the finding's imperative tail is built from.

    Asserted on the index rather than on a live call, because whether
    ``openai`` is importable is a property of the machine running the suite
    and this is a property of the packaging metadata.
    """
    hint = dependencies._install_hint(["openai"], "azure_openai")
    assert any("azure-openai" in c for c in hint["commands"]), hint


def test_provider_import_gaps_imports_nothing():
    """``validate`` must stay side-effect free.

    A provider SDK's import can register handlers, read the environment or
    open a config file, so this probes with ``find_spec`` where ``_health``
    imports.  Measured in a CLEAN interpreter — inside this session the
    vendor module may already be in ``sys.modules`` for unrelated reasons,
    and a check that cannot fail proves nothing.
    """
    probe = textwrap.dedent("""
        import sys
        from shared.scaffold import dependencies
        dependencies.provider_import_gaps("anthropic")
        print("anthropic" in sys.modules)
    """)
    out = subprocess.run([sys.executable, "-c", probe], check=True,
                         capture_output=True, text=True)
    assert out.stdout.strip().endswith("False"), out.stdout


def test_the_probe_would_notice_an_import():
    """The control run: the same shape, with an import, reports True.

    Without it the test above passes on a typo in the probe just as
    happily as on a correct implementation.
    """
    probe = textwrap.dedent("""
        import sys
        from shared.scaffold import dependencies
        dependencies._health(["anthropic"])
        print("anthropic" in sys.modules)
    """)
    out = subprocess.run([sys.executable, "-c", probe], check=True,
                         capture_output=True, text=True)
    assert out.stdout.strip().endswith("True"), out.stdout


def test_an_unknown_provider_package_asserts_nothing():
    # No in-tree source to parse is not evidence that anything is missing.
    assert dependencies.provider_import_gaps("not_a_provider") == ((), ())


@pytest.fixture()
def one_gap(monkeypatch):
    """Report ``echo`` as missing an SDK, whatever this machine has installed.

    The validator's WIRING is what these assert — that the question is asked
    at all, for the flat provider and for each tier's — and driving that off
    a real provider would make the test pass or fail on which extras the
    suite happens to be running under.
    """
    def _gaps(name):
        if name == "echo":
            return ("pretend_sdk",), ("pip install 'jaato-server[echo]'",)
        return (), ()
    monkeypatch.setattr(dependencies, "provider_import_gaps", _gaps)


def test_a_missing_sdk_is_a_warning_naming_the_fix(registry, one_gap):
    out = _codes(_run(registry, provider="echo", model="m"),
                 "provider_dependency_missing")
    assert len(out) == 1
    assert out[0].severity == "warn"        # CI may validate for another host
    assert "pip install" in out[0].message
    assert "pretend_sdk" in out[0].message


def test_an_installed_provider_is_silent(registry, one_gap):
    assert _codes(_run(registry, provider="anthropic", model="m"),
                  "provider_dependency_missing") == []


def test_a_tier_provider_is_checked_too(registry, one_gap):
    # A tier binds a (provider, model) PAIR, and the second provider is the
    # one nobody notices until enter_tier.
    out = _codes(_run(registry, provider="anthropic", model="m",
                      model_tiers={"voz": {"model": "m", "provider": "echo"}}),
                 "provider_dependency_missing")
    assert [d.where for d in out] == ["provider.echo"]


# ------------------------------------------------------ the gate's own shape

_PROC = [CompletionProcessor(script="acceptance.py")]


def test_processors_without_a_schema_is_an_error(registry):
    out = _codes(_run(registry, provider="echo", model="m",
                      completion_processors=_PROC),
                 "completion_processors_without_schema")
    assert [d.severity for d in out] == ["error"]


def test_processors_with_a_schema_are_fine(registry):
    assert _codes(_run(registry, provider="echo", model="m",
                       completion_processors=_PROC,
                       completion_payload_schema="completion_schemas/x.json"),
                  "completion_processors_without_schema") == []


def test_an_abstract_base_is_silent(registry):
    # Processors are inherited, so every concrete descendant is checked in
    # resolved form — where the pair is completed by the child or reported
    # against it.  Nothing is lost by staying quiet here.
    assert _codes(_run(registry, completion_processors=_PROC),
                  "completion_processors_without_schema") == []


def test_a_tiers_profile_counts_as_binding_a_model(registry):
    out = _codes(_run(registry, provider="echo",
                      model_tiers={"planner": {"model": "m"}},
                      completion_processors=_PROC),
                 "completion_processors_without_schema")
    assert len(out) == 1
