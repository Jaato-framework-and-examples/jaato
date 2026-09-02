"""The env-scope catalog must match the code it describes.

WHAT THIS GUARDS.  ``shared/env_scope.py`` answers, for every env var the
framework reads, two questions the source scan cannot: is a per-session value
meaningful, and does a typed profile key already cover it (issue #775).  Those
are declarations, and a declaration that is not re-derived is a document.  The
one in the issue -- a first pass that reported ~110 "orphans" -- was wrong
within a day of being written, because it compared env names against top-level
profile fields only and so read every nested block as a gap.

So this module re-derives the var list from the installed source on every run
and fails when the catalog and the code disagree in EITHER direction:

  * a var the code reads and the catalog does not classify -- the case the
    issue asks for by name: "a guard that fails when a new ``os.getenv`` lands
    in session-scoped code with no typed equivalent and no host/read-only tag";
  * a catalog entry for a var nothing reads any more -- stale, and a stale
    entry is how a catalog starts lying;
  * a ``typed_key`` naming a profile field, nested block field or provider knob
    that does not exist -- the claim "this is already covered" is worthless
    unresolved, and the ~110-orphan number is what unverified claims cost;
  * a session-scoped var without a typed key that is missing from (or stale in)
    the :data:`~shared.env_scope.AWAITING_TYPED_KEY` ratchet.

WHY THE RATCHET IS SET EQUALITY AND NOT A COUNT.  A ceiling ("no more than 43")
lets one promotion pay for one regression, silently.  Set equality means
promoting a knob is a deletion and adding an untyped session knob is an
addition -- both in a diff, both under review, both next to the words "may only
shrink".  ``test_session_env_audit.py``'s ALLOWLIST is the same mechanism for
the orthogonal question of read ROUTE, and it is the repository's own
precedent.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Set

import pytest

from shared import env_scope
from shared.env_scope import (
    AWAITING_TYPED_KEY,
    CATALOG,
    SCOPES,
    SESSION,
    EnvClass,
)
from shared.scaffold import introspect

# The plugin NAME list comes from the directory listing, never from a live
# ``registry.discover()``: discovery imports every plugin, so an unrelated
# dependency skew (an MCP SDK version bump, say) would fail this guard for a
# reason that has nothing to do with the catalog. ``builtin_plugin_names`` is
# the same ``pkgutil`` read the entry-point trust policy uses.
from shared.plugins.entry_point_trust import builtin_plugin_names


# --------------------------------------------------------------------------
# Reversions -- see test_every_guard_detects_its_own_reversion.py.  A guard
# that cannot notice its own reversion is not evidence.
# --------------------------------------------------------------------------
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

REVERSIONS = [
    Reversion(
        target="jaato-server/shared/env_scope.py",
        find='    "JAATO_PROVIDER_TRACE": EnvClass(SESSION, "trace.provider_log",',
        replace='    "JAATO_PROVIDER_TRACE": EnvClass(SESSION, "trace.no_such_key",',
        because="a typed_key that names nothing must not read as coverage",
        test="test_every_typed_key_resolves",
    ),
    Reversion(
        target="jaato-server/shared/env_scope.py",
        find='    "JAATO_PARALLEL_TOOLS": EnvClass(SESSION, None,',
        replace='    "JAATO_PARALLEL_TOOLS": EnvClass(HOST, None,',
        because=("a session knob relabelled `host` drops out of the ratchet "
                 "-- the exact way a catalog is 'fixed' into a lie"),
        test="test_ratchet_matches_the_catalog",
    ),
    Reversion(
        target="jaato-server/shared/env_scope.py",
        find='        "A", "tools.parallel",',
        replace='        "A", "plugin_configs.no_such_plugin.parallel",',
        because=("a proposal aimed at a plugin nobody ships is a typo, and "
                 "a typo in a plan is a plan nobody can follow"),
        test="test_proposed_keys_name_a_real_owner",
    ),
    Reversion(
        target="jaato-server/shared/env_scope.py",
        find='    "PERMISSION_WEBHOOK_TOKEN": EnvClass(SESSION, "plugin_configs.permission.channel_config.auth_token",',
        replace='    "PERMISSION_WEBHOOK_TOKEN": EnvClass(SESSION, "plugin_configs.permission.no_such_block.nonexistent",',
        because=("a non-provider plugin key was verified only as far as the "
                 "plugin NAME, so a bogus knob path passed as coverage"),
        test="test_every_typed_key_resolves",
    ),
]


def _scanned() -> Dict[str, object]:
    return introspect.env_vars()


def _untyped_session_vars(catalog: Dict[str, EnvClass]) -> Set[str]:
    return {name for name, entry in catalog.items()
            if entry.scope == SESSION and not entry.typed_key}


# --------------------------------------------------------------------------
# Completeness, in both directions
# --------------------------------------------------------------------------

def test_every_env_var_the_code_reads_is_classified():
    """A new ``os.getenv`` must be classified before it can ship.

    This is the guard the issue asks for.  The failure message names the
    var and the four answers, because the cost of getting this wrong is
    paid by whoever reads the catalog later, not by whoever skipped it.
    """
    unclassified = sorted(set(_scanned()) - set(CATALOG))
    assert not unclassified, (
        "env vars are read but not classified in shared/env_scope.py:\n"
        + "\n".join(f"  - {n}" for n in unclassified)
        + "\n\nAdd each to CATALOG with the scope that is TRUE of it:\n"
          "  SESSION   two sessions on one host may legitimately differ.\n"
          "            Give it a typed_key, or add it to AWAITING_TYPED_KEY\n"
          "            with its tier and accept that it is tech debt.\n"
          "  HOST      process/host-scoped; a per-session value would be a lie.\n"
          "  AMBIENT   the host environment being READ, not configured.\n"
          "  INTERNAL  one framework process handing a value to another."
    )


def test_no_stale_catalog_entries():
    """A catalog entry for a var nothing reads is a lie in waiting."""
    stale = sorted(set(CATALOG) - set(_scanned()))
    assert not stale, (
        "shared/env_scope.py classifies vars the installed tree no longer "
        "reads:\n" + "\n".join(f"  - {n}" for n in stale)
        + "\n\nDelete them. A catalog that keeps entries for deleted code "
          "cannot be trusted about the code that is left."
    )


def test_every_scope_is_a_declared_scope():
    """No scope arrives by typo."""
    bad = {n: e.scope for n, e in CATALOG.items() if e.scope not in SCOPES}
    assert not bad, f"unknown scope(s): {bad}. Declared: {list(SCOPES)}"


def test_only_session_vars_carry_a_typed_key():
    """A typed profile key for a host/ambient/internal var is a category error.

    A profile field for the cgroup root or for ``PATH`` would promise a
    per-session value the framework cannot deliver.
    """
    bad = {n: (e.scope, e.typed_key) for n, e in CATALOG.items()
           if e.typed_key and e.scope != SESSION}
    assert not bad, (
        f"non-session vars carrying a typed_key: {bad}. Either the scope is "
        f"wrong or the key promises something per-session that is not."
    )


def test_every_entry_says_why():
    """Every classification carries its one-line reason.

    Load-bearing for ``host`` and ``internal`` especially: their whole job
    is to stop a later reader "fixing" something that is already right,
    and an unexplained tag cannot do that.
    """
    silent = sorted(n for n, e in CATALOG.items() if not e.note.strip())
    assert not silent, (
        "catalog entries with no note:\n" + "\n".join(f"  - {n}" for n in silent)
    )


# --------------------------------------------------------------------------
# The ratchet
# --------------------------------------------------------------------------

def test_ratchet_matches_the_catalog():
    """AWAITING_TYPED_KEY is exactly the session vars with no typed key.

    Set equality, not a ceiling: a ceiling lets one promotion silently pay
    for one regression.
    """
    derived = _untyped_session_vars(CATALOG)
    declared = set(AWAITING_TYPED_KEY)
    missing = sorted(derived - declared)
    stale = sorted(declared - derived)
    assert not missing, (
        "session-scoped vars with no typed key are missing from "
        "AWAITING_TYPED_KEY:\n" + "\n".join(f"  - {n}" for n in missing)
        + "\n\nEither give the var a typed_key (promote it), or add it here "
          "with its tier (A behaviour / B plugin knob / E credential) and "
          "accept the entry as declared tech debt."
    )
    assert not stale, (
        "AWAITING_TYPED_KEY lists vars that are no longer session-scoped-"
        "and-untyped:\n" + "\n".join(f"  - {n}" for n in stale)
        + "\n\nIf you promoted one, delete its line -- that is what makes "
          "this set shrink. If you relabelled its scope, say why in the "
          "catalog note."
    )


def test_ratchet_tiers_are_from_the_assessment():
    """Only the three tiers the assessment defines."""
    bad = {n: a.tier for n, a in AWAITING_TYPED_KEY.items()
           if a.tier not in ("A", "B", "E")}
    assert not bad, (
        f"unknown tier(s) {bad}. A = agent-behaviour knob, B = plugin knob, "
        f"E = credential (needs the policy decided, not a default)."
    )


def test_every_ratchet_entry_proposes_a_home():
    """A debt entry must say where the key should go.

    "This wants a typed key" without "and it belongs here" is a
    complaint, not a plan -- and it is what makes a ratchet sit at the
    same size for a year. The proposal is also the reviewable claim: a
    wrong `proposed_key` can be argued with, a missing one cannot.
    """
    silent = sorted(n for n, a in AWAITING_TYPED_KEY.items()
                    if not a.proposed_key.strip())
    assert not silent, (
        "ratchet entries with no proposed_key:\n"
        + "\n".join(f"  - {n}" for n in silent)
    )


def test_proposed_keys_name_a_real_owner():
    """The proposal's SHAPE is checked even though it cannot resolve.

    A ``proposed_key`` names something that does not exist yet -- that
    is what makes it a proposal -- so the guard cannot resolve it the
    way it resolves ``typed_key``. What it CAN check is that the owner
    exists: ``plugin_configs.<x>`` must name a real plugin or provider,
    and a top-level proposal must not collide with an unrelated
    existing field. A proposal aimed at a plugin nobody ships is a
    typo, and a typo in a plan is a plan nobody can follow.
    """
    from shared.plugins.subagent import config as cfg

    known_plugins = set(builtin_plugin_names())
    profile_fields = {f.name for f in dataclasses.fields(cfg.SubagentProfile)}
    bad = []
    for name in sorted(AWAITING_TYPED_KEY):
        key = AWAITING_TYPED_KEY[name].proposed_key
        if key.startswith("plugin_configs."):
            owner = key.split(".")[1]
            if (owner not in known_plugins
                    and introspect.resolve_provider(owner) is None):
                bad.append((name, key, f"no plugin or provider {owner!r}"))
            continue
        head = key.split(".")[0]
        # A top-level proposal is either a NEW block (tools:, retry:) or an
        # extension of one that exists (trace.log_dir). Both are fine; a
        # proposal that lands on an unrelated existing scalar is not.
        if head in profile_fields and head not in ("trace", "gc", "cache"):
            bad.append((name, key, f"collides with existing profile field {head!r}"))

    assert not bad, (
        "proposed_key entries that name nothing usable:\n"
        + "\n".join(f"  - {n}: {k}  ({why})" for n, k, why in bad)
    )


# --------------------------------------------------------------------------
# The typed keys must actually resolve
# --------------------------------------------------------------------------

def _profile_field_type(path: str) -> Optional[type]:
    """Resolve a dotted profile path to the type of its final field.

    ``model`` -> ``SubagentProfile.model``; ``gc.threshold_percent`` ->
    the field on ``GCProfileConfig``.  Returns ``None`` when any segment
    is missing, which is what the caller reports as an unresolved claim.
    """
    from shared.plugins.subagent import config as cfg

    current = cfg.SubagentProfile
    for segment in path.split("."):
        fields = {f.name: f for f in dataclasses.fields(current)}
        if segment not in fields:
            return None
        annotation = fields[segment].type
        # Nested block fields are annotated as strings / Optional[X]; map
        # the two the catalog actually uses rather than evaluating
        # arbitrary annotations.
        nested = {
            "gc": cfg.GCProfileConfig,
            "cache": cfg.CacheProfileConfig,
            "trace": cfg.TraceProfileConfig,
        }
        current = nested.get(segment, object)
        if current is object:
            return type(annotation)
    return current


def _provider_knob_exists(provider: str, knob_path: str) -> bool:
    """True when *knob_path* names a knob the provider declares.

    ``api_key`` is looked for in every layer; ``api_params.thinking_budget``
    is looked for in the named layer.  An opaque layer (OpenRouter's
    ``routing``, LM Studio's ``load``) accepts any key by contract, so a
    knob under one resolves by construction.
    """
    info = introspect.resolve_provider(provider)
    if info is None or info.knobs is None:
        return False
    if "." in knob_path:
        layer_name, knob = knob_path.split(".", 1)
        for layer in info.knobs.layers:
            if layer.layer != layer_name:
                continue
            return layer.opaque or any(k.name == knob for k in layer.knobs)
        return False
    for layer in info.knobs.layers:
        if any(k.name == knob_path for k in layer.knobs):
            return True
    return False


def _plugin_knob_unresolved(plugin: str, knob_path: str) -> Optional[str]:
    """Why *knob_path* does not resolve against *plugin*, or ``None``.

    THE HALF THIS GUARD USED TO MISS.  Provider keys were verified down to
    the knob, but a non-provider plugin key was verified only as far as the
    plugin NAME -- so ``plugin_configs.permission.no_such_block.nonexistent``
    passed. That is exactly the "unverified claim is indistinguishable from
    a wrong one" failure this module's docstring is about, and every key the
    correcting commit added (permission, mermaid_formatter, todo) fell in
    the unverified half.

    Every segment must be a key the plugin actually consumes, so a nested
    path (``channel_config.auth_token``) is checked at both levels rather
    than only at its root.

    A plugin that reads no config at all is reported as unverifiable
    rather than passed: "we could not check" and "we checked and it is
    fine" are different answers, and only one of them is coverage.
    """
    known = introspect.plugin_config_keys(plugin)
    if not known:
        return (f"{plugin} declares no config surface to resolve against "
                f"(cannot verify — not the same as verified)")
    missing = [seg for seg in knob_path.split(".") if seg not in known]
    if missing:
        return f"{plugin} reads no config key(s) {missing}"
    return None


def test_every_typed_key_resolves():
    """A ``typed_key`` must name something that exists.

    This is the check the issue's first pass could not make, and the
    reason its orphan count was wrong: "already covered by a nested block"
    is a claim, and an unverified claim is indistinguishable from a wrong
    one.
    """
    from jaato_sdk.events import ClientConfigRequest

    unresolved = []
    for name in sorted(CATALOG):
        key = CATALOG[name].typed_key
        if not key:
            continue
        if key.startswith("client."):
            field = key.split(".", 1)[1]
            if field not in ClientConfigRequest.model_fields:
                unresolved.append((name, key, "no such ClientConfigRequest field"))
        elif key.startswith("plugin_configs."):
            _, plugin, rest = key.split(".", 2)
            if introspect.resolve_provider(plugin) is not None:
                if not _provider_knob_exists(plugin, rest):
                    unresolved.append((name, key, f"no such {plugin} knob"))
            elif plugin not in builtin_plugin_names():
                unresolved.append((name, key, "no such plugin"))
            else:
                why = _plugin_knob_unresolved(plugin, rest)
                if why:
                    unresolved.append((name, key, why))
        elif _profile_field_type(key) is None:
            unresolved.append((name, key, "no such profile field"))

    assert not unresolved, (
        "typed_key entries that name nothing:\n"
        + "\n".join(f"  - {n}: {k}  ({why})" for n, k, why in unresolved)
    )


def test_the_trace_block_covers_exactly_its_two_env_vars():
    """The promotion this catalog shipped with stays wired.

    ``trace:`` exists to give the two trace paths a typed home; if the
    block and the catalog drift apart, the promotion has silently been
    undone and the catalog would still claim coverage.
    """
    from shared.plugins.subagent.config import TRACE_ENV_VARS

    claimed = {name for name, e in CATALOG.items()
               if (e.typed_key or "").startswith("trace.")}
    assert claimed == set(TRACE_ENV_VARS.values()), (
        f"catalog claims trace.* covers {sorted(claimed)}, the block seeds "
        f"{sorted(TRACE_ENV_VARS.values())}"
    )


# --------------------------------------------------------------------------
# The knob view
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["PATH", "TERM", "HOME", "USER"])
def test_host_environment_is_not_a_knob(name):
    """The vars that inflate a naive diff are excluded from the knob view."""
    if name not in CATALOG:  # not read by this build
        pytest.skip(f"{name} is not read by the installed tree")
    assert not env_scope.is_knob(name)


def test_unclassified_counts_as_a_knob():
    """The conservative answer, so a gap is visible rather than hidden."""
    assert env_scope.is_knob("JAATO_SOMETHING_NOBODY_CLASSIFIED_YET")
