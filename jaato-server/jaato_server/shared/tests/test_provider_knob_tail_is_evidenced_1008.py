"""A provider-knob finding claims only what the tree establishes (#1008).

The two PROVIDER-side ``unknown_knob`` sites ended every message with

    (silently ignored at runtime)

which is a runtime consequence the validator cannot observe -- the defect
class #910 describes, and #1005 fixed on the plugin side.  Here it was also
**wrong**, in the direction that matters most:

* for the fourteen providers inheriting ``_openai_compat``, an unrecognized
  ``api_params`` key is dropped BEFORE the request and ``_read_api_params``
  logs a WARNING naming it.  Loudly ignored, not silently;
* ``vllm`` FORWARDS ``audio`` and ``modalities`` (inherited through
  ``MEDIA_API_PARAMS``) while its own ``PROVIDER_KNOBS`` omits them, so
  ``main`` reported a key that demonstrably reaches the request as an
  ``error`` that is silently ignored.  A false positive at these sites, which
  #1008 recorded as having none.

THE TRAP THIS GUARD EXISTS FOR.  The obvious way to learn the allow-list is to
collect the string literals of the ``_FORWARDED_API_PARAMS`` assignment.  That
OVER-approximates, and ``minimax`` and ``mimo`` subtract three keys from the
base set::

    (OpenAICompatProvider._FORWARDED_API_PARAMS
     - frozenset({"frequency_penalty", "presence_penalty", "seed"}))

so a literal scan reports exactly the removed keys as forwarded -- and the
message then tells an author their key reaches the request while the provider
strips it.  That is #1008's own defect committed by its fix; it was written
that way first and caught by measuring it.  Hence
:func:`~shared.scaffold.introspect._eval_set_expr`, which evaluates ``|``,
``-``, ``&`` and the method spellings exactly, and answers ``None`` -- *not
established* -- rather than guessing.
"""

import re
from pathlib import Path

from jaato_server.shared.scaffold import introspect
from jaato_server.shared.scaffold import validate as V
from jaato_server.shared.tests.reversion import Reversion

_VALIDATE = "jaato-server/jaato_server/shared/scaffold/validate.py"
_INTROSPECT = "jaato-server/jaato_server/shared/scaffold/introspect.py"

_PROVIDER_DIR = (Path(__file__).resolve().parents[1]
                 / "plugins" / "model_provider")

#: The three keys `minimax` / `mimo` subtract from the base allow-list.
_SUBTRACTED = {"frequency_penalty", "presence_penalty", "seed"}


REVERSIONS = [
    Reversion(
        target=_VALIDATE,
        find='''    return ("error", "unknown_knob",
            f"not a valid {cfg_name} {layer_name} knob — PROVIDER_KNOBS "
            f"declares the accepted set and this key is outside it; what the "
            f"provider does with it next was not established here")''',
        replace='''    return ("error", "unknown_knob",
            f"not a valid {cfg_name} {layer_name} knob "
            "(silently ignored at runtime)")''',
        test="test_no_provider_finding_claims_silent_ignoring",
        because="the provider-side finding asserts a runtime consequence the "
                "validator cannot observe, and which is false for every "
                "allow-listing provider",
    ),
    Reversion(
        target=_INTROSPECT,
        find="""    if isinstance(node.op, ast.Sub):
        return left - right""",
        replace="""    if isinstance(node.op, ast.Sub):
        return left | right""",
        test="test_the_allow_list_honours_subtraction",
        because="the allow-list reader is back to over-approximating, so a key "
                "the provider strips is reported as reaching the request",
    ),
]


def _profile_findings(tmp_path, name, provider, model, body):
    """Validate one profile in a throwaway workspace; return its diagnostics."""
    pdir = tmp_path / ".jaato" / "profiles"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / f"{name}.yaml").write_text(
        f"name: {name}\ndescription: d\nprovider: {provider}\n"
        f"model: {model}\nplugins: []\n{body}")
    return V.validate_workspace(str(tmp_path), only=name)


def _codes(findings, code):
    return [d for d in findings if d.code == code]


# --------------------------------------------------- the claim that was wrong

def test_no_provider_finding_claims_silent_ignoring(tmp_path):
    """No provider-knob message may assert the key is silently ignored.

    Behavioural rather than a grep over the source: the phrase still appears
    in two docstrings that EXPLAIN why it was removed, and a source scan would
    either fail on those or have to special-case them -- at which point it is
    checking prose, not findings.

    Every BRANCH is swept, not one profile.  The first version checked a
    ``kimi`` profile only, so putting the phrase back on the no-allow-list
    tail left it green -- the meta-guard called it decorative, correctly: a
    check that exercises one of three exits says nothing about the other two.
    """
    cases = [
        # (provider, model, layer body, which tail it exercises)
        ("kimi", "kimi-k3",
         "    api_params:\n      nonsense_knob: 1\n", "allow-listed"),
        ("anthropic", "claude-sonnet-4-20250514",
         "    api_params:\n      nonsense_knob: 1\n", "no allow-list"),
        ("anthropic", "claude-sonnet-4-20250514",
         "    bogus_top_level: 2\n", "top-level"),
    ]
    for i, (provider, model, body, which) in enumerate(cases):
        findings = _profile_findings(
            tmp_path / f"c{i}", "p", provider, model,
            f"plugin_configs:\n  {provider}:\n{body}")
        assert _codes(findings, "unknown_knob"), (
            f"the {which} case produced no unknown_knob finding")
        for d in findings:
            assert "silently ignored at runtime" not in d.message, (
                f"{which}: {d.code} still asserts a runtime consequence the "
                f"validator cannot observe: {d.message}"
            )


def test_an_allow_listed_provider_says_the_key_is_dropped_loudly(tmp_path):
    """Where an allow-list governs the layer, the message says so, with a site.

    ``kimi`` declares its own allow-list, so the finding must name the drop and
    quote where the allow-list lives -- the evidence #1005 established as the
    standard for this family.
    """
    findings = _profile_findings(
        tmp_path, "p", "kimi", "kimi-k3",
        "plugin_configs:\n  kimi:\n    api_params:\n      nonsense_knob: 1\n")
    knob = _codes(findings, "unknown_knob")
    assert len(knob) == 1, [d.message for d in knob]
    msg = knob[0].message
    assert knob[0].severity == "error", "a closed declared set stays an error"
    assert "dropped before the request" in msg and "WARNING" in msg, msg
    assert re.search(r"\S+\.py:\d+", msg), f"no evidence site quoted: {msg}"


def test_the_quoted_allow_list_site_is_real(tmp_path):
    """The evidence must be checkable -- a file:line that says what we claim.

    A quoted location nobody verifies is the same thing as prose.
    """
    findings = _profile_findings(
        tmp_path, "p", "kimi", "kimi-k3",
        "plugin_configs:\n  kimi:\n    api_params:\n      nonsense_knob: 1\n")
    msg = _codes(findings, "unknown_knob")[0].message
    rel, lineno = re.search(r"\(([^()]+\.py):(\d+)\)", msg).groups()
    lines = (_PROVIDER_DIR / rel).read_text(encoding="utf-8").splitlines()
    assert "_FORWARDED_API_PARAMS" in lines[int(lineno) - 1], (
        f"{rel}:{lineno} does not declare the allow-list the message cites"
    )


def test_a_provider_with_no_allow_list_claims_nothing(tmp_path):
    """`anthropic` reads api_params key by key, so nothing is established.

    The honest answer is the weak one: the declaration was violated, full
    stop.  Asserting silence here would be the old defect with new wording.
    """
    findings = _profile_findings(
        tmp_path, "p", "anthropic", "claude-sonnet-4-20250514",
        "plugin_configs:\n  anthropic:\n    api_params:\n      nonsense: 1\n")
    knob = _codes(findings, "unknown_knob")
    assert len(knob) == 1, [d.message for d in knob]
    msg = knob[0].message
    assert "was not established" in msg, msg
    assert "dropped" not in msg and "WARNING" not in msg, (
        f"a consequence is claimed for a provider with no allow-list: {msg}")


def test_a_key_the_provider_forwards_blames_the_declaration(tmp_path):
    """`vllm` forwards `audio`; the finding must not call it invalid.

    This is the false positive `main` produced -- an ``error`` saying a key
    that reaches the request is silently ignored.  It becomes a WARNING about
    the incomplete DECLARATION, mirroring #1005's ``undeclared_knob``.
    """
    fwd = introspect.provider_api_params_forwarding("vllm")
    assert fwd and fwd["forwarded"] and "audio" in fwd["forwarded"], (
        "vllm no longer forwards 'audio' — pick another provider whose "
        "allow-list outruns its PROVIDER_KNOBS, or drop this test"
    )
    findings = _profile_findings(
        tmp_path, "p", "vllm", "Qwen/Qwen2.5-7B-Instruct",
        "plugin_configs:\n  vllm:\n    api_params:\n"
        "      audio: {voice: cedar}\n")
    assert not _codes(findings, "unknown_knob"), (
        "a key the provider demonstrably forwards is reported as invalid")
    und = _codes(findings, "undeclared_knob")
    assert len(und) == 1 and und[0].severity == "warn", [d.message for d in und]
    assert "does reach the request" in und[0].message, und[0].message


# ------------------------------------------- the allow-list reader itself

def test_the_allow_list_honours_subtraction():
    """`minimax` / `mimo` strip three keys; the reader must not report them.

    The literal-scan shortcut passes every other check in this module and
    fails only here, which is why it is a test of its own.
    """
    for provider in ("minimax", "mimo"):
        fwd = introspect.provider_api_params_forwarding(provider)
        assert fwd and fwd["forwarded"] is not None, (
            f"{provider}'s allow-list no longer evaluates statically")
        leaked = _SUBTRACTED & fwd["forwarded"]
        assert not leaked, (
            f"{provider} subtracts {sorted(_SUBTRACTED)} from the base "
            f"allow-list, and the reader reports {sorted(leaked)} as forwarded"
        )


def test_the_base_allow_list_evaluates_at_all():
    """A reader that answers `None` everywhere would pass the subtraction test.

    ``_openai_compat``'s own set is ``frozenset({...}) | MEDIA_API_PARAMS`` --
    a module-level constant, and the reason the first implementation resolved
    nothing.  "Not established" is the safe answer and it is not a free pass.
    """
    fwd = introspect.provider_api_params_forwarding("nebius")
    assert fwd and fwd["forwarded"] is not None
    assert {"temperature", "max_tokens"} <= fwd["forwarded"]
    assert {"modalities", "audio"} <= fwd["forwarded"], (
        "the module-level MEDIA_API_PARAMS constant is not being resolved")


def test_a_provider_without_an_allow_list_is_distinguishable_from_an_empty_one():
    """`None` (no allow-list) must not be confused with an empty set."""
    assert introspect.provider_api_params_forwarding("anthropic") is None
    assert introspect.provider_api_params_forwarding("google_genai") is None
    assert introspect.provider_api_params_forwarding("openrouter") is None
