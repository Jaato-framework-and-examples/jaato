"""Validator knob checks: an api_key_param credential key is a valid top-level
knob (FINDING 1); vllm's api_params layer is honored (FINDING 2).  Both were
false positives because the declaration didn't reach the validator."""
from types import SimpleNamespace
from shared.scaffold import introspect
from shared.scaffold.validate import validate_profile


def _validate(provider, plugin_configs):
    prof = SimpleNamespace(provider=provider, model="m", plugin_configs=plugin_configs)
    return validate_profile(
        prof, providers=introspect.providers(), plugins=introspect.plugins(),
        gc_names=list(introspect.gc_strategies().keys()))


def test_zhipuai_api_key_is_accepted_not_unknown_knob():
    diags = _validate("zhipuai", {"zhipuai": {"api_key": "x"}})
    assert not any(d.code == "unknown_knob" and "api_key" in (d.where or "")
                   for d in diags), [(d.code, d.where) for d in diags]


def test_bogus_zhipuai_top_level_knob_is_still_flagged():
    diags = _validate("zhipuai", {"zhipuai": {"bogus_knob": "x"}})
    assert any(d.code == "unknown_knob" and "bogus_knob" in (d.where or "")
               for d in diags)


def test_vllm_api_token_credential_accepted():
    # vllm's api_key_param AuthSource name is `api_token`
    diags = _validate("vllm", {"vllm": {"api_token": "t"}})
    assert not any(d.code == "unknown_knob" and "api_token" in (d.where or "")
                   for d in diags), [(d.code, d.where) for d in diags]


def test_vllm_api_params_layer_and_temperature_accepted():
    diags = _validate("vllm", {"vllm": {"api_params": {"temperature": 0}}})
    assert not any(d.code == "unknown_layer" for d in diags), \
        [(d.code, d.where) for d in diags]
    assert not any(d.code == "unknown_knob" and "temperature" in (d.where or "")
                   for d in diags)
