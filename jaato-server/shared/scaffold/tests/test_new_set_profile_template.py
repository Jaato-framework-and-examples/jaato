"""What `jaato-scaffold new profile-set` writes into a tier-2 set profile.

Two defects, and the second only became visible once the first was fixed:

**No `description:`.**  The tier-1 base emitted one; the set profile did not,
and `description` is child-REPLACES — so every generated set profile
overrode its base's description with the empty string, which is the prose the
subagent tool advertises to the model.  `validate` now reports it
(`missing_description`), and the generator self-validates, so the template had
to supply one.

**A live `temperature: 0.0`.**  A valid provider knob, so `validate` passes
it, and a `400` on the reasoning models that accept only their default — a
generated profile bound to one of those could not make a single request.  It
is emitted COMMENTED OUT now, header included: a live `api_params:` over
nothing but comments parses as a YAML null, which `validate` correctly reports
as `unknown_knob`, so the generator would have failed the file it just wrote.
"""

import yaml

from shared.scaffold.build import _base_profile_yaml, _set_profile_yaml


def _parsed(**kw):
    return yaml.safe_load(_set_profile_yaml("worker", "openrouter",
                                            "anthropic/claude-sonnet-4.5", **kw))


def test_set_profile_carries_its_own_description():
    doc = _parsed()
    assert doc["description"].strip()
    assert "worker" in doc["description"]


def test_base_profile_still_carries_one():
    doc = yaml.safe_load(_base_profile_yaml("worker"))
    assert doc["description"].strip()


def test_description_asks_to_be_replaced():
    """A placeholder that does not say it is one just becomes the description."""
    assert "replace" in _parsed()["description"].lower()


def test_temperature_is_not_set_live():
    doc = _parsed()
    api_params = (doc.get("plugin_configs", {})
                     .get("openrouter", {}) or {}).get("api_params")
    assert api_params is None, "temperature must not be emitted live"


def test_the_api_params_header_is_commented_too():
    """A live header over only comments is a YAML null, which validate rejects."""
    doc = _parsed()
    cfg = doc.get("plugin_configs", {}).get("openrouter", {}) or {}
    assert "api_params" not in cfg


def test_the_knob_stays_discoverable_as_a_comment():
    text = _set_profile_yaml("worker", "openrouter", "m")
    assert "# temperature: 0.0" in text or "#   temperature: 0.0" in text
    assert "400" in text                     # says WHY it is commented out
    assert "explain provider openrouter" in text


def test_no_secret_still_documents_the_knob():
    """--secrets none emits no api_key, so the whole section would vanish."""
    text = _set_profile_yaml("worker", "openrouter", "m", kind="none")
    doc = yaml.safe_load(text)
    assert "plugin_configs" not in doc       # fully commented — no YAML null
    assert "temperature: 0.0" in text        # ... but still shown


def test_generated_set_profile_is_valid_yaml_with_the_expected_shape():
    doc = _parsed()
    assert doc["name"] == "worker"
    assert doc["inherits"] == ["_base_worker"]
    assert doc["provider"] == "openrouter"
    assert doc["model"] == "anthropic/claude-sonnet-4.5"
