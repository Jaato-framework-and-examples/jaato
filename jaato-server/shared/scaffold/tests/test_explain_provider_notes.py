"""`explain provider <name>` must carry the caveats no knob table can express.

Two findings, one page:

**The Azure `model:` field.**  In an Azure OpenAI profile, `model:` is the
DEPLOYMENT name from the resource, not a catalog model id.  The provider
module's own docstring said so; `explain provider azure_openai` did not, and
that page is what a session reads.  Getting it wrong fails at request time
with `DeploymentNotFound`, and nothing before that names the cause.
`PROVIDER_NOTES` is declared beside the capabilities and knobs so the note
cannot drift from the provider it describes.

**The scope of the `api_params` check.**  `validate` checks an `api_params`
key against the PROVIDER's allow-list — the only thing it can check — so
`temperature: 0.0` validates clean and is a `400` on a model that accepts only
its default.  The page now says where the check stops, and that omitting a
parameter is always safe.  (Deliberately NOT a per-model incompatibility
table: a stale row would reject what the vendor accepts, or pass what it
rejects, and either beats saying nothing only by accident.)
"""

import pytest

from shared.scaffold import introspect
from shared.scaffold.explain import provider


def test_azure_note_says_model_is_the_deployment_name():
    data, text = provider("azure_openai")
    joined = " ".join(data["notes"])
    assert "DEPLOYMENT" in joined
    assert "DeploymentNotFound" in joined
    assert "read this first:" in text
    assert "DEPLOYMENT" in text                     # rendered, not just in JSON


def test_azure_notes_are_rendered_above_the_knobs():
    """A knob table read under the wrong premise is still read wrong."""
    _, text = provider("azure_openai")
    assert text.index("read this first:") < text.index("knobs (")


def test_notes_are_optional_and_absent_providers_render_unchanged():
    info = introspect.resolve_provider("anthropic")
    assert info.notes == ()
    data, text = provider("anthropic")
    assert data["notes"] == []
    assert "read this first:" not in text


@pytest.mark.parametrize("name", ["azure_openai", "anthropic", "openrouter"])
def test_api_params_caveat_appears_wherever_the_layer_does(name):
    info = introspect.resolve_provider(name)
    has_layer = bool(info.knobs) and any(
        l.layer == "api_params" for l in info.knobs.layers)
    _, text = provider(name)
    assert has_layer, f"{name} was expected to declare an api_params layer"
    assert "api_params — scope of the check:" in text
    assert "OMITTING a parameter is always safe" in text


def test_every_declared_note_is_a_string():
    """`PROVIDER_NOTES` is prose for a human; a stray tuple would render as one."""
    for name, info in introspect.providers().items():
        for note in info.notes:
            assert isinstance(note, str) and note.strip(), name
