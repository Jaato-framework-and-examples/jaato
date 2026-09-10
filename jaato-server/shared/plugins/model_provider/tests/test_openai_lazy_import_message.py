"""The message a missing `openai` produces is the only thing the user sees.

`openai` is an optional extra, and the wire that needs it is inherited rather
than written: `azure_openai` (and most of the OpenAI-compatible fleet) owns no
`import openai` at all.  So the failure surfaces in shared machinery, and
"Install with: pip install openai" sent the reader looking for a dependency
their provider's directory does not mention — while `jaato-scaffold explain
provider azure_openai deps` named `azure`, the other missing package, and not
this one.

The message therefore has to carry three things: that this is an extra rather
than a broken install, a command that works, and where to find the extra for
the provider actually in use.
"""
import sys

import pytest

from shared.plugins.model_provider._openai_compat import _lazy


@pytest.fixture
def openai_unimportable(monkeypatch):
    """Make `import openai` fail regardless of what is installed."""
    monkeypatch.setattr(_lazy, "_openai_module", None)
    monkeypatch.setitem(sys.modules, "openai", None)
    yield


def test_the_message_says_it_is_an_extra_not_a_broken_install(openai_unimportable):
    with pytest.raises(ImportError) as exc:
        _lazy.get_openai_module()
    text = str(exc.value)
    assert "extra" in text.lower()
    assert "jaato-server[openai]" in text


def test_the_message_points_at_the_command_that_names_the_right_extra(openai_unimportable):
    """Five extras declare `openai`; only the report knows which is yours."""
    with pytest.raises(ImportError) as exc:
        _lazy.get_openai_module()
    assert "jaato-scaffold explain provider" in str(exc.value)
    assert "deps" in str(exc.value)
