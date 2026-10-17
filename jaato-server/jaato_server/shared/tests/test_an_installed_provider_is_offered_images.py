"""A provider from another distribution is offered images like a built-in one.

``decide`` offers ``images`` and the ``box`` question type only to a
session with a decision model whose provider declares ``decision_images``
(decision-models.md §12.5).  The capability is read without building the
provider, from the provider package's ``PROVIDER_CAPABILITIES``.  Until
this change that read only knew the built-in package, so a provider
registered under the ``jaato.model_providers`` entry point loaded and ran
but was never offered images.

These tests install a real package on ``sys.path`` and a real
``EntryPoint`` for it, and drive the real ``load_provider``, the real
capability read, the real ``JaatoSession`` and the real decision plugin
against the local stand-in.  No test touches the network.
"""

from __future__ import annotations

import base64
import importlib.metadata
import sys
import textwrap
from unittest.mock import MagicMock

import pytest

from jaato_server.shared.jaato_session import JaatoSession
from jaato_server.shared.plugins.decision.plugin import TOOL_NAME, DecisionPlugin
from jaato_server.shared.plugins.model_provider import (
    MODEL_PROVIDER_ENTRY_POINT,
    load_provider,
)
from jaato_server.shared.plugins.model_provider._decision_capability import (
    provider_declares_decision_images,
)
from jaato_server.shared.plugins.model_provider.typesafe.provider import (
    TypeSafeProvider,
)
from jaato_server.shared.session_context import (
    isolated_current_session,
    reset_workspace_root,
    set_current_session,
    set_workspace_root,
)
from jaato_server.shared.tests.decision_standin import (
    DecisionStandIn,
    png_bytes,
    typesafe_against,
)
from jaato_server.shared.tests.reversion import Reversion

_CAPABILITY = ("jaato-server/jaato_server/shared/plugins/model_provider/"
               "_decision_capability.py")

REVERSIONS = [
    Reversion(
        target=_CAPABILITY,
        find="    entry_module = _entry_point_module(provider_name)\n",
        replace="    entry_module = None\n",
        test="test_an_entry_point_provider_declares_images",
        because="only built-in providers are asked, so a provider from "
                "another distribution is never offered images",
    ),
    Reversion(
        target=_CAPABILITY,
        find="        parts.pop()\n",
        replace="        break\n",
        test="test_an_entry_point_provider_declares_images",
        because="a provider declaring its capabilities in its package "
                "__init__, as built-in providers do, reads as declaring none",
    ),
]

_PROVIDER_SRC = textwrap.dedent("""\
    from jaato_server.shared.plugins.model_provider.typesafe.provider import (
        TypeSafeProvider,
    )

    def create_provider():
        return TypeSafeProvider()
""")

_CAPS_SRC = textwrap.dedent("""\
    from jaato_server.shared.plugins.model_provider.base import (
        ProviderCapabilities,
    )

    PROVIDER_CAPABILITIES = ProviderCapabilities(
        chat=False, decisions=True, decision_images={images})
""")


@pytest.fixture(autouse=True)
def _restore_session():
    with isolated_current_session():
        yield


@pytest.fixture
def standin():
    with DecisionStandIn() as server:
        yield server


@pytest.fixture
def install(tmp_path, monkeypatch):
    """Install packages and register them as ``jaato.model_providers``."""
    registered = []
    real = importlib.metadata.entry_points

    def fake_entry_points(**params):
        if params.get("group") == MODEL_PROVIDER_ENTRY_POINT:
            return [importlib.metadata.EntryPoint(name, value,
                                                  MODEL_PROVIDER_ENTRY_POINT)
                    for name, value in registered]
        return real(**params)

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    monkeypatch.syspath_prepend(str(tmp_path))

    def _install(ep_name, package, images=True):
        """``images`` True/False declares it; ``None`` declares nothing."""
        root = tmp_path / package
        root.mkdir()
        (root / "__init__.py").write_text(
            "" if images is None else _CAPS_SRC.format(images=images))
        (root / "provider.py").write_text(_PROVIDER_SRC)
        registered.append((ep_name, f"{package}.provider:create_provider"))
        monkeypatch.delitem(sys.modules, package, raising=False)
        monkeypatch.delitem(sys.modules, f"{package}.provider", raising=False)

    yield _install
    for name in [m for m in sys.modules if m.startswith("jaato_test_ext_")]:
        sys.modules.pop(name, None)


# ============================================================ the provider runs


def test_an_entry_point_provider_is_loaded_from_its_distribution(install):
    install("eyes_ext", "jaato_test_ext_load")
    provider = load_provider("eyes_ext")
    assert isinstance(provider, TypeSafeProvider)


# ============================================================ the capability


def test_an_entry_point_provider_declares_images(install):
    install("eyes_ext", "jaato_test_ext_caps")
    assert provider_declares_decision_images("eyes_ext") is True


def test_a_package_declaring_nothing_is_not_offered_images(install):
    install("blind_ext", "jaato_test_ext_none", images=None)
    assert provider_declares_decision_images("blind_ext") is False


def test_a_package_declaring_false_is_not_offered_images(install):
    install("plain_ext", "jaato_test_ext_false", images=False)
    assert provider_declares_decision_images("plain_ext") is False


def test_an_unknown_provider_is_not_offered_images(install):
    assert provider_declares_decision_images("nobody_ext") is False


def test_a_built_in_provider_wins_a_name_clash(install):
    # The directory scan wins in discover_providers, so it wins here too.
    install("typesafe", "jaato_test_ext_clash_ts", images=False)
    install("laya", "jaato_test_ext_clash_laya", images=True)
    assert provider_declares_decision_images("typesafe") is True
    assert provider_declares_decision_images("laya") is False


# ============================================================ end to end


def _session(standin):
    runtime = MagicMock()
    runtime.provider_name = "eyes_ext"
    runtime.ledger = MagicMock()
    runtime.create_provider.side_effect = (
        lambda model=None, provider_name=None, **_kw:
        typesafe_against(standin, model="jev-1.14.0"))
    runtime.get_tool_schemas.return_value = []
    runtime.get_executors.return_value = {}
    runtime.get_system_instructions.return_value = None
    runtime.permission_plugin = None
    runtime.reliability_plugin = None
    runtime.registry = MagicMock()
    runtime.registry.get_exposed_tool_schemas.return_value = []
    session = JaatoSession(runtime, "chat-model", provider_name="eyes_ext")
    session.configure(decision_models={
        "eyes": {"provider": "eyes_ext", "model": "jev-1.14.0"}})
    set_current_session(session)
    return session


def test_decide_offers_and_sends_images_for_an_entry_point_provider(
        install, standin, tmp_path):
    install("eyes_ext", "jaato_test_ext_e2e")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    data = png_bytes(width=400, height=300)
    (workspace / "door.png").write_bytes(data)
    token = set_workspace_root(str(workspace))
    try:
        session = _session(standin)
        assert session.decision_image_model_names() == ["eyes"]
        plugin = DecisionPlugin()
        schema = plugin.narrow_tool_schema(plugin.get_tool_schemas()[0])
        assert "images" in schema.parameters["properties"]

        result = plugin.get_executors()[TOOL_NAME]({
            "model": "eyes", "state": "{{image:door}}",
            "images": [{"name": "door", "path": "door.png"}],
            "questions": [{"id": "here", "type": "noul",
                           "instructions": "Is there a person in {{image:door}}?"}]})
    finally:
        reset_workspace_root(token)
    assert not isinstance(result, tuple), result
    (_path, _headers, body), = standin.posts()
    (asset,) = body["assets"].values()
    assert base64.b64decode(asset["data_base64"]) == data
