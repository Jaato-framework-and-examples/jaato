"""The one path the stubbed tests cannot cover: real boto3 session wiring.

Everything else about this provider is testable with an injected client, but
the whole point of the credential story is that jaato does NOT resolve a
credential — it builds a ``boto3.Session`` and asks it what it found.  That
claim is only true if the session is built with the right arguments and its
answer is actually consulted, which needs the real library.

No network and no account: ``get_credentials()`` and ``client()`` are local,
and the fake key material below never leaves the process.
"""

import pytest

boto3 = pytest.importorskip("boto3")

from shared.plugins.model_provider.base import ProviderConfig  # noqa: E402

from ..errors import (  # noqa: E402
    CredentialsNotFoundError,
    RegionNotConfiguredError,
)
from ..provider import BedrockProvider  # noqa: E402


@pytest.fixture(autouse=True)
def _no_ambient_aws(monkeypatch, tmp_path):
    """Isolate from the developer's own AWS environment.

    Without this the suite passes on a laptop with ``~/.aws/credentials`` and
    fails in CI, or worse signs something with a real identity.
    """
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_REGION",
                "AWS_DEFAULT_REGION", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                "JAATO_BEDROCK_REGION", "JAATO_BEDROCK_PROFILE",
                "JAATO_BEDROCK_CONTEXT_LENGTH", "JAATO_BEDROCK_MODEL",
                "JAATO_BEDROCK_ENDPOINT_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "creds"))
    # The EC2 metadata probe would otherwise reach for 169.254.169.254 and
    # spend its timeout budget on every case here.
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def _fake_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/EXAMPLEKEY")


def test_initialize_builds_a_runtime_client_from_the_resolved_session(monkeypatch):
    _fake_credentials(monkeypatch)
    provider = BedrockProvider()
    provider.initialize(ProviderConfig(extra={"region": "eu-west-1",
                                              "context_length": 200_000}))
    assert provider._region == "eu-west-1"
    assert provider._client.meta.service_model.service_name == "bedrock-runtime"


def test_the_credential_source_boto3_found_is_reported(monkeypatch):
    """"Connected to" names the source, because "why does this work on my
    laptop and not in the container" is the question that line answers."""
    _fake_credentials(monkeypatch)
    provider = BedrockProvider()
    provider.initialize(ProviderConfig(extra={"region": "us-east-1",
                                              "context_length": 200_000}))
    assert "env" in provider.get_auth_info()


def test_no_credentials_anywhere_names_the_whole_chain(monkeypatch):
    provider = BedrockProvider()
    with pytest.raises(CredentialsNotFoundError) as exc:
        provider.initialize(ProviderConfig(extra={"region": "us-east-1"}))
    message = str(exc.value)
    assert "instance metadata" in message
    assert "aws sso login" in message


def test_no_region_anywhere_is_refused_before_any_credential_work(monkeypatch):
    """Bedrock is regional and availability differs by region, so a default
    would be a confusing 404 on a model that plainly exists."""
    _fake_credentials(monkeypatch)
    provider = BedrockProvider()
    with pytest.raises(RegionNotConfiguredError):
        provider.initialize(ProviderConfig())


def test_boto3s_own_region_variable_still_works(monkeypatch):
    """jaato asks boto3 LAST rather than not at all — which is what lets an
    EC2/EKS host need no jaato configuration."""
    _fake_credentials(monkeypatch)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-2")
    provider = BedrockProvider()
    provider.initialize(ProviderConfig(extra={"context_length": 200_000}))
    assert provider._region == "ap-southeast-2"


def test_the_vendors_documented_region_variable_works_too(monkeypatch):
    """``AWS_REGION`` is what the AWS docs name first and what every other AWS
    SDK honours, and Python's botocore maps ``region`` to ``AWS_DEFAULT_REGION``
    ALONE — so without jaato reading it, a host configured the documented way
    reports "no region configured" while the AWS CLI on it works fine."""
    _fake_credentials(monkeypatch)
    monkeypatch.setenv("AWS_REGION", "ca-central-1")
    assert boto3.Session().region_name is None, (
        "botocore now honours AWS_REGION itself — drop the read in "
        "bedrock/env.py:resolve_region and this case with it"
    )
    provider = BedrockProvider()
    provider.initialize(ProviderConfig(extra={"context_length": 200_000}))
    assert provider._region == "ca-central-1"


def test_the_knob_outranks_boto3s_variable(monkeypatch):
    _fake_credentials(monkeypatch)
    monkeypatch.setenv("AWS_REGION", "ap-southeast-2")
    provider = BedrockProvider()
    provider.initialize(ProviderConfig(extra={"region": "us-east-1",
                                              "context_length": 200_000}))
    assert provider._region == "us-east-1"


def test_jaatos_own_region_variable_is_read(monkeypatch):
    _fake_credentials(monkeypatch)
    monkeypatch.setenv("JAATO_BEDROCK_REGION", "us-west-2")
    provider = BedrockProvider()
    provider.initialize(ProviderConfig(extra={"context_length": 200_000}))
    assert provider._region == "us-west-2"


def test_an_endpoint_override_reaches_the_client(monkeypatch):
    """A VPC endpoint is the deployment shape this exists for."""
    _fake_credentials(monkeypatch)
    provider = BedrockProvider()
    provider.initialize(ProviderConfig(extra={
        "region": "us-east-1", "context_length": 200_000,
        "endpoint_url": "https://vpce-abc.bedrock-runtime.us-east-1.vpce.amazonaws.com",
    }))
    assert "vpce-abc" in provider._client.meta.endpoint_url


def test_verify_auth_answers_before_initialize_and_without_a_client(monkeypatch):
    """The runtime calls verify_auth() on a FRESH provider instance, so it
    must not touch anything initialize() sets."""
    _fake_credentials(monkeypatch)
    provider = BedrockProvider()
    assert provider.verify_auth(config=ProviderConfig(
        extra={"region": "us-east-1"})) is True
    assert provider._client is None


def test_verify_auth_raises_when_the_chain_is_empty(monkeypatch):
    provider = BedrockProvider()
    with pytest.raises(CredentialsNotFoundError):
        provider.verify_auth(config=ProviderConfig(extra={"region": "us-east-1"}))


def test_shutdown_drops_the_pooled_connections(monkeypatch):
    _fake_credentials(monkeypatch)
    provider = BedrockProvider()
    provider.initialize(ProviderConfig(extra={"region": "us-east-1",
                                              "context_length": 200_000}))
    provider.shutdown()
    assert provider._client is None
    assert provider.is_connected is False


def test_connect_warns_that_thinking_is_not_replayed(monkeypatch, caplog):
    """A thinking turn that also calls tools may be rejected, and a silent
    knob is how that becomes a mystery 400."""
    _fake_credentials(monkeypatch)
    provider = BedrockProvider()
    provider.initialize(ProviderConfig(extra={
        "region": "us-east-1", "context_length": 200_000,
        "api_params": {"enable_thinking": True}}))
    with caplog.at_level("WARNING"):
        provider.connect("anthropic.claude-sonnet-4-5-20250929-v1:0")
    assert any("not replayed" in r.message or "signature" in r.message
               for r in caplog.records)
