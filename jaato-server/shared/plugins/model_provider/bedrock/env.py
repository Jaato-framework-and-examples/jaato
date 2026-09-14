"""Environment resolution for the Bedrock provider.

Deliberately short, and the reason is the interesting part: **jaato does not
resolve an AWS credential.**  SigV4 signing belongs to botocore, which has a
documented resolution chain (env vars, a named profile, IAM Identity Center,
container and instance roles) that a framework re-implementing it could only
get subtly wrong.  So ``AWS_ACCESS_KEY_ID`` and friends are read by boto3,
never by this module -- which is why they do not appear in jaato's env
catalog: nothing in this tree reads them.

What IS jaato's to resolve is where the request goes and how the session
should size it: the region, the model, the context window, and the two
routing knobs (a named profile, an endpoint override).  Each is mirrored by a
typed ``plugin_configs.bedrock`` knob that outranks it.

The one vendor variable this module DOES read is ``AWS_REGION``, and that is
a botocore gap rather than a preference -- see :func:`resolve_region`.  It is
not a credential, so the paragraph above still holds exactly.

Every read goes through ``get_session_env`` rather than ``os.environ``.  The
daemon overlays each session's ``env:`` map onto its own ``os.environ`` for
the duration of a turn, so on a daemon serving two tenants a plain read can
return the OTHER session's value -- non-deterministically and with no error.
For ``profile`` that is the wrong AWS account signing the request."""

from typing import List, Optional

from shared.session_context import get_session_env


def resolve_region() -> Optional[str]:
    """Region from jaato's own variable, else from the vendor's.

    ``AWS_DEFAULT_REGION`` is deliberately NOT read here: botocore honours it
    itself, so reading it would only put jaato's precedence in front of the
    AWS chain for no gain.

    ``AWS_REGION`` is read, and that asymmetry is not an oversight.  It is
    the variable the AWS documentation names first and every other AWS SDK
    honours, but Python's botocore session-variable table maps ``region`` to
    ``AWS_DEFAULT_REGION`` **alone** -- so a host configured the documented
    way, on which the AWS CLI works, resolves no region at all through boto3.
    Reading it here is what stops that becoming "no region configured" on a
    machine the operator has every reason to think is configured.
    """
    return (
        get_session_env("JAATO_BEDROCK_REGION")  # env: AWS region for Bedrock; outranks the vendor's own variables
        or get_session_env("AWS_REGION")  # env: the vendor's documented region variable, which Python's botocore does NOT itself honour (it reads AWS_DEFAULT_REGION only)
        or None
    )


def resolve_model() -> Optional[str]:
    """Default Bedrock model id / inference-profile id."""
    return get_session_env("JAATO_BEDROCK_MODEL") or None  # env: default Bedrock model or inference-profile id


def resolve_context_length() -> Optional[int]:
    """Context window override, as an int, or ``None`` when unset/unparsable.

    An unparsable value returns ``None`` rather than raising, so the caller
    falls through to the "not configured" error that names every way to set
    it -- one message beats two.
    """
    raw = get_session_env("JAATO_BEDROCK_CONTEXT_LENGTH")  # env: context window for the Bedrock model; required, the catalog reports none
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def resolve_profile() -> Optional[str]:
    """Named AWS profile to build the boto3 session from.

    Distinct from boto3's own ``AWS_PROFILE``, which stays in force when this
    is unset: this is the per-session override, so two sessions on one host
    can use two different AWS accounts.
    """
    return get_session_env("JAATO_BEDROCK_PROFILE") or None  # env: named AWS profile for the Bedrock session; boto3's AWS_PROFILE applies when unset


def resolve_endpoint_url() -> Optional[str]:
    """Endpoint override -- a VPC endpoint, or a local stand-in for tests."""
    return get_session_env("JAATO_BEDROCK_ENDPOINT_URL") or None  # env: bedrock-runtime endpoint override (VPC endpoint or a local stand-in)


def get_checked_credential_locations() -> List[str]:
    """The chain a "credentials not found" message should name.

    These are boto3's sources, not jaato's, and listing them is the whole
    point: an operator whose instance role failed needs to know that an
    instance role was one of the things tried.
    """
    return [
        "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (+ AWS_SESSION_TOKEN)",
        "the named AWS profile (plugin_configs.bedrock.profile, "
        "JAATO_BEDROCK_PROFILE, or AWS_PROFILE)",
        "~/.aws/credentials and ~/.aws/config",
        "IAM Identity Center / SSO cached credentials",
        "container credentials (ECS / EKS task role)",
        "instance metadata (EC2 instance role)",
    ]
