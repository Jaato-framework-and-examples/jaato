"""Domain errors for the Bedrock provider.

botocore raises one exception class (``ClientError``) for every API failure
and puts the actual diagnosis in ``response["Error"]["Code"]``.  That is fine
for a script and useless for a retry layer: ``with_retry`` decides from the
exception TYPE, so a throttle and a bad model id arriving as the same class
means either everything is retried or nothing is.  These types are the
translation, and :func:`shared.plugins.model_provider.bedrock.provider`
performs it in one place.

The split that matters is transient vs not:

* :class:`ThrottlingError`, :class:`ServiceUnavailableError` and
  :class:`ModelNotReadyError` are RAISED out of ``complete()`` so the retry
  layer sees them;
* everything else becomes a ``TurnResult.from_exception`` -- a terminal
  answer, because retrying a 403 or a malformed request just spends the
  budget on the same failure.
"""

from typing import List, Optional


class BedrockProviderError(Exception):
    """Base exception for Bedrock provider errors."""


class CredentialsNotFoundError(BedrockProviderError):
    """No AWS credentials could be resolved for the Bedrock call.

    Unlike every API-key provider in this tree, jaato does not itself hold a
    Bedrock credential: SigV4 signing is boto3's, and boto3 has its own
    documented resolution chain (env vars, a named profile, SSO, container
    and instance roles).  So this error's job is to name that chain rather
    than to offer a key to paste.
    """

    def __init__(self, checked_locations: Optional[List[str]] = None,
                 region: Optional[str] = None):
        self.checked_locations = checked_locations or []
        self.region = region
        checked = "\n".join(f"    - {loc}" for loc in self.checked_locations)
        message = (
            "AWS credentials for Amazon Bedrock not found.\n"
            f"Checked (boto3's own resolution chain):\n{checked}\n\n"
            "To fix this, choose one option:\n\n"
            "  Option 1 - an AWS profile:\n"
            "    aws configure --profile my-profile\n"
            "    then set plugin_configs.bedrock.profile: my-profile\n\n"
            "  Option 2 - environment credentials:\n"
            "    export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...\n"
            "    (plus AWS_SESSION_TOKEN for temporary credentials)\n\n"
            "  Option 3 - IAM Identity Center (SSO):\n"
            "    aws sso login --profile my-profile\n\n"
            "  Option 4 - an instance/task/pod role: nothing to set; the\n"
            "    role is picked up automatically where one is attached.\n\n"
            "The identity also needs bedrock:InvokeModel and\n"
            "bedrock:InvokeModelWithResponseStream on the model you name"
            + (f", in region {region}." if region else ".")
        )
        super().__init__(message)


class RegionNotConfiguredError(BedrockProviderError):
    """No AWS region was resolved.

    Bedrock is regional and model availability differs by region, so there is
    no default worth guessing: a wrong region is a confusing 404 on a model
    that plainly exists.
    """

    def __init__(self) -> None:
        super().__init__(
            "No AWS region configured for Amazon Bedrock. Bedrock is regional "
            "and model availability differs by region, so there is no default.\n"
            "Set one of:\n"
            "    plugin_configs.bedrock.region: us-east-1\n"
            "    JAATO_BEDROCK_REGION=us-east-1\n"
            "    AWS_REGION=us-east-1 (the vendor's documented variable; jaato\n"
            "        reads it because Python's botocore does not)\n"
            "    AWS_DEFAULT_REGION=us-east-1 (the one botocore itself reads)\n"
            "    the `region` of the AWS profile you named"
        )


class AccessDeniedError(BedrockProviderError):
    """The credentials are valid but not entitled to this model.

    On Bedrock this is usually model ACCESS rather than IAM: a foundation
    model must be enabled for the account in the console before any identity
    can invoke it, and the failure looks identical to a policy problem.
    """

    def __init__(self, model: Optional[str] = None, region: Optional[str] = None,
                 original_error: Optional[str] = None):
        self.model = model
        self.region = region
        where = f" in {region}" if region else ""
        message = (
            f"Access denied invoking {model or 'the model'}{where}.\n\n"
            "Two different things produce this, and the second is the common one:\n"
            "  1. the IAM identity lacks bedrock:InvokeModel /\n"
            "     bedrock:InvokeModelWithResponseStream on the model;\n"
            "  2. the model is not ENABLED for this account in this region --\n"
            "     Bedrock console -> Model access -> request access.\n"
        )
        if original_error:
            message += f"\nOriginal error: {original_error}"
        super().__init__(message)


class ModelNotFoundError(BedrockProviderError):
    """The model id or inference-profile id does not resolve in this region."""

    def __init__(self, model: Optional[str] = None, region: Optional[str] = None,
                 original_error: Optional[str] = None):
        self.model = model
        self.region = region
        message = (
            f"Bedrock model {model!r} was not found"
            + (f" in region {region}" if region else "")
            + ".\n\nCheck that:\n"
            "  - the id is the Bedrock id, not the vendor's own\n"
            "    (e.g. anthropic.claude-sonnet-4-5-20250929-v1:0);\n"
            "  - a cross-region model is addressed through its inference\n"
            "    profile, which carries a region prefix (us. / eu. / apac.);\n"
            "  - the model is offered in this region at all."
        )
        if original_error:
            message += f"\nOriginal error: {original_error}"
        super().__init__(message)


class ThrottlingError(BedrockProviderError):
    """TRANSIENT. Account quota or per-model TPS limit exceeded."""

    def __init__(self, retry_after: Optional[float] = None,
                 original_error: Optional[str] = None):
        self.retry_after = retry_after
        message = "Bedrock throttled the request (account or per-model quota)."
        if retry_after:
            message += f" Retry after {retry_after} seconds."
        if original_error:
            message += f"\nOriginal error: {original_error}"
        super().__init__(message)


class ServiceUnavailableError(BedrockProviderError):
    """TRANSIENT. Bedrock or the upstream model is temporarily unavailable."""

    def __init__(self, original_error: Optional[str] = None):
        message = "Bedrock is temporarily unavailable."
        if original_error:
            message += f"\nOriginal error: {original_error}"
        super().__init__(message)


class ModelNotReadyError(BedrockProviderError):
    """TRANSIENT. A provisioned or imported model is still warming up."""

    def __init__(self, model: Optional[str] = None,
                 original_error: Optional[str] = None):
        message = (
            f"Bedrock model {model!r} is not ready to serve requests yet "
            "(a provisioned or imported model still warming up)."
        )
        if original_error:
            message += f"\nOriginal error: {original_error}"
        super().__init__(message)


class ContextLimitError(BedrockProviderError):
    """The request exceeded the model's input window."""

    def __init__(self, original_error: Optional[str] = None):
        message = (
            "Bedrock rejected the request as too long for the model's context "
            "window. Lower the GC threshold, or check that "
            "plugin_configs.bedrock.context_length matches the model."
        )
        if original_error:
            message += f"\nOriginal error: {original_error}"
        super().__init__(message)


class ContextLengthNotConfiguredError(BedrockProviderError):
    """No context window is known for the model, and none was configured.

    Bedrock's ``ListFoundationModels`` reports modalities and streaming
    support and NOT capacity, so there is nothing to detect: an operator
    assertion is the only source of truth available, and guessing one would
    silently truncate a session or overfill a request.
    """

    def __init__(self, model: Optional[str] = None):
        super().__init__(
            f"No context window configured for Bedrock model {model!r}.\n"
            "Bedrock's model catalog reports no per-model context length, so "
            "one must be declared:\n"
            "    plugin_configs.bedrock.context_length: 200000\n"
            "    or JAATO_BEDROCK_CONTEXT_LENGTH=200000\n"
            "No hardcoded fallback exists, per the project's no-fallback rule: "
            "a guessed window truncates a session without saying so."
        )


class BotocoreNotInstalledError(BedrockProviderError):
    """The optional ``boto3`` dependency is missing."""

    def __init__(self) -> None:
        super().__init__(
            "The Bedrock provider needs boto3 (which brings botocore's SigV4 "
            "signer).\n    pip install 'jaato-server[bedrock]'"
        )
