"""Amazon Bedrock model provider plugin.

The third of the direct integrations #508 asked for, and the only one of them
that is not an OpenAI wire in disguise: Bedrock serves Anthropic, Amazon Nova,
Meta, Mistral, Cohere, AI21 and DeepSeek behind ONE message-shaped API
(``Converse`` / ``ConverseStream``), inside the customer's own AWS account,
authenticated with SigV4.

What that buys a deployment that already reaches these models elsewhere:

- **the data stays in the account** — a VPC endpoint, the account's own
  CloudTrail and Bedrock invocation logs, the account's own guardrails;
- **one credential for every vendor** — the IAM identity the host already
  has (an instance role, an EKS pod identity, an SSO session), so a
  confined runner needs no provider key in its environment at all;
- **committed capacity** — provisioned throughput and cross-region
  inference profiles are addressed as ordinary model ids.

**jaato resolves no credential here**, which is the structural difference
from every other provider in this tree.  SigV4 signing belongs to botocore,
and so does the chain behind it; :meth:`BedrockProvider.initialize` builds a
``boto3.Session`` and asks it what it found.  That is why
``PROVIDER_AUTH_RESOLUTION`` below names boto3's sources rather than jaato's,
and why no ``api_key`` knob exists.

``plugin_configs.bedrock.context_length`` is required in practice: Bedrock's
catalog reports modalities and streaming support, never capacity, so
``connect()`` fails loud rather than guessing a window that would silently
truncate a session.

``pip install 'jaato-server[bedrock]'``.
"""

from .provider import BedrockProvider, create_provider

__all__ = ["BedrockProvider", "create_provider"]


# --- Provider capability contract (see docs/model-provider-capabilities.md) ---
from ..base import (  # noqa: E402
    ProviderCapabilities, ProviderKnobs, KnobLayer, KnobSpec, AuthSource,
)

PROVIDER_CAPABILITIES = ProviderCapabilities(
    # Converse's own content blocks, each with a closed format vocabulary:
    # ``image`` (png/jpeg/gif/webp), ``document`` (pdf + office + plain
    # text), ``audio``.  A mime outside a vocabulary is withheld with a
    # note, never relabelled into one that is inside it (#829).
    user_message_images=True,
    tool_result_images=True,
    pdf_input=True,
    audio_input=True,
    # ``toolConfig.toolChoice`` — auto / any / a named tool, the name
    # routed through the wire-id mapper because the tools array is hashed.
    tool_choice_forwarding=True,
    # EXTRACTION only: ``reasoningContent`` blocks are read out and reach
    # the UI.  Requesting reasoning is available behind
    # ``api_params.enable_thinking``, which warns at connect — see
    # reasoning_replay below for why it is not the default.
    thinking=True,
    # ``cachePoint`` blocks on the system array and the tool list, behind
    # ``enable_caching`` / ``cache_ttl``.  Bedrock's own spelling of
    # Anthropic's ``cache_control``, and configurable, so this column has
    # something to deliver.
    prompt_caching=True,
    streaming=True,
    cancellation=True,
    # Converse returns no model-generated media on this path.
    output_media=False,
    # NOT declared, and the reason is a framework gap rather than a wire
    # one: Converse's ``reasoningContent`` block carries the upstream's
    # ``signature`` beside the text and the Anthropic-family models
    # REQUIRE it back on the next request of a tool-call loop, while
    # ``Part.thought`` has nowhere to put a signature.  Replaying the text
    # alone would be rejected by the very models that produced it, so the
    # provider reads reasoning out and does not write it back.  Wiring
    # this properly needs a signature-carrying part.
    reasoning_replay=False,
)

# --- Provider config-knob contract (authored from provider.py read sites) ---
PROVIDER_KNOBS = ProviderKnobs(layers=(
    KnobLayer("top_level", (
        KnobSpec("region", "str", None,
                 "AWS region, e.g. us-east-1.  Bedrock is regional and model "
                 "availability differs by region, so there is no default.  "
                 "Unset falls through to JAATO_BEDROCK_REGION, then AWS_REGION "
                 "(read here because Python's botocore does not), then boto3's "
                 "own AWS_DEFAULT_REGION / profile region"),
        KnobSpec("profile", "str", None,
                 "named AWS profile to build the boto3 session from — the "
                 "per-session override of boto3's AWS_PROFILE, so two "
                 "sessions on one host can use two AWS accounts"),
        KnobSpec("endpoint_url", "str", None,
                 "bedrock-runtime endpoint override (a VPC endpoint, or a "
                 "local stand-in in tests)"),
        KnobSpec("context_length", "int", None,
                 "REQUIRED — Bedrock's catalog reports no per-model context "
                 "window, so an operator assertion is the only source of "
                 "truth and connect() fails loud without one"),
        KnobSpec("modalities", "list", None,
                 "assert INPUT modalities, e.g. [\"text\",\"image\",\"file\"].  "
                 "Layered over the per-family table; needed for a model the "
                 "table does not name, since Bedrock's catalog vocabulary "
                 "(TEXT|IMAGE|EMBEDDING) cannot express documents or audio"),
        KnobSpec("output_modalities", "list", None,
                 "assert what the model can EMIT; no catalog reports output "
                 "modalities, so without this the floor is text and the "
                 "startup check refuses an outbound tier role"),
        KnobSpec("enable_caching", "bool", False,
                 "place Converse cachePoint blocks after the system prompt "
                 "and the tool list — Bedrock's spelling of Anthropic's "
                 "cache_control breakpoints"),
        KnobSpec("cache_ttl", "str", None,
                 "\"5m\" or \"1h\"; anything else is dropped rather than "
                 "forwarded, because Bedrock rejects values outside the enum"),
        KnobSpec("additional_model_request_fields", "dict", None,
                 "Converse's own escape hatch for vendor-specific inference "
                 "parameters.  Applied LAST, so it overrides anything the "
                 "framework put there (the thinking config included)"),
        KnobSpec("guardrail_config", "dict", None,
                 "guardrailIdentifier / guardrailVersion / trace, forwarded "
                 "verbatim to Converse"),
        KnobSpec("performance_latency", "str", None,
                 "\"standard\" or \"optimized\" — Bedrock's performanceConfig"),
    ), description="region / credential routing / wire extensions"),
    KnobLayer("api_params", (
        KnobSpec("max_tokens", "int", 8192,
                 "output cap; always sent, because Converse has no "
                 "server-side default worth relying on across seven vendors"),
        KnobSpec("temperature", "float"),
        KnobSpec("top_p", "float"),
        KnobSpec("stop_sequences", "list"),
        KnobSpec("tool_choice", "str", None,
                 "auto / required (Converse's \"any\") / none, or a dict "
                 "naming one tool; \"none\" drops the whole toolConfig, "
                 "which is how this wire says it"),
        KnobSpec("enable_thinking", "bool", False,
                 "request extended reasoning.  WARNS at connect: reasoning "
                 "is extracted but not replayed, so a thinking turn that "
                 "also calls tools may be rejected"),
        KnobSpec("thinking_budget", "int", 4096,
                 "reasoning token budget when enable_thinking is on"),
    ), description="Converse inferenceConfig (filtered allow-list)"),
))

# No quirks: the quirk vocabulary in this tree is OpenAI-compat machinery
# (prose-emulated tool calls, small-model tool-choice workarounds) and none
# of it applies to a wire that is not OpenAI's.
PROVIDER_QUIRKS = frozenset()

# --- Provider credential-resolution contract ---
#
# Every entry here is BOTO3's, not jaato's: the provider builds a
# ``boto3.Session`` and reads ``get_credentials()``.  Listing the chain is
# the point — an operator whose instance role failed needs to know an
# instance role was one of the things tried.
PROVIDER_AUTH_RESOLUTION = (
    AuthSource("env", "AWS_ACCESS_KEY_ID",
               "with AWS_SECRET_ACCESS_KEY (+ AWS_SESSION_TOKEN); read by "
               "boto3, never by jaato — which is why it is absent from "
               "jaato's env catalog"),
    AuthSource("stored", "~/.aws/credentials",
               "and ~/.aws/config; the profile is selected by "
               "plugin_configs.bedrock.profile, JAATO_BEDROCK_PROFILE or "
               "boto3's own AWS_PROFILE"),
    AuthSource("cli", "aws sso login",
               "IAM Identity Center cached credentials; the interactive "
               "login is AWS's own and runs outside this process, which is "
               "why verify_auth() offers none"),
    AuthSource("adc", "container and instance roles",
               "ECS / EKS task role, then EC2 instance metadata — the "
               "deployment shape that needs no jaato credential config at "
               "all beyond the region"),
)
