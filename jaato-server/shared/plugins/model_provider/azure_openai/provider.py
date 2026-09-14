"""Azure OpenAI model provider.

Azure serves OpenAI's models on the same request and response shapes, so
the transport is the shared :class:`OpenAICompatProvider` unchanged.  What
is Azure's own is everything *around* the request, and each piece exists
because a corporate deployment cannot work without it:

**Deployment-name routing.**  A request is addressed to
``/openai/deployments/<deployment>/chat/completions``, where
``<deployment>`` is a name the subscription chose — it may or may not
resemble the model id behind it, and the same name can be repointed at a
new model version without changing.  jaato's ``model:`` field carries that
deployment name, and the SDK's ``AzureOpenAI`` client builds the URL from
it.  This is also why no per-model context table could be right here: the
key it would be looked up by does not identify a model.

**A pinned ``api-version``.**  Azure freezes its API surface per date, and
the date decides which request fields exist.  It is required and has no
default: a framework-chosen default would silently decide what a user's
deployment accepts, and would change under them when it moved.

**Two credential kinds.**  A resource key, or Microsoft Entra ID (bearer
tokens minted per request from a managed identity, a workload identity or
``az login``).  Entra is what most enterprises mandate, and it is the
reason "no key found" cannot be the end of the credential search here —
the correct configuration for many deployments has no key anywhere.

**Wire extensions are NOT assumed.**  PDFs (``file`` blocks) and audio
(``input_audio``) are gated on Azure by api-version *and* by what the
deployment is pointed at, so this provider declares ``pdf_input=False``
and ``audio_input=False`` and uses the shared images-only converter.  The
native ``openai`` provider declares both because its one endpoint carries
them unconditionally; here it would be a guess about somebody's resource,
and a guess in that direction is #829.

Environment variables:
    JAATO_AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_ENDPOINT: resource URL
    JAATO_AZURE_OPENAI_API_KEY / AZURE_OPENAI_API_KEY: resource key
    JAATO_AZURE_OPENAI_API_VERSION / AZURE_OPENAI_API_VERSION: api-version
    JAATO_AZURE_OPENAI_DEPLOYMENT / AZURE_OPENAI_DEPLOYMENT: default deployment
    JAATO_AZURE_OPENAI_CONTEXT_LENGTH: context window (required in practice)
    JAATO_AZURE_OPENAI_AUTH: "key" (default) or "aad"
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Set

from .._openai_compat.base import OpenAICompatProvider
from ..base import (
    MODALITY_TEXT,
    ProviderConfig,
    resolve_context_window,
    resolve_modalities,
)
from .env import (
    AUTH_AAD,
    AUTH_KEY,
    ENV_AZURE_API_KEY,
    ENV_AZURE_API_VERSION,
    ENV_AZURE_CONTEXT_LENGTH,
    ENV_AZURE_ENDPOINT,
    VALID_AUTH,
    ENV_VENDOR_API_KEY,
    get_checked_credential_locations,
    resolve_api_key,
    resolve_api_version,
    resolve_auth_method,
    resolve_context_length,
    resolve_deployment,
    resolve_endpoint,
)
from .errors import (
    APIKeyNotFoundError,
    AuthenticationError,
    ConfigurationError,
    ContextLimitError,
    InfrastructureError,
    ModelNotFoundError,
    RateLimitError,
)

#: INPUT modalities by the MODEL id a deployment is pointed at — used only
#: when a profile declares ``model_name``, because the deployment name
#: itself says nothing about what is behind it.  Without that declaration
#: the floor is text and vision is asserted through the ``modalities``
#: knob, which is the honest default on a surface where the addressable
#: name is chosen by whoever provisioned the resource.
MODEL_INPUT_MODALITIES: Dict[str, FrozenSet[str]] = {
    "gpt-4o": frozenset({"text", "image"}),
    "gpt-4.1": frozenset({"text", "image"}),
    "gpt-4-turbo": frozenset({"text", "image"}),
    "gpt-5": frozenset({"text", "image"}),
    "o1": frozenset({"text", "image"}),
    "o3": frozenset({"text", "image"}),
    "o4": frozenset({"text", "image"}),
}


class AzureOpenAIProvider(OpenAICompatProvider):
    """Azure OpenAI provider (deployment routing, api-version, key/Entra auth).

    Usage::

        provider = AzureOpenAIProvider()
        provider.initialize(ProviderConfig(extra={
            'endpoint': 'https://my-resource.openai.azure.com',
            'api_version': '2024-10-21',
            'context_length': 128000,
        }))
        provider.connect('my-gpt4o-deployment')   # the DEPLOYMENT name
        result = provider.complete(messages, system_instruction="Be helpful.")
    """

    # Parameterize the base's shared error mapping with Azure's taxonomy.
    _ERR_AUTHENTICATION = AuthenticationError
    _ERR_RATE_LIMIT = RateLimitError
    _ERR_MODEL_NOT_FOUND = ModelNotFoundError
    _ERR_CONTEXT_LIMIT = ContextLimitError
    _ERR_INFRASTRUCTURE = InfrastructureError

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        super().__init__()
        self._endpoint: Optional[str] = None
        self._api_version: Optional[str] = None
        self._auth_method: str = AUTH_KEY
        self._token_provider: Optional[Any] = None
        #: The model id behind the deployment, when a profile says so.
        self._model_name_hint: Optional[str] = None
        self._context_length_knob: Optional[int] = None
        self._modalities_knob: Optional[List[str]] = None

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "azure_openai"

    @property
    def auth_method(self) -> str:
        """Which credential kind this session uses: ``key`` or ``aad``."""
        return self._auth_method

    # ==================== Credential / config hooks ====================

    def _resolve_credentials(self, config: ProviderConfig) -> None:
        """Resolve endpoint, api-version and the credential; validate all three.

        Every failure here is raised at ``initialize()`` rather than left
        to the first request, because each one produces a misleading error
        later: a missing endpoint becomes a connection failure to a URL
        nobody meant to build, and a missing api-version becomes a 404 on
        a deployment that exists.
        """
        extra = config.extra or {}
        self._endpoint = extra.get("endpoint") or resolve_endpoint()
        if not self._endpoint:
            raise ConfigurationError(
                "endpoint",
                "Set plugin_configs.azure_openai.endpoint, or "
                f"{ENV_AZURE_ENDPOINT}, to your resource URL "
                "(https://<resource>.openai.azure.com — the 'Keys and "
                "Endpoint' blade in the Azure portal).",
            )
        self._api_version = extra.get("api_version") or resolve_api_version()
        if not self._api_version:
            raise ConfigurationError(
                "api_version",
                "Azure pins its API surface per date and jaato does not "
                "pick one for you — a default would decide which request "
                "fields your deployment accepts, and would change under "
                "you when it moved.  Set "
                "plugin_configs.azure_openai.api_version, or "
                f"{ENV_AZURE_API_VERSION} (e.g. '2024-10-21').",
            )
        # ``_base_url`` is what the base traces and reports; the SDK
        # client is built from ``_endpoint`` + ``_api_version``.
        self._base_url = self._endpoint

        self._auth_method = self._resolve_auth_method(config)
        if self._auth_method == AUTH_AAD:
            self._api_key = None
            return

        self._api_key = config.api_key or extra.get("api_key") or resolve_api_key(
            workspace_path=extra.get("workspace_path"),
            config_root=extra.get("config_root"),
        )
        if not self._api_key:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(config=config),
            )

    def _resolve_auth_method(self, config: ProviderConfig) -> str:
        """Resolve the credential kind: profile knob → env → ``key``.

        Raises:
            ValueError: for an unrecognised value.  Falling back to key
                auth would answer a profile that asked for Entra with a
                session that looks for a secret and reports it missing —
                a confusing error about the wrong thing.
        """
        raw = (config.extra or {}).get("auth") or resolve_auth_method() or AUTH_KEY
        method = str(raw).strip().lower()
        if method not in VALID_AUTH:
            raise ValueError(
                f"azure_openai 'auth' must be one of {list(VALID_AUTH)}, got "
                f"{raw!r}.  Set plugin_configs.azure_openai.auth (or "
                f"JAATO_AZURE_OPENAI_AUTH)."
            )
        return method

    def _resolve_context(self, config: ProviderConfig) -> None:
        """Stash the manual tiers; the window is resolved at ``connect()``.

        Also picks up ``model_name`` — the model id behind the deployment
        — which is the only thing that can make the modality table
        applicable, since the deployment name is arbitrary.
        """
        extra = config.extra or {}
        self._context_length_knob = (
            extra.get("context_length") or resolve_context_length()
        )
        self._model_name_hint = extra.get("model_name")
        modalities_override = extra.get("modalities")
        if modalities_override is not None:
            if not isinstance(modalities_override, (list, tuple)) or not all(
                isinstance(m, str) for m in modalities_override
            ):
                raise TypeError(
                    "azure_openai 'modalities' config must be a list of "
                    f"strings (e.g. [\"text\", \"image\"]), got "
                    f"{type(modalities_override).__name__}"
                )
            self._modalities_knob = list(modalities_override)

    def _create_client(self) -> Any:
        """Create the ``AzureOpenAI`` client for this resource.

        Uses the SDK's Azure client rather than a base-URL'd ``OpenAI``
        one: it is what knows to build ``/openai/deployments/<name>/...``
        from the ``model`` argument and to attach ``api-version`` to every
        request, and on the Entra path it is what re-invokes the token
        provider as tokens expire — a header set once would work until the
        first hour boundary and then 401 for the rest of the session.
        """
        from .._openai_compat._lazy import get_openai_module

        azure_client_class = get_openai_module().AzureOpenAI
        kwargs: Dict[str, Any] = {
            "azure_endpoint": self._endpoint,
            "api_version": self._api_version,
        }
        if self._auth_method == AUTH_AAD:
            kwargs["azure_ad_token_provider"] = self._build_token_provider()
        else:
            kwargs["api_key"] = self._api_key
        return azure_client_class(**kwargs)

    def _build_token_provider(self) -> Any:
        """The Entra bearer-token provider, or a clear ImportError.

        Raises:
            ConfigurationError: If ``azure-identity`` is not installed.
                The SDK's own failure here is an obscure attribute error;
                this one names the extra to install.
        """
        from .auth import azure_identity_available, build_token_provider

        if not azure_identity_available():
            raise ConfigurationError(
                "azure-identity",
                "plugin_configs.azure_openai.auth is 'aad', which mints a "
                "bearer token per request from the host's Azure identity, "
                "but the 'azure-identity' package is not installed.  "
                "Install it with: pip install 'jaato-server[azure-openai]'.",
            )
        if self._token_provider is None:
            self._token_provider = build_token_provider()
        return self._token_provider

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Set the active DEPLOYMENT and resolve its context window.

        ``model`` is the deployment name — the SDK turns it into the URL
        path.  When no deployment is named, the
        ``JAATO_AZURE_OPENAI_DEPLOYMENT`` default fills in, because a
        profile shared across resources may legitimately leave it to the
        environment.
        """
        self._model_name = model or resolve_deployment() or ""
        if not self._model_name:
            raise ConfigurationError(
                "deployment",
                "No deployment name was given.  On Azure the `model:` field "
                "carries the DEPLOYMENT name your subscription chose, not "
                "the model id.  Set it in the profile, or "
                "JAATO_AZURE_OPENAI_DEPLOYMENT in the environment.",
            )
        self._context_length = resolve_context_window(
            profile_value=self._context_length_knob,
        ) or 0
        if not self._context_length:
            raise ValueError(
                f"Azure OpenAI provider: context_length is not configured for "
                f"deployment {self._model_name!r}.  Azure's data plane lists "
                f"deployments, not their capacities, and a deployment name "
                f"does not identify the model version behind it — the same "
                f"name can be repointed at a model with a different window — "
                f"so it cannot be inferred either.  Set "
                f"plugin_configs.azure_openai.context_length in the profile, "
                f"or {ENV_AZURE_CONTEXT_LENGTH} in the environment.  No "
                f"hardcoded fallback exists per the project's no-fallback "
                f"rule."
            )
        self._trace(
            f"[CONNECT] deployment={self._model_name} "
            f"api_version={self._api_version} auth={self._auth_method} "
            f"context_length={self._context_length}"
        )

    # ==================== Catalog / capabilities ====================

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List what the resource's data plane reports, sorted.

        Returns an empty list when the listing is unavailable (an
        api-version that predates it, or a credential without the role for
        it) rather than raising: a failed catalog must not be the reason a
        working session cannot start.

        Note this lists MODELS available to the resource, which is not the
        same set as the DEPLOYMENTS you can address — deployments live on
        the control plane.  It is still the useful answer to "what is this
        resource for", and the deployment names are in the portal.
        """
        if self._client is None:
            return []
        try:
            listing = self._client.models.list()
        except Exception as exc:
            self._trace(f"[CATALOG] fetch failed: {type(exc).__name__}: {exc}")
            return []
        ids = [
            entry.id for entry in listing
            if getattr(entry, "id", None)
        ]
        if prefix:
            ids = [m for m in ids if m.startswith(prefix)]
        return sorted(ids)

    def _lookup_input_modalities(
        self, model: Optional[str] = None
    ) -> Optional[FrozenSet[str]]:
        """Table lookup, keyed on the declared model id — not the deployment.

        Returns ``None`` unless a profile declared ``model_name``, because
        matching the table against a deployment name would be reading
        meaning into a string the subscription owner was free to choose.
        """
        candidate = model or self._model_name_hint
        if not candidate:
            return None
        if candidate in MODEL_INPUT_MODALITIES:
            return MODEL_INPUT_MODALITIES[candidate]
        matches = [p for p in MODEL_INPUT_MODALITIES if candidate.startswith(p)]
        if not matches:
            return None
        return MODEL_INPUT_MODALITIES[max(matches, key=len)]

    def modalities(self, model: Optional[str] = None) -> Set[str]:
        """INPUT modalities this deployment accepts.

        ``modalities`` knob → the table, keyed on a declared ``model_name``
        → text floor.  A vision deployment therefore needs one of the two
        declarations; there is nothing on this wire to detect it from.
        """
        resolved = resolve_modalities(
            profile_value=self._modalities_knob,
            table_value=self._lookup_input_modalities(model),
        )
        return resolved if resolved is not None else {MODALITY_TEXT}

    # ==================== Auth introspection ====================

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None,
        config: Optional["ProviderConfig"] = None,
    ) -> bool:
        """Verify that a credential is configured, before ``initialize()``.

        Must not touch ``self._client``.  The Entra branch reports
        configured-ness rather than probing: acquiring a token from the
        credential chain can involve an IMDS round trip or an interactive
        browser flow, neither of which belongs in a pre-flight check —
        a credential that cannot mint a token fails loudly at the first
        request instead.

        Raises:
            APIKeyNotFoundError: On the key path, when no key is found
                (unless ``allow_interactive``).
        """
        from .auth import azure_identity_available

        extra = (config.extra if config is not None else None) or {}
        method = (
            str(extra.get("auth") or resolve_auth_method() or AUTH_KEY)
            .strip().lower()
        )
        if method == AUTH_AAD:
            return self._verify_entra(on_message, azure_identity_available())

        if _key_is_configured(config, extra, on_message):
            return True

        if not allow_interactive:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(config=config)
            )
        return False

    @staticmethod
    def _verify_entra(on_message, identity_available: bool) -> bool:
        """Report whether the Entra path is usable, without minting a token."""
        if identity_available:
            if on_message:
                on_message(
                    "Azure OpenAI configured for Microsoft Entra ID "
                    "(no stored key needed)"
                )
            return True
        if on_message:
            on_message(
                "auth is 'aad' but the 'azure-identity' package is not "
                "installed: pip install 'jaato-server[azure-openai]'"
            )
        return False

    def get_auth_info(self) -> str:
        """Return a short description of the credential source used."""
        import os

        if self._auth_method == AUTH_AAD:
            return f"Microsoft Entra ID ({self._endpoint})"

        for var in (ENV_AZURE_API_KEY, ENV_VENDOR_API_KEY):
            if os.environ.get(var):
                return f"Azure OpenAI key ({var})"

        try:
            from .auth import get_credential_file_path
            extra = getattr(self._config, "extra", None) or {}
            cred_path = get_credential_file_path(
                workspace_path=extra.get("workspace_path"),
                config_root=extra.get("config_root"),
            )
            if cred_path:
                return f"Azure OpenAI key ({cred_path})"
        except ImportError:
            pass

        return "Azure OpenAI key"


def _say(on_message, text: str) -> None:
    """Report a credential-search step, when the caller wants to hear it."""
    if on_message:
        on_message(text)


def _key_is_configured(config, extra, on_message) -> bool:
    """Whether a resource key is reachable, naming where it came from.

    Profile knob (which the daemon may have expanded from a ``pass://``
    URI) → either env spelling → the stored credential file.  A file that
    exists and will not load says so: "present but corrupt" and "not
    configured" have different fixes.
    """
    import os
    from .auth import try_load_credentials_with_reason

    profile_key = (config.api_key if config is not None else None) or \
        extra.get("api_key")
    if profile_key:
        _say(on_message, "Found Azure OpenAI key (profile config)")
        return True

    for var in (ENV_AZURE_API_KEY, ENV_VENDOR_API_KEY):
        if os.environ.get(var):
            _say(on_message, f"Found Azure OpenAI key ({var})")
            return True

    credentials, load_error = try_load_credentials_with_reason()
    if credentials and credentials.api_key:
        _say(on_message, "Found Azure OpenAI key (stored credentials)")
        return True
    if load_error:
        _say(on_message,
             f"Azure OpenAI credentials file found but could not be "
             f"loaded: {load_error}")
    return False


def create_provider() -> AzureOpenAIProvider:
    """Factory function for plugin discovery.

    Returns:
        A new AzureOpenAIProvider instance.
    """
    return AzureOpenAIProvider()
