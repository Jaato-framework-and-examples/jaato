"""Environment variable resolution for Google GenAI provider.

This module handles loading configuration from environment variables,
following the hybrid naming convention:
- Provider standard vars for secrets (GOOGLE_GENAI_API_KEY, GOOGLE_APPLICATION_CREDENTIALS)
- JAATO-prefixed vars for framework-specific settings (JAATO_GOOGLE_AUTH_METHOD)

Resolution priority:
1. Explicit config passed in code
2. JAATO_GOOGLE_* environment variables
3. Provider standard environment variables
4. Defaults
"""

import os
from typing import Literal, Optional, Tuple

# Auth method type
AuthMethod = Literal["auto", "api_key", "service_account_file", "adc", "impersonation"]

# ============================================================
# Environment Variable Names
# ============================================================

# Framework-level
ENV_DEFAULT_PROVIDER = "JAATO_DEFAULT_PROVIDER"

# Google-specific (JAATO namespace)
ENV_GOOGLE_AUTH_METHOD = "JAATO_GOOGLE_AUTH_METHOD"
ENV_GOOGLE_USE_VERTEX = "JAATO_GOOGLE_USE_VERTEX"
ENV_GOOGLE_PROJECT = "JAATO_GOOGLE_PROJECT"
ENV_GOOGLE_LOCATION = "JAATO_GOOGLE_LOCATION"
ENV_GOOGLE_TARGET_SERVICE_ACCOUNT = "JAATO_GOOGLE_TARGET_SERVICE_ACCOUNT"

# Google standard (industry convention)
ENV_GOOGLE_API_KEY = "GOOGLE_GENAI_API_KEY"
ENV_GOOGLE_APP_CREDENTIALS = "GOOGLE_APPLICATION_CREDENTIALS"
ENV_GOOGLE_CLOUD_PROJECT = "GOOGLE_CLOUD_PROJECT"

# Legacy jaato vars (for backwards compatibility)
ENV_LEGACY_PROJECT_ID = "PROJECT_ID"
ENV_LEGACY_LOCATION = "LOCATION"
ENV_LEGACY_MODEL_NAME = "MODEL_NAME"


def _get_bool_env(name: str, default: bool = False) -> bool:
    """Get a boolean environment variable.

    Recognizes: 1, true, yes, on (case-insensitive) as True.
    """
    value = os.environ.get(name, "").lower()
    if not value:
        return default
    return value in ("1", "true", "yes", "on")


def _get_env_with_fallback(*names: str, default: Optional[str] = None) -> Optional[str]:
    """Get the first defined environment variable from a list of names."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def resolve_auth_method() -> AuthMethod:
    """Resolve the authentication method from environment.

    Priority:
    1. JAATO_GOOGLE_AUTH_METHOD if set
    2. Infer from available credentials:
       - GOOGLE_GENAI_API_KEY present → api_key
       - GOOGLE_APPLICATION_CREDENTIALS present → service_account_file
       - JAATO_GOOGLE_TARGET_SERVICE_ACCOUNT present → impersonation
       - Otherwise → auto (ADC)

    Returns:
        The resolved authentication method.
    """
    explicit = os.environ.get(ENV_GOOGLE_AUTH_METHOD, "").lower()
    if explicit in ("api_key", "service_account_file", "adc", "auto", "impersonation"):
        return explicit  # type: ignore

    # Infer from available credentials
    if os.environ.get(ENV_GOOGLE_API_KEY):
        return "api_key"
    if os.environ.get(ENV_GOOGLE_TARGET_SERVICE_ACCOUNT):
        return "impersonation"
    if os.environ.get(ENV_GOOGLE_APP_CREDENTIALS):
        return "service_account_file"

    return "auto"


def resolve_use_vertex() -> bool:
    """Resolve whether to use Vertex AI or AI Studio.

    Priority:
    1. JAATO_GOOGLE_USE_VERTEX if set
    2. Infer from auth method:
       - api_key → False (AI Studio)
       - Otherwise → True (Vertex AI)

    Returns:
        True for Vertex AI, False for AI Studio.
    """
    explicit = os.environ.get(ENV_GOOGLE_USE_VERTEX)
    if explicit is not None:
        return _get_bool_env(ENV_GOOGLE_USE_VERTEX, default=True)

    # Infer from auth method
    auth_method = resolve_auth_method()
    if auth_method == "api_key":
        return False

    return True


def resolve_api_key() -> Optional[str]:
    """Resolve API key from environment.

    Returns:
        API key if found, None otherwise.
    """
    return os.environ.get(ENV_GOOGLE_API_KEY)


def resolve_credentials_path() -> Optional[str]:
    """Resolve service account credentials path from environment.

    Returns:
        Path to credentials file if found, None otherwise.
    """
    return os.environ.get(ENV_GOOGLE_APP_CREDENTIALS)


def resolve_project() -> Optional[str]:
    """Resolve GCP project ID from environment.

    Priority:
    1. JAATO_GOOGLE_PROJECT
    2. GOOGLE_CLOUD_PROJECT
    3. PROJECT_ID (legacy)

    Returns:
        Project ID if found, None otherwise.
    """
    return _get_env_with_fallback(
        ENV_GOOGLE_PROJECT,
        ENV_GOOGLE_CLOUD_PROJECT,
        ENV_LEGACY_PROJECT_ID,
    )


def resolve_location() -> Optional[str]:
    """Resolve GCP location/region from environment.

    Priority:
    1. JAATO_GOOGLE_LOCATION
    2. LOCATION (legacy)

    Returns:
        Location if found, None otherwise.
    """
    return _get_env_with_fallback(
        ENV_GOOGLE_LOCATION,
        ENV_LEGACY_LOCATION,
    )


def resolve_target_service_account() -> Optional[str]:
    """Resolve target service account for impersonation from environment.

    Returns:
        Target service account email if found, None otherwise.
    """
    return os.environ.get(ENV_GOOGLE_TARGET_SERVICE_ACCOUNT)


def resolve_model_name() -> Optional[str]:
    """Resolve model name from environment.

    Returns:
        Model name if found, None otherwise.
    """
    return os.environ.get(ENV_LEGACY_MODEL_NAME)


def get_checked_credential_locations(auth_method: AuthMethod) -> list[str]:
    """Get list of locations checked for credentials.

    Used for error messages to help users understand what was checked.

    Args:
        auth_method: The authentication method being used.

    Returns:
        List of location descriptions.
    """
    locations = []

    if auth_method in ("api_key",):
        api_key = os.environ.get(ENV_GOOGLE_API_KEY)
        if api_key:
            locations.append(f"{ENV_GOOGLE_API_KEY}: set (length={len(api_key)})")
        else:
            locations.append(f"{ENV_GOOGLE_API_KEY}: not set")

    if auth_method in ("auto", "adc", "service_account_file", "impersonation"):
        creds_path = os.environ.get(ENV_GOOGLE_APP_CREDENTIALS)
        if creds_path:
            if os.path.exists(creds_path):
                locations.append(f"{ENV_GOOGLE_APP_CREDENTIALS}: {creds_path} (exists)")
            else:
                locations.append(f"{ENV_GOOGLE_APP_CREDENTIALS}: {creds_path} (NOT FOUND)")
        else:
            locations.append(f"{ENV_GOOGLE_APP_CREDENTIALS}: not set")

    if auth_method in ("auto", "adc", "impersonation"):
        # Check default ADC locations
        adc_path = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
        if os.path.exists(adc_path):
            locations.append(f"ADC user credentials: {adc_path} (exists)")
        else:
            locations.append(f"ADC user credentials: {adc_path} (not found)")

        # Note about metadata server
        locations.append("GCE metadata server: checked at runtime by SDK")

    if auth_method == "impersonation":
        target_sa = os.environ.get(ENV_GOOGLE_TARGET_SERVICE_ACCOUNT)
        if target_sa:
            locations.append(f"{ENV_GOOGLE_TARGET_SERVICE_ACCOUNT}: {target_sa}")
        else:
            locations.append(f"{ENV_GOOGLE_TARGET_SERVICE_ACCOUNT}: not set")

    return locations


def load_google_config_from_env() -> Tuple[
    AuthMethod,
    bool,
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
]:
    """Load all Google GenAI configuration from environment.

    Returns:
        Tuple of (auth_method, use_vertex, api_key, credentials_path, project, location, model_name, target_service_account)
    """
    return (
        resolve_auth_method(),
        resolve_use_vertex(),
        resolve_api_key(),
        resolve_credentials_path(),
        resolve_project(),
        resolve_location(),
        resolve_model_name(),
        resolve_target_service_account(),
    )
