"""Authentication and configuration error types for Google GenAI provider.

These exceptions wrap underlying SDK errors with actionable guidance
for users to resolve authentication and configuration issues.
"""

from typing import List, Optional


class JaatoAuthError(Exception):
    """Base class for authentication errors."""

    pass


class CredentialsNotFoundError(JaatoAuthError):
    """No valid credentials could be located.

    Raised when the provider cannot find any credentials through
    the configured authentication method.
    """

    def __init__(
        self,
        auth_method: str,
        checked_locations: Optional[List[str]] = None,
        suggestion: Optional[str] = None,
    ):
        self.auth_method = auth_method
        self.checked_locations = checked_locations or []
        self.suggestion = suggestion

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        lines = [
            f"No credentials found for authentication method: {self.auth_method}",
            "",
        ]

        if self.checked_locations:
            lines.append("Checked locations:")
            for loc in self.checked_locations:
                lines.append(f"  - {loc}")
            lines.append("")

        if self.suggestion:
            lines.append("To fix:")
            lines.append(f"  {self.suggestion}")
        else:
            lines.extend(self._default_suggestions())

        return "\n".join(lines)

    def _default_suggestions(self) -> List[str]:
        if self.auth_method == "api_key":
            return [
                "To fix:",
                "  1. Get an API key from https://aistudio.google.com/apikey",
                "  2. Set GOOGLE_GENAI_API_KEY=your-api-key",
            ]
        elif self.auth_method in ("auto", "adc"):
            return [
                "To fix, either:",
                "  1. Run: gcloud auth application-default login",
                "  2. Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json",
                "  3. Use API key: set GOOGLE_GENAI_API_KEY for AI Studio mode",
            ]
        elif self.auth_method == "service_account_file":
            return [
                "To fix:",
                "  Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json",
            ]
        return []


class CredentialsInvalidError(JaatoAuthError):
    """Credentials were found but are invalid or malformed.

    Raised when credentials exist but cannot be parsed or are
    rejected by the authentication system.
    """

    def __init__(
        self,
        auth_method: str,
        reason: str,
        credentials_source: Optional[str] = None,
    ):
        self.auth_method = auth_method
        self.reason = reason
        self.credentials_source = credentials_source

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        lines = [f"Invalid credentials for authentication method: {self.auth_method}"]

        if self.credentials_source:
            lines.append(f"Source: {self.credentials_source}")

        lines.append(f"Reason: {self.reason}")

        if self.auth_method == "api_key":
            lines.extend([
                "",
                "To fix:",
                "  Verify your API key is correct at https://aistudio.google.com/apikey",
            ])
        elif self.auth_method == "service_account_file":
            lines.extend([
                "",
                "To fix:",
                "  1. Verify the service account key file is valid JSON",
                "  2. Ensure the file contains 'type': 'service_account'",
                "  3. Check the service account has not been deleted in GCP",
            ])

        return "\n".join(lines)


class CredentialsPermissionError(JaatoAuthError):
    """Credentials lack required permissions.

    Raised when authentication succeeds but the credentials do not
    have sufficient permissions for the requested operation.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        service_account: Optional[str] = None,
        missing_role: Optional[str] = None,
        original_error: Optional[str] = None,
    ):
        self.project = project
        self.service_account = service_account
        self.missing_role = missing_role
        self.original_error = original_error

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        lines = ["Credentials lack required permissions for Vertex AI."]

        if self.project:
            lines.append(f"Project: {self.project}")
        if self.service_account:
            lines.append(f"Service Account: {self.service_account}")
        if self.missing_role:
            lines.append(f"Missing role: {self.missing_role}")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "To fix, grant the required role:",
        ])

        if self.project and self.service_account:
            lines.append(
                f"  gcloud projects add-iam-policy-binding {self.project} \\"
            )
            lines.append(
                f"    --member='serviceAccount:{self.service_account}' \\"
            )
            lines.append(
                f"    --role='roles/aiplatform.user'"
            )
        else:
            lines.append("  Grant 'roles/aiplatform.user' to your service account")

        return "\n".join(lines)


class ProjectConfigurationError(JaatoAuthError):
    """Project or location misconfigured for Vertex AI.

    Raised when the GCP project ID or location is missing, invalid,
    or not enabled for Vertex AI.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        location: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        self.project = project
        self.location = location
        self.reason = reason

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        lines = ["Invalid Vertex AI project configuration."]

        if self.project:
            lines.append(f"Project: {self.project}")
        else:
            lines.append("Project: not set")

        if self.location:
            lines.append(f"Location: {self.location}")
        else:
            lines.append("Location: not set")

        if self.reason:
            lines.append(f"Reason: {self.reason}")

        lines.extend([
            "",
            "To fix:",
        ])

        if not self.project:
            lines.append("  1. Set JAATO_GOOGLE_PROJECT or GOOGLE_CLOUD_PROJECT")
        if not self.location:
            lines.append("  2. Set JAATO_GOOGLE_LOCATION (e.g., 'us-central1')")

        lines.extend([
            "",
            "Ensure Vertex AI API is enabled:",
            "  gcloud services enable aiplatform.googleapis.com --project=YOUR_PROJECT",
        ])

        return "\n".join(lines)


class ImpersonationError(JaatoAuthError):
    """Service account impersonation failed.

    Raised when impersonation is configured but fails due to missing
    target service account, permission issues, or other impersonation errors.
    """

    def __init__(
        self,
        target_service_account: Optional[str] = None,
        source_principal: Optional[str] = None,
        reason: Optional[str] = None,
        original_error: Optional[str] = None,
    ):
        self.target_service_account = target_service_account
        self.source_principal = source_principal
        self.reason = reason
        self.original_error = original_error

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        if not self.target_service_account:
            # Missing target service account
            lines = [
                "Service account impersonation requires a target service account.",
                "",
                "To fix:",
                "  Set JAATO_GOOGLE_TARGET_SERVICE_ACCOUNT=your-sa@project.iam.gserviceaccount.com",
            ]
            return "\n".join(lines)

        lines = ["Service account impersonation failed."]
        lines.append(f"Target: {self.target_service_account}")

        if self.source_principal:
            lines.append(f"Source: {self.source_principal}")

        if self.reason:
            lines.append(f"Reason: {self.reason}")

        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "To fix, grant the Service Account Token Creator role:",
        ])

        if self.source_principal and self.target_service_account:
            lines.extend([
                f"  gcloud iam service-accounts add-iam-policy-binding \\",
                f"    {self.target_service_account} \\",
                f"    --member='{self.source_principal}' \\",
                f"    --role='roles/iam.serviceAccountTokenCreator'",
            ])
        else:
            lines.append("  Grant 'roles/iam.serviceAccountTokenCreator' to the source principal")

        lines.extend([
            "",
            "Also ensure the target service account has 'roles/aiplatform.user':",
        ])

        if self.target_service_account:
            sa_email = self.target_service_account
            lines.extend([
                f"  gcloud projects add-iam-policy-binding YOUR_PROJECT \\",
                f"    --member='serviceAccount:{sa_email}' \\",
                f"    --role='roles/aiplatform.user'",
            ])

        return "\n".join(lines)
