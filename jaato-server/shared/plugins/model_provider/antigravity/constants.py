"""Constants for Antigravity provider.

Antigravity is Google's IDE backend that provides access to AI models
(Gemini 3, Claude) through Google OAuth authentication.

API endpoints and OAuth configuration sourced from:
https://github.com/NoeFabris/opencode-antigravity-auth
"""

# ==================== OAuth Configuration ====================
# Google OAuth client for Antigravity
OAUTH_CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
OAUTH_CLIENT_SECRET = "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"

# OAuth endpoints
OAUTH_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
OAUTH_USERINFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo?alt=json"

# Local callback configuration
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 51121
CALLBACK_PATH = "/oauth-callback"

# OAuth scopes required for Antigravity
OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]

# ==================== API Endpoints ====================
# Antigravity API endpoints (with fallbacks)
# Daily sandbox is primary, autopush and production are fallbacks
ANTIGRAVITY_ENDPOINTS = [
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
    "https://autopush-cloudcode-pa.sandbox.googleapis.com",
    "https://cloudcode-pa.googleapis.com",
]

# Default primary endpoint
ANTIGRAVITY_PRIMARY_ENDPOINT = "https://daily-cloudcode-pa.sandbox.googleapis.com"

# Gemini CLI endpoint (for non-Antigravity quota models)
GEMINI_CLI_ENDPOINT = "https://cloudcode-pa.googleapis.com"

# Code Assist API endpoint (for project ID resolution)
CODE_ASSIST_ENDPOINTS = [
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
    "https://autopush-cloudcode-pa.sandbox.googleapis.com",
    "https://cloudcode-pa.googleapis.com",
]

# ==================== Request Headers ====================
# User-Agent for Antigravity requests
ANTIGRAVITY_USER_AGENT = "antigravity/1.11.5 windows/amd64"

# API client header for Antigravity
ANTIGRAVITY_API_CLIENT = "google-cloud-sdk vscode_cloudshelleditor/0.1"

# Client metadata for Antigravity
ANTIGRAVITY_CLIENT_METADATA = (
    '{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}'
)

# Headers for Gemini CLI quota models
GEMINI_CLI_USER_AGENT = "google-api-nodejs-client/9.15.1"
GEMINI_CLI_API_CLIENT = "gl-node/22.17.0"
GEMINI_CLI_CLIENT_METADATA = (
    "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI"
)

# ==================== Default Configuration ====================
# Default project ID (used when API doesn't return one)
DEFAULT_PROJECT_ID = "rising-fact-p41fc"

# Provider ID for registration
PROVIDER_ID = "antigravity"

# ==================== Available Models ====================
# Models available through Antigravity quota
ANTIGRAVITY_MODELS = {
    # Gemini 3 models
    "antigravity-gemini-3-pro": {
        "context_limit": 1_048_576,
        "output_limit": 65_536,
        "thinking_levels": ["low", "high"],
        "api_model": "gemini-3.0-pro",
    },
    "antigravity-gemini-3-flash": {
        "context_limit": 1_048_576,
        "output_limit": 65_536,
        "thinking_levels": ["minimal", "low", "medium", "high"],
        "api_model": "gemini-3.0-flash",
    },
    # Claude models (via Antigravity)
    "antigravity-claude-sonnet-4-5": {
        "context_limit": 200_000,
        "output_limit": 64_000,
        "thinking_levels": [],
        "api_model": "claude-sonnet-4-5",
    },
    "antigravity-claude-sonnet-4-5-thinking": {
        "context_limit": 200_000,
        "output_limit": 64_000,
        "thinking_budgets": [8192, 32768],  # "low" = 8192, "max" = 32768
        "api_model": "claude-sonnet-4-5-thinking",
    },
    "antigravity-claude-opus-4-5-thinking": {
        "context_limit": 200_000,
        "output_limit": 64_000,
        "thinking_budgets": [8192, 32768],
        "api_model": "claude-opus-4-5-thinking",
    },
}

# Models available through Gemini CLI quota
GEMINI_CLI_MODELS = {
    "gemini-2.5-flash": {
        "context_limit": 1_048_576,
        "output_limit": 65_536,
        "api_model": "gemini-2.5-flash",
    },
    "gemini-2.5-pro": {
        "context_limit": 1_048_576,
        "output_limit": 65_536,
        "api_model": "gemini-2.5-pro",
    },
    "gemini-3-flash-preview": {
        "context_limit": 1_048_576,
        "output_limit": 65_536,
        "api_model": "gemini-3.0-flash-preview",
    },
    "gemini-3-pro-preview": {
        "context_limit": 1_048_576,
        "output_limit": 65_536,
        "api_model": "gemini-3.0-pro-preview",
    },
}

# Default context and output limits
DEFAULT_CONTEXT_LIMIT = 200_000
DEFAULT_OUTPUT_LIMIT = 8192
