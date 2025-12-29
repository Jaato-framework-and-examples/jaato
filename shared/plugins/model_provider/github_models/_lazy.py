"""Lazy loading for Azure AI Inference SDK.

This module defers importing the heavy azure.ai.inference SDK until it's
actually needed, significantly improving startup time when this provider
isn't used.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import (
        AssistantMessage,
        SystemMessage,
        UserMessage,
    )
    from azure.core.credentials import AzureKeyCredential

# Cached module references
_inference_module = None
_models_module = None
_credentials_module = None


def get_chat_client_class() -> Any:
    """Get the ChatCompletionsClient class, importing it lazily.

    Returns:
        The ChatCompletionsClient class.

    Raises:
        ImportError: If the azure-ai-inference package is not installed.
    """
    global _inference_module
    if _inference_module is None:
        try:
            from azure.ai import inference
            _inference_module = inference
        except ImportError as e:
            raise ImportError(
                "azure-ai-inference package not installed. "
                "Install with: pip install azure-ai-inference"
            ) from e
    return _inference_module.ChatCompletionsClient


def get_models() -> Any:
    """Get the azure.ai.inference.models module, importing it lazily.

    Returns:
        The azure.ai.inference.models module.

    Raises:
        ImportError: If the azure-ai-inference package is not installed.
    """
    global _models_module
    if _models_module is None:
        try:
            from azure.ai.inference import models
            _models_module = models
        except ImportError as e:
            raise ImportError(
                "azure-ai-inference package not installed. "
                "Install with: pip install azure-ai-inference"
            ) from e
    return _models_module


def get_azure_key_credential() -> Any:
    """Get the AzureKeyCredential class, importing it lazily.

    Returns:
        The AzureKeyCredential class.

    Raises:
        ImportError: If the azure-core package is not installed.
    """
    global _credentials_module
    if _credentials_module is None:
        try:
            from azure.core import credentials
            _credentials_module = credentials
        except ImportError as e:
            raise ImportError(
                "azure-core package not installed. "
                "Install with: pip install azure-core"
            ) from e
    return _credentials_module.AzureKeyCredential


def get_response_format_json() -> Any:
    """Get the ChatCompletionsResponseFormatJSON class if available.

    Returns:
        The ChatCompletionsResponseFormatJSON class, or None if not available.
    """
    models = get_models()
    return getattr(models, 'ChatCompletionsResponseFormatJSON', None)
