"""Multimodal plugin for image handling via @file references.

This plugin enables the model to view images when needed by:
1. Detecting @file references to image files in user prompts
2. Enriching prompts to inform the model about viewImage tool availability
3. Providing a viewImage tool that returns images as multimodal function responses

The key insight is that the plugin does NOT automatically send images to the model.
Instead, it informs the model that images are available, and the model decides
whether to request them based on the user's intent.

Example:
    User: "What's in @screenshot.png?"
    -> Plugin detects image reference, enriches prompt
    -> Model sees: "What's in screenshot.png?" + info about viewImage tool
    -> Model calls viewImage("screenshot.png") to see the image
    -> Model describes the image content

    User: "Move @photo.jpg to /archive/"
    -> Plugin detects image reference, enriches prompt
    -> Model sees the prompt but decides it doesn't need to view the image
    -> Model just calls file move tool without viewing the image
"""

import mimetypes
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional

from ..model_provider.types import ToolSchema
from ..base import PromptEnrichmentResult, UserCommand


# Image file extensions we recognize
IMAGE_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.tif', '.ico', '.svg'
}

# Pattern to find @references (matches @path/to/file.ext)
AT_REFERENCE_PATTERN = re.compile(r'@([\w./\-]+\.\w+)')


class MultimodalPlugin:
    """Plugin that provides multimodal image viewing capabilities.

    This plugin:
    1. Subscribes to prompt enrichment to detect @image references
    2. Adds instructions about viewImage tool when images are detected
    3. Provides viewImage tool that returns multimodal function responses

    Requires Gemini 3 Pro or later for multimodal function responses.

    Configuration:
        base_path: Base directory for resolving relative file paths (default: cwd)
        max_image_size_mb: Maximum image file size in MB (default: 10)
    """

    # Model requirements: Gemini 3 Pro or later for multimodal function responses
    MODEL_REQUIREMENTS = [
        "gemini-3-pro*",
        "gemini-3.5-*",
        "gemini-4*",
    ]

    def __init__(self):
        self._base_path: Path = Path.cwd()
        self._max_image_size_mb: float = 10.0
        self._initialized = False
        # Track detected images from last prompt enrichment
        self._detected_images: Dict[str, Path] = {}

    @property
    def name(self) -> str:
        return "multimodal"

    def get_model_requirements(self) -> Optional[List[str]]:
        """Return model patterns required for multimodal function responses."""
        return self.MODEL_REQUIREMENTS

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the multimodal plugin.

        Args:
            config: Optional dict with:
                - base_path: Base directory for resolving relative paths
                - max_image_size_mb: Maximum image file size in MB
        """
        if config:
            if 'base_path' in config:
                self._base_path = Path(config['base_path'])
            if 'max_image_size_mb' in config:
                self._max_image_size_mb = float(config['max_image_size_mb'])
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the multimodal plugin."""
        self._detected_images.clear()
        self._initialized = False

    # ==================== Prompt Enrichment ====================

    def subscribes_to_prompt_enrichment(self) -> bool:
        """This plugin subscribes to prompt enrichment."""
        return True

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        """Detect @image references and enrich prompt with tool info.

        Finds @file.ext references where the extension indicates an image,
        and adds instructions about the viewImage tool.

        IMPORTANT: Does NOT remove @references - that's the framework's job.

        Args:
            prompt: The user's prompt text.

        Returns:
            PromptEnrichmentResult with enriched prompt and detected images metadata.
        """
        # Find all @references
        matches = AT_REFERENCE_PATTERN.findall(prompt)

        # Filter to image files that exist
        detected_images: Dict[str, str] = {}  # reference -> resolved path
        for ref in matches:
            ext = Path(ref).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                # Try to resolve the path
                resolved = self._resolve_path(ref)
                if resolved and resolved.exists():
                    detected_images[ref] = str(resolved)
                    self._detected_images[ref] = resolved

        # If no images detected, return unchanged
        if not detected_images:
            return PromptEnrichmentResult(prompt=prompt)

        # Build enrichment text
        image_list = ", ".join(detected_images.keys())
        enrichment = (
            f"\n\n[System: The following image files are referenced: {image_list}. "
            f"Use the viewImage(path) tool if you need to see the visual content of any image. "
            f"Only call viewImage if understanding the image content is necessary for the task.]"
        )

        enriched_prompt = prompt + enrichment

        return PromptEnrichmentResult(
            prompt=enriched_prompt,
            metadata={
                "detected_images": detected_images,
                "image_count": len(detected_images)
            }
        )

    def _resolve_path(self, ref: str) -> Optional[Path]:
        """Resolve a file reference to an absolute path.

        Args:
            ref: The file reference (possibly relative).

        Returns:
            Resolved Path or None if cannot resolve.
        """
        path = Path(ref)

        # If absolute, use as-is
        if path.is_absolute():
            return path

        # Try relative to base_path
        resolved = self._base_path / path
        if resolved.exists():
            return resolved.resolve()

        # Try relative to cwd
        resolved = Path.cwd() / path
        if resolved.exists():
            return resolved.resolve()

        return None

    # ==================== Function Declarations ====================

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return the ToolSchema for the viewImage tool."""
        return [ToolSchema(
            name='viewImage',
            description=(
                'View the visual content of an image file. Call this tool when you need to '
                'see what an image contains in order to answer the user\'s question. '
                'Do NOT call this for tasks that don\'t require seeing the image '
                '(e.g., moving, copying, or deleting image files).'
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to the image file. Can be absolute or relative to the "
                            "working directory. Supported formats: PNG, JPEG, GIF, WebP, BMP."
                        )
                    }
                },
                "required": ["path"]
            }
        )]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return the executor mapping."""
        return {'viewImage': self._execute_view_image}

    def _execute_view_image(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the viewImage tool.

        This returns a special result that the framework should interpret
        as a multimodal function response containing image data.

        Args:
            args: Dict containing 'path' to the image file.

        Returns:
            Dict with either:
            - Success: {'_multimodal': True, 'image_data': bytes, 'mime_type': str, ...}
            - Error: {'error': str}
        """
        try:
            path_str = args.get('path')
            if not path_str:
                return {'error': 'viewImage: path is required'}

            # Resolve the path
            resolved = self._resolve_path(path_str)
            if not resolved:
                # Also check if it was tracked from enrichment
                if path_str in self._detected_images:
                    resolved = self._detected_images[path_str]
                else:
                    return {'error': f'viewImage: file not found: {path_str}'}

            if not resolved.exists():
                return {'error': f'viewImage: file not found: {resolved}'}

            # Check file size
            size_mb = resolved.stat().st_size / (1024 * 1024)
            if size_mb > self._max_image_size_mb:
                return {
                    'error': f'viewImage: file too large ({size_mb:.1f}MB > {self._max_image_size_mb}MB)'
                }

            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(str(resolved))
            if not mime_type or not mime_type.startswith('image/'):
                mime_type = 'image/png'  # Default fallback

            # Read image data
            image_data = resolved.read_bytes()

            # Return special multimodal result
            # The framework/executor should recognize this and build
            # a proper FunctionResponsePart with multimodal data
            return {
                '_multimodal': True,
                '_multimodal_type': 'image',
                'image_data': image_data,
                'mime_type': mime_type,
                'display_name': resolved.name,
                'file_path': str(resolved),
                'size_bytes': len(image_data),
            }

        except PermissionError:
            return {'error': f'viewImage: permission denied: {args.get("path")}'}
        except Exception as exc:
            return {'error': f'viewImage: {str(exc)}'}

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for the multimodal plugin."""
        return None  # Instructions are added dynamically during prompt enrichment

    def get_auto_approved_tools(self) -> List[str]:
        """viewImage is auto-approved as it only reads files."""
        return ['viewImage']

    def get_user_commands(self) -> List[UserCommand]:
        """Multimodal plugin provides model tools only, no user commands."""
        return []


def create_plugin() -> MultimodalPlugin:
    """Factory function to create the multimodal plugin instance."""
    return MultimodalPlugin()
