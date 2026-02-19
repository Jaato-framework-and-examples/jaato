# Multimodal Plugin

Image viewing via `@file` references with model-driven decision to load visual content.

## Overview

The multimodal plugin enables the model to view images when needed by:

1. **Detecting** `@image.png` references in user prompts
2. **Enriching** prompts to inform the model about the `viewImage` tool
3. **Providing** a `viewImage` tool that returns images as multimodal function responses

The key insight is that the plugin does **not** automatically send images to the model. Instead, it informs the model that images are available, and the model decides whether to request them based on the user's intent.

## Requirements

- **Gemini 3 Pro or later** - Multimodal function responses require Gemini 3+
- The plugin will not load on earlier models (graceful skip with warning)

## How It Works

### Flow Example: "What's in @photo.jpg?"

```
User: "What's in @photo.jpg?"
         ↓
Plugin detects @photo.jpg (image file exists)
         ↓
Plugin enriches prompt:
  "What's in @photo.jpg?
   [System: Image files referenced: photo.jpg. Use viewImage(path) if needed.]"
         ↓
Framework strips @references:
  "What's in photo.jpg? [System: ...]"
         ↓
Model reasons: "User wants contents → I need to see it → call viewImage"
         ↓
viewImage("photo.jpg") returns multimodal response with image bytes
         ↓
Model "sees" the image and describes it
```

### Flow Example: "Move @photo.jpg to /archive/"

```
User: "Move @photo.jpg to /archive/"
         ↓
Plugin detects @photo.jpg, enriches prompt
         ↓
Model reasons: "User wants to move file → I just need the path"
         ↓
Model calls cli_based_tool or file operation directly
         ↓
Image is NEVER loaded (efficient!)
```

## Usage

### Expose the Plugin

```python
from shared.plugins.registry import PluginRegistry

registry = PluginRegistry(model_name="gemini-3-pro")
registry.discover()
registry.expose_tool('multimodal')  # Will skip if model doesn't match
```

### Use in Prompts

Reference images with `@` prefix:

```
What's in @screenshot.png?
Compare @before.jpg with @after.jpg
Describe the UI elements in @mockup.png
```

### Configuration

```python
registry.expose_tool('multimodal', config={
    'base_path': '/path/to/images',  # Base directory for relative paths
    'max_image_size_mb': 5.0,        # Maximum file size (default: 10MB)
})
```

## Model Tool

### viewImage

View the visual content of an image file.

**Parameters:**
- `path` (string, required): Path to the image file

**Returns:**
Multimodal response with image data that the model can "see".

**Supported Formats:**
PNG, JPEG, GIF, WebP, BMP, TIFF, ICO, SVG

## Architecture

### Prompt Enrichment Pipeline

The plugin subscribes to the framework's prompt enrichment pipeline:

```python
class MultimodalPlugin:
    def subscribes_to_prompt_enrichment(self) -> bool:
        return True

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        # Detect @image references
        # Add viewImage tool availability info
        # Return enriched prompt (keep @ intact!)
```

### Model Requirements

Plugins can declare model compatibility:

```python
class MultimodalPlugin:
    MODEL_REQUIREMENTS = [
        "gemini-3-pro*",
        "gemini-3.5-*",
        "gemini-4*",
    ]

    def get_model_requirements(self) -> List[str]:
        return self.MODEL_REQUIREMENTS
```

If the current model doesn't match, the plugin is skipped with a warning.

### Multimodal Function Responses

The `viewImage` tool returns a special dict that the framework converts to multimodal Parts:

```python
{
    '_multimodal': True,
    '_multimodal_type': 'image',
    'image_data': bytes,
    'mime_type': 'image/png',
    'display_name': 'screenshot.png',
}
```

The framework builds:
- Function response Part with status
- Inline data Part with image bytes

## Auto-Approval

The `viewImage` tool is auto-approved since it only reads files (no side effects).

## Related

- [Plugin Development](../README.md) - How to create plugins
- [Architecture](../../../docs/architecture.md) - Framework architecture
