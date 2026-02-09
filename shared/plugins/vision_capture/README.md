# Vision Capture Plugin

Client-side TUI screenshot capture using Rich Console recording.

## Overview

The vision capture plugin provides:
- TUI state capture as SVG, PNG, or HTML
- Manual capture via `screenshot` command
- Auto-capture on turn end
- Periodic capture during streaming
- Integration with the output formatter pipeline

This is a **client-side only** feature - captures are saved locally and not sent to the model automatically.

## Components

### VisionCapture

Core capture utility that uses Rich Console with `record=True` to capture renderables.

```python
from shared.plugins.vision_capture import VisionCapture, CaptureContext

capture = VisionCapture()
capture.initialize()

# Capture a Rich renderable
result = capture.capture(
    panel,
    context=CaptureContext.USER_REQUESTED,
    turn_index=5,
    agent_id="main"
)

print(result.path)  # /tmp/jaato_vision/capture_20240115_143022_main_t5.svg
```

### VisionCaptureFormatter

Pipeline plugin that observes output and triggers captures at configured moments.

```python
from shared.plugins.vision_capture import VisionCaptureFormatter

formatter = VisionCaptureFormatter(
    capture_callback=my_capture_function
)

# Enable auto-capture on turn end
formatter.set_auto_capture(True)

# Enable periodic capture every 500ms during streaming
formatter.set_capture_interval(500)
```

The formatter runs at **priority 95** in the output pipeline, after all other formatters.

## Capture Contexts

| Context | Description |
|---------|-------------|
| `USER_REQUESTED` | Manual `screenshot` command |
| `TURN_END` | Auto-capture when model finishes responding |
| `PERIODIC` | Interval-based capture during streaming |
| `TOOL_END` | After tool execution (not currently used) |
| `ERROR` | On error (not currently used) |

## Configuration

### CaptureConfig

```python
from shared.plugins.vision_capture import CaptureConfig, CaptureFormat

config = CaptureConfig(
    output_dir="/tmp/jaato_vision",  # Output directory
    format=CaptureFormat.SVG,        # SVG, PNG, or HTML
    width=120,                       # Console width
    height=50,                       # Console height
    title="Jaato TUI",               # Window title in capture
    auto_cleanup_hours=24,           # Delete old captures
)

capture = VisionCapture()
capture.initialize(config)
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `JAATO_VISION_DIR` | Output directory (default: `/tmp/jaato_vision`) |
| `JAATO_VISION_FORMAT` | Format: `svg` (default), `png`, `html` |

## Output Formats

### SVG (default)

- Vector format, scales perfectly
- Small file size
- Readable in any browser
- No additional dependencies

### PNG

- Raster image
- Requires `cairosvg` Python package **and** the native Cairo library
- Better for sharing in tools that don't support SVG
- If Cairo is not available, the plugin automatically falls back to SVG

#### Installing Cairo for PNG support

PNG conversion uses the pipeline: Rich SVG export → CairoSVG → PNG. CairoSVG is a Python wrapper around the [Cairo](https://cairographics.org/) 2D graphics library, which must be installed separately as a system dependency.

**Linux (Debian/Ubuntu):**

```bash
sudo apt install libcairo2-dev
pip install cairosvg
```

**Linux (Fedora/RHEL):**

```bash
sudo dnf install cairo-devel
pip install cairosvg
```

**macOS:**

```bash
brew install cairo
pip install cairosvg
```

**Windows:**

Cairo does not ship as a standalone installer on Windows. Choose one of these options:

1. **Conda (recommended)** — installs a prebuilt Cairo binary into your environment:

   ```
   conda install cairo
   pip install cairosvg
   ```

2. **GTK3 runtime** — the [GTK for Windows](https://github.com/niclas-niclasen/gtk-3-runtime-build/releases) runtime bundle includes `libcairo-2.dll`. After installing, ensure the GTK `bin/` directory is on your `PATH` so Python can find the DLL.

3. **MSYS2** — install Cairo via the MSYS2 package manager:

   ```
   pacman -S mingw-w64-x86_64-cairo
   pip install cairosvg
   ```
   Ensure the MSYS2 `mingw64/bin` directory is on your `PATH`.

> **Troubleshooting on Windows:** If you see an error like:
> ```
> no library called "cairo-2" was found
> cannot load library 'libcairo-2.dll': error 0x7e
> ```
> This means the `cairosvg` Python package is installed but the native Cairo DLL is not found on the system `PATH`. Install one of the native options above and ensure the directory containing `libcairo-2.dll` is on your `PATH` environment variable. The screenshot command will fall back to SVG format automatically when Cairo is unavailable.

### HTML

- Interactive, can be opened in browser
- Preserves all Rich formatting
- Larger file size

## File Naming

Captures are named with context information:

```
capture_YYYYMMDD_HHMMSS_<agent>_t<turn>[_p<periodic>].<format>
```

Examples:
- `capture_20240115_143022_main_t5.svg` - Turn 5, main agent
- `capture_20240115_143025_main_t5_p3.svg` - Turn 5, 3rd periodic capture

## Pipeline Integration

The `VisionCaptureFormatter` integrates with the output formatter pipeline:

```
Output text
    ↓
hidden_content_filter (priority 5)
    ↓
diff_formatter (priority 20)
    ↓
code_block_formatter (priority 40)
    ↓
vision_capture_formatter (priority 95)  ← Observes, triggers captures
    ↓
Display
```

The formatter:
1. Passes through all text unchanged
2. Tracks turn boundaries via `reset()`
3. Triggers callback for configured capture modes

## Rich Client Integration

The rich client initializes vision capture lazily on first use:

```python
def _init_vision_capture(self):
    if self._vision_capture is None:
        self._vision_capture = VisionCapture()
        self._vision_capture.initialize()

        self._vision_formatter = VisionCaptureFormatter(
            capture_callback=self._on_vision_capture_triggered
        )
        self._display.register_formatter(self._vision_formatter)
```

## API Reference

### CaptureResult

```python
@dataclass
class CaptureResult:
    path: str                    # Full path to capture file
    format: CaptureFormat        # SVG, PNG, or HTML
    timestamp: datetime          # When captured
    context: CaptureContext      # What triggered capture
    width: int                   # Capture width
    height: int                  # Capture height
    turn_index: Optional[int]    # Turn number
    agent_id: Optional[str]      # Agent identifier
    error: Optional[str]         # Error message if failed

    @property
    def success(self) -> bool:
        return self.error is None
```

### VisionCapturePlugin Protocol

```python
@runtime_checkable
class VisionCapturePlugin(Protocol):
    @property
    def name(self) -> str: ...

    def initialize(self, config: Optional[CaptureConfig] = None) -> None: ...

    def capture(
        self,
        renderable,
        context: CaptureContext = CaptureContext.USER_REQUESTED,
        turn_index: Optional[int] = None,
        agent_id: Optional[str] = None,
    ) -> CaptureResult: ...

    def capture_ansi(
        self,
        ansi_text: str,
        context: CaptureContext = CaptureContext.USER_REQUESTED,
        turn_index: Optional[int] = None,
        agent_id: Optional[str] = None,
    ) -> CaptureResult: ...

    def get_last_capture(self) -> Optional[CaptureResult]: ...

    def cleanup_old_captures(self) -> int: ...
```

## Related

- [Rich Client README](../../../rich-client/README.md) - Client documentation
- [Plugin Development](../README.md) - How to create plugins
