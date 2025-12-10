# Moon Phase Calculator Plugin for jaato

A plugin that calculates moon phases for any date, providing phase names and illumination percentages using astronomical algorithms.

## Features

- Calculate moon phase for any date
- Returns phase name (New Moon, Waxing Crescent, First Quarter, etc.)
- Provides illumination percentage
- Optional detailed astronomical information
- Uses standard astronomical algorithms based on synodic month

## Installation

Install the plugin using pip in development mode:

```bash
cd jaato-moon-phase
pip install -e .
```

Or install from source:

```bash
pip install .
```

## Usage

Once installed, the plugin is automatically discovered by jaato's PluginRegistry:

```python
from shared import JaatoClient, PluginRegistry

# Create client and registry
client = JaatoClient()
client.connect(project="my-project", location="us-central1", model="gemini-2.5-flash")

registry = PluginRegistry(model_name="gemini-2.5-flash")
registry.discover()  # Finds moon_phase plugin via entry point

# Expose the plugin
registry.expose_tool("moon_phase")

# Configure client with tools
client.configure_tools(registry)

# Now the model can use the moon phase calculator
response = client.send_message(
    "What's the moon phase today?",
    on_output=lambda s, t, m: print(t, end="")
)
```

## Tool Schema

### calculate_moon_phase

Calculate the current moon phase for a given date.

**Parameters:**
- `date` (string, optional): Date in YYYY-MM-DD format. If not provided, uses current date.
- `include_details` (boolean, optional): Include additional astronomical details like age and distance.

**Returns:**
String with phase name, illumination percentage, and optional details.

## Configuration

The plugin accepts the following configuration options:

```python
registry.expose_tool("moon_phase", config={
    "precision": 2  # Decimal places for illumination percentage
})
```

## Testing

Run the standalone test script:

```bash
python test_moon_phase.py
```

## License

MIT
