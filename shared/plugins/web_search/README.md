# Web Search Plugin

The web search plugin provides the `web_search` function for performing internet searches using DuckDuckGo in the jaato framework.

## Overview

This plugin allows models to search the web for current information on any topic. It uses DuckDuckGo as the search backend, which provides good results without requiring API keys.

## Tool Declaration

The plugin exposes a single tool:

| Tool | Description |
|------|-------------|
| `web_search` | Search the web for information on any topic |

### Parameters

```json
{
  "query": "The search query to find information about",
  "max_results": 10
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | The search query to find information about |
| `max_results` | integer | No | Maximum number of results to return (default: 10) |

### Response

```json
{
  "query": "Python programming language",
  "result_count": 3,
  "results": [
    {
      "title": "Welcome to Python.org",
      "url": "https://www.python.org/",
      "snippet": "The official home of the Python Programming Language..."
    },
    {
      "title": "Python (programming language) - Wikipedia",
      "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
      "snippet": "Python is a high-level, general-purpose programming language..."
    }
  ]
}
```

On error:
```json
{
  "error": "Error message",
  "hint": "Optional hint for resolution"
}
```

## Usage

### Basic Setup

```python
from shared.plugins.registry import PluginRegistry

registry = PluginRegistry()
registry.discover()
registry.expose_all()  # web_search plugin is exposed by default
```

### With Configuration

```python
registry.expose_all({
    "web_search": {
        "max_results": 5,
        "safesearch": "strict",
        "region": "us-en"
    }
})
```

### With JaatoClient

```python
from shared import JaatoClient, PluginRegistry

client = JaatoClient()
client.connect(project_id, location, model_name)

registry = PluginRegistry()
registry.discover()
registry.expose_all()

client.configure_tools(registry)
response = client.send_message("Search for the latest Python 3.12 features")
```

## Search Features

### Effective Search Tips

1. **Be specific with queries**:
   ```python
   web_search(query="Python asyncio tutorial 2024")
   ```

2. **Include relevant keywords**:
   ```python
   web_search(query="machine learning image classification PyTorch")
   ```

3. **Add year/date for time-sensitive information**:
   ```python
   web_search(query="climate change statistics 2024")
   ```

4. **Limit results for focused responses**:
   ```python
   web_search(query="best Python IDE", max_results=5)
   ```

## Auto-Approval

The `web_search` tool is automatically approved and does not require user permission since it is a read-only operation that poses no security risk to the local system.

## System Instructions

The plugin provides these system instructions to the model:

```
You have access to `web_search` which searches the web for current information.

Use it to find up-to-date information about any topic, including:
- Current events and news
- Technical documentation and tutorials
- Product information and reviews
- Research and academic topics
- Any information that may have changed since your training cutoff

Example usage:
- Search for news: web_search(query="latest AI developments 2024")
- Find documentation: web_search(query="Python asyncio tutorial")
- Look up information: web_search(query="climate change statistics 2024")

The tool returns a list of search results with titles, URLs, and snippets.

Tips for effective searches:
- Be specific with your queries for better results
- Include relevant keywords and context
- Use quotes for exact phrase matching
- Add year/date for time-sensitive information
```

## Configuration Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_results` | int | `10` | Maximum number of search results to return |
| `timeout` | int | `10` | Request timeout in seconds |
| `region` | str | `"wt-wt"` | Region for search results (e.g., "us-en", "uk-en") |
| `safesearch` | str | `"moderate"` | Safe search level: "off", "moderate", or "strict" |

## Dependencies

This plugin requires the `ddgs` package:

```bash
pip install ddgs>=7.0.0
```

The package is listed in the project's `requirements.txt`.
