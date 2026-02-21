# jaato-sdk

Lightweight Python SDK for building clients that connect to a jaato server.

## Installation

```bash
pip install --extra-index-url https://test.pypi.org/simple/ jaato-sdk
```

## Quick Start

```python
import asyncio
from jaato_sdk import IPCRecoveryClient
from jaato_sdk.events import AgentOutputEvent, ToolCallStartEvent

async def main():
    # Connect to a running jaato server (default: /tmp/jaato.sock)
    client = IPCRecoveryClient()
    await client.connect()

    # Send a message
    await client.send_message("Hello, what can you help me with?")

    # Process events
    async for event in client.events():
        if isinstance(event, AgentOutputEvent):
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolCallStartEvent):
            print(f"\n[Tool: {event.tool_name}]")

asyncio.run(main())
```

## What's Included

- **Event Protocol** - All event types for client-server communication (`jaato_sdk.events`)
- **IPC Client** - Low-level async client (`jaato_sdk.IPCClient`)
- **Recovery Client** - Auto-reconnecting client with state recovery (`jaato_sdk.IPCRecoveryClient`)
- **Configuration** - Recovery and connection settings (`jaato_sdk.RecoveryConfig`)

## Building a Custom Client

The SDK provides everything needed to build your own jaato client:

```python
from jaato_sdk import IPCClient
from jaato_sdk.events import (
    # Server -> Client events
    AgentOutputEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    PermissionRequestedEvent,

    # Client -> Server requests
    SendMessageRequest,
    PermissionResponseRequest,
    StopRequest,
)
```

See the [jaato-tui](../jaato-tui/) for a full TUI implementation example.

## Requirements

- Python 3.10+
- No heavy dependencies (just `python-dotenv`)
