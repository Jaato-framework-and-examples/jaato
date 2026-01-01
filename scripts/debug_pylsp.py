#!/usr/bin/env python3
"""Debug script to test pylsp's find_references directly.

Usage:
    python scripts/debug_pylsp.py <workspace_dir>

Example:
    python scripts/debug_pylsp.py /path/to/test_2/src
"""

import asyncio
import json
import os
import sys
import subprocess
from pathlib import Path


class LSPClient:
    """Simple LSP client for debugging."""

    def __init__(self):
        self.process = None
        self.request_id = 0
        self._pending = {}

    async def start(self):
        """Start pylsp process."""
        self.process = await asyncio.create_subprocess_exec(
            "pylsp", "-v",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Start reader task
        asyncio.create_task(self._read_responses())

    async def _read_responses(self):
        """Read responses from pylsp."""
        while True:
            try:
                # Read header
                header = b""
                while b"\r\n\r\n" not in header:
                    chunk = await self.process.stdout.read(1)
                    if not chunk:
                        return
                    header += chunk

                # Parse content length
                header_str = header.decode("utf-8")
                content_length = 0
                for line in header_str.split("\r\n"):
                    if line.startswith("Content-Length:"):
                        content_length = int(line.split(":")[1].strip())

                # Read content
                content = await self.process.stdout.read(content_length)
                msg = json.loads(content.decode("utf-8"))

                # Handle response
                if "id" in msg and msg["id"] in self._pending:
                    self._pending[msg["id"]].set_result(msg)

            except Exception as e:
                print(f"Error reading response: {e}", file=sys.stderr)
                break

    async def send_request(self, method: str, params: dict) -> dict:
        """Send a request and wait for response."""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params,
        }
        content = json.dumps(request)
        message = f"Content-Length: {len(content)}\r\n\r\n{content}"
        self.process.stdin.write(message.encode("utf-8"))
        await self.process.stdin.drain()

        # Wait for response
        future = asyncio.Future()
        self._pending[self.request_id] = future
        result = await asyncio.wait_for(future, timeout=30.0)
        del self._pending[self.request_id]
        return result

    async def send_notification(self, method: str, params: dict):
        """Send a notification (no response expected)."""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        content = json.dumps(notification)
        message = f"Content-Length: {len(content)}\r\n\r\n{content}"
        self.process.stdin.write(message.encode("utf-8"))
        await self.process.stdin.drain()

    async def initialize(self, root_uri: str):
        """Initialize the LSP server."""
        result = await self.send_request("initialize", {
            "processId": os.getpid(),
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {
                    "references": {"dynamicRegistration": False},
                    "documentSymbol": {"dynamicRegistration": False},
                },
                "workspace": {
                    "workspaceFolders": True,
                    "didChangeConfiguration": {"dynamicRegistration": False},
                    "didChangeWatchedFiles": {"dynamicRegistration": False},
                },
            },
            "workspaceFolders": [{"uri": root_uri, "name": "workspace"}],
        })
        await self.send_notification("initialized", {})
        return result

    async def open_document(self, path: str, version: int = 1):
        """Open a document."""
        uri = Path(path).as_uri()
        with open(path, "r") as f:
            content = f.read()
        await self.send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": "python",
                "version": version,
                "text": content,
            }
        })
        print(f"  Opened: {path}")

    async def update_document(self, path: str, version: int):
        """Notify about document change (simulates updateFile tool)."""
        uri = Path(path).as_uri()
        with open(path, "r") as f:
            content = f.read()
        await self.send_notification("textDocument/didChange", {
            "textDocument": {"uri": uri, "version": version},
            "contentChanges": [{"text": content}]
        })
        print(f"  Updated: {path} (version {version})")

    async def configure_extra_paths(self, paths: list):
        """Send extra_paths configuration."""
        await self.send_notification("workspace/didChangeConfiguration", {
            "settings": {
                "pylsp": {
                    "plugins": {
                        "jedi": {
                            "extra_paths": paths
                        }
                    }
                }
            }
        })
        print(f"  Configured extra_paths: {paths}")

    async def notify_file_created(self, path: str):
        """Notify about file creation."""
        uri = Path(path).as_uri()
        await self.send_notification("workspace/didChangeWatchedFiles", {
            "changes": [{"uri": uri, "type": 1}]  # 1 = Created
        })
        print(f"  Notified file created: {path}")

    async def get_document_symbols(self, path: str) -> list:
        """Get symbols in a document."""
        uri = Path(path).as_uri()
        result = await self.send_request("textDocument/documentSymbol", {
            "textDocument": {"uri": uri}
        })
        return result.get("result", [])

    async def find_references(self, path: str, line: int, char: int) -> list:
        """Find references to symbol at position."""
        uri = Path(path).as_uri()
        result = await self.send_request("textDocument/references", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": char},
            "context": {"includeDeclaration": False},
        })
        return result.get("result", [])

    async def stop(self):
        """Stop pylsp."""
        try:
            await self.send_request("shutdown", {})
            await self.send_notification("exit", {})
        except:
            pass
        self.process.terminate()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_pylsp.py <workspace_dir>")
        print("Example: python scripts/debug_pylsp.py /tmp/test_lsp")
        sys.exit(1)

    workspace_dir = os.path.abspath(sys.argv[1])
    root_uri = Path(workspace_dir).as_uri()

    print(f"\n=== Debug pylsp ===")
    print(f"Workspace: {workspace_dir}")

    # Create workspace directory if needed
    os.makedirs(workspace_dir, exist_ok=True)

    # Start client FIRST (before creating files, like the real client does)
    client = LSPClient()
    await client.start()
    print("\n1. Starting pylsp...")

    # Initialize
    print("\n2. Initializing...")
    init_result = await client.initialize(root_uri)
    caps = init_result.get("result", {}).get("capabilities", {})
    print(f"  Server capabilities: referencesProvider={caps.get('referencesProvider')}")

    # Configure extra_paths
    print("\n3. Configuring extra_paths...")
    await client.configure_extra_paths([workspace_dir])

    # Create lib.py (simulating writeNewFile tool)
    lib_py = Path(workspace_dir) / "lib.py"
    lib_content = '''def hello():
    print("Hello")

def goodbye():
    print("Goodbye")
'''
    print(f"\n4. Creating lib.py...")
    with open(lib_py, "w") as f:
        f.write(lib_content)
    print(f"  Written: {lib_py}")

    # Notify about lib.py and open it
    await client.notify_file_created(str(lib_py))
    await client.open_document(str(lib_py))

    # TEST 1: find_references BEFORE main.py exists (this is what the plugin does!)
    print(f"\n5. Testing find_references BEFORE main.py exists...")
    symbols = await client.get_document_symbols(str(lib_py))
    for s in symbols:
        name = s.get("name", "?")
        loc = s.get("location", {}).get("range", {}).get("start", {})
        line = loc.get("line", 0)
        char = 4 if loc.get("character", 0) == 0 else loc.get("character", 0)
        refs = await client.find_references(str(lib_py), line, char)
        if refs:
            print(f"    {name}: found {len(refs)} references!")
        else:
            print(f"    {name}: NO REFERENCES (expected - main.py doesn't exist yet)")

    # Create main.py (simulating writeNewFile tool)
    main_py = Path(workspace_dir) / "main.py"
    main_content = '''from lib import hello

hello()
'''
    print(f"\n6. Creating main.py...")
    with open(main_py, "w") as f:
        f.write(main_content)
    print(f"  Written: {main_py}")

    # Notify about main.py and open it
    await client.notify_file_created(str(main_py))
    await client.open_document(str(main_py))

    # TEST 2: find_references AFTER main.py exists (but without updating lib.py)
    print(f"\n7. Testing find_references AFTER main.py exists (no lib.py update)...")
    for s in symbols:
        name = s.get("name", "?")
        loc = s.get("location", {}).get("range", {}).get("start", {})
        line = loc.get("line", 0)
        char = 4 if loc.get("character", 0) == 0 else loc.get("character", 0)
        refs = await client.find_references(str(lib_py), line, char)
        if refs:
            print(f"    {name}: found {len(refs)} references!")
            for ref in refs:
                ref_uri = ref.get("uri", "")
                ref_range = ref.get("range", {}).get("start", {})
                ref_line = ref_range.get("line", 0)
                ref_char = ref_range.get("character", 0)
                print(f"      - {os.path.basename(ref_uri)}:{ref_line+1}:{ref_char}")
        else:
            print(f"    {name}: NO REFERENCES FOUND")

    # TEST 3: Update lib.py and test find_references (this is the real use case!)
    lib_content_v2 = '''def hello():
    print("Hello World!")  # Modified

def goodbye():
    print("Goodbye")

def new_function():
    print("New!")
'''
    print(f"\n8. Updating lib.py (simulating updateFile tool)...")
    with open(lib_py, "w") as f:
        f.write(lib_content_v2)
    await client.update_document(str(lib_py), version=2)

    # Get new symbols after update
    symbols = await client.get_document_symbols(str(lib_py))
    print(f"  Now has {len(symbols)} symbols")

    print(f"\n9. Testing find_references AFTER updating lib.py...")
    for s in symbols:
        name = s.get("name", "?")
        loc = s.get("location", {}).get("range", {}).get("start", {})
        line = loc.get("line", 0)
        char = 4 if loc.get("character", 0) == 0 else loc.get("character", 0)
        refs = await client.find_references(str(lib_py), line, char)
        if refs:
            print(f"    {name}: found {len(refs)} references!")
            for ref in refs:
                ref_uri = ref.get("uri", "")
                ref_range = ref.get("range", {}).get("start", {})
                ref_line = ref_range.get("line", 0)
                ref_char = ref_range.get("character", 0)
                print(f"      - {os.path.basename(ref_uri)}:{ref_line+1}:{ref_char}")
        else:
            print(f"    {name}: NO REFERENCES FOUND")

    # Stop
    print("\n10. Stopping pylsp...")
    await client.stop()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
