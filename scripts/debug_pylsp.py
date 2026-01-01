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

    async def open_document(self, path: str):
        """Open a document."""
        uri = Path(path).as_uri()
        with open(path, "r") as f:
            content = f.read()
        await self.send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": "python",
                "version": 1,
                "text": content,
            }
        })
        print(f"  Opened: {path}")

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
        print("Example: python scripts/debug_pylsp.py /path/to/src")
        sys.exit(1)

    workspace_dir = os.path.abspath(sys.argv[1])
    root_uri = Path(workspace_dir).as_uri()

    print(f"\n=== Debug pylsp ===")
    print(f"Workspace: {workspace_dir}")

    # Find Python files
    py_files = list(Path(workspace_dir).glob("*.py"))
    print(f"Found {len(py_files)} Python files: {[f.name for f in py_files]}")

    if not py_files:
        print("No Python files found!")
        sys.exit(1)

    # Start client
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

    # Notify about files
    print("\n4. Notifying about files...")
    for f in py_files:
        await client.notify_file_created(str(f))

    # Wait a bit
    print("\n5. Waiting 2s for jedi to process...")
    await asyncio.sleep(2.0)

    # Open all documents
    print("\n6. Opening documents...")
    for f in py_files:
        await client.open_document(str(f))

    # Wait more
    print("\n7. Waiting 3s for jedi to analyze...")
    await asyncio.sleep(3.0)

    # Find lib.py
    lib_py = None
    for f in py_files:
        if f.name == "lib.py":
            lib_py = f
            break

    if lib_py:
        print(f"\n8. Getting symbols from {lib_py.name}...")
        symbols = await client.get_document_symbols(str(lib_py))
        print(f"  Found {len(symbols)} symbols:")
        for s in symbols:
            name = s.get("name", "?")
            kind = s.get("kind", "?")
            loc = s.get("location", {}).get("range", {}).get("start", {})
            line = loc.get("line", 0)
            char = loc.get("character", 0)
            print(f"    - {name} (kind={kind}) at line {line}, char {char}")

        # Try find_references for each symbol
        print(f"\n9. Testing find_references for each symbol...")
        for s in symbols:
            name = s.get("name", "?")
            loc = s.get("location", {}).get("range", {}).get("start", {})
            line = loc.get("line", 0)
            char = loc.get("character", 0)

            # Adjust char to point to the symbol name (after 'def ')
            if char == 0:
                char = 4  # 'def ' is 4 chars

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

    else:
        print("\n  No lib.py found, skipping symbol test")

    # Stop
    print("\n10. Stopping pylsp...")
    await client.stop()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
