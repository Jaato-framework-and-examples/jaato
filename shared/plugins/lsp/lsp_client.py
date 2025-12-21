"""LSP client implementation using JSON-RPC over stdio."""

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TextIO, Tuple


@dataclass
class Position:
    """A position in a text document (0-indexed line and character)."""
    line: int
    character: int

    def to_dict(self) -> Dict[str, int]:
        return {"line": self.line, "character": self.character}

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "Position":
        return cls(line=d["line"], character=d["character"])


@dataclass
class Range:
    """A range in a text document."""
    start: Position
    end: Position

    def to_dict(self) -> Dict[str, Any]:
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Range":
        return cls(
            start=Position.from_dict(d["start"]),
            end=Position.from_dict(d["end"])
        )


@dataclass
class Location:
    """A location in a document."""
    uri: str
    range: Range

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Location":
        return cls(uri=d["uri"], range=Range.from_dict(d["range"]))


@dataclass
class Diagnostic:
    """A diagnostic (error, warning, etc.)."""
    range: Range
    message: str
    severity: int = 1  # 1=Error, 2=Warning, 3=Info, 4=Hint
    source: Optional[str] = None
    code: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Diagnostic":
        return cls(
            range=Range.from_dict(d["range"]),
            message=d["message"],
            severity=d.get("severity", 1),
            source=d.get("source"),
            code=str(d.get("code")) if d.get("code") else None
        )

    @property
    def severity_name(self) -> str:
        return {1: "Error", 2: "Warning", 3: "Info", 4: "Hint"}.get(self.severity, "Unknown")


@dataclass
class CompletionItem:
    """A completion item."""
    label: str
    kind: Optional[int] = None
    detail: Optional[str] = None
    documentation: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CompletionItem":
        doc = d.get("documentation")
        if isinstance(doc, dict):
            doc = doc.get("value", str(doc))
        return cls(
            label=d["label"],
            kind=d.get("kind"),
            detail=d.get("detail"),
            documentation=doc
        )


@dataclass
class Hover:
    """Hover information."""
    contents: str
    range: Optional[Range] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Hover":
        contents = d.get("contents", "")
        if isinstance(contents, dict):
            contents = contents.get("value", str(contents))
        elif isinstance(contents, list):
            parts = []
            for c in contents:
                if isinstance(c, dict):
                    parts.append(c.get("value", str(c)))
                else:
                    parts.append(str(c))
            contents = "\n".join(parts)
        return cls(
            contents=contents,
            range=Range.from_dict(d["range"]) if d.get("range") else None
        )


@dataclass
class SymbolInformation:
    """Information about a symbol."""
    name: str
    kind: int
    location: Location
    container_name: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SymbolInformation":
        return cls(
            name=d["name"],
            kind=d["kind"],
            location=Location.from_dict(d["location"]),
            container_name=d.get("containerName")
        )

    @property
    def kind_name(self) -> str:
        kinds = {
            1: "File", 2: "Module", 3: "Namespace", 4: "Package",
            5: "Class", 6: "Method", 7: "Property", 8: "Field",
            9: "Constructor", 10: "Enum", 11: "Interface", 12: "Function",
            13: "Variable", 14: "Constant", 15: "String", 16: "Number",
            17: "Boolean", 18: "Array", 19: "Object", 20: "Key",
            21: "Null", 22: "EnumMember", 23: "Struct", 24: "Event",
            25: "Operator", 26: "TypeParameter"
        }
        return kinds.get(self.kind, f"Unknown({self.kind})")


@dataclass
class ServerCapabilities:
    """Capabilities reported by the language server."""
    hover: bool = False
    completion: bool = False
    definition: bool = False
    references: bool = False
    document_symbol: bool = False
    workspace_symbol: bool = False
    rename: bool = False
    diagnostics: bool = False  # via textDocument/publishDiagnostics
    code_action: bool = False

    @classmethod
    def from_dict(cls, caps: Dict[str, Any]) -> "ServerCapabilities":
        return cls(
            hover=bool(caps.get("hoverProvider")),
            completion=bool(caps.get("completionProvider")),
            definition=bool(caps.get("definitionProvider")),
            references=bool(caps.get("referencesProvider")),
            document_symbol=bool(caps.get("documentSymbolProvider")),
            workspace_symbol=bool(caps.get("workspaceSymbolProvider")),
            rename=bool(caps.get("renameProvider")),
            code_action=bool(caps.get("codeActionProvider")),
            diagnostics=True  # Most servers support this via notifications
        )


@dataclass
class ServerConfig:
    """Configuration for an LSP server."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None
    root_uri: Optional[str] = None
    language_id: Optional[str] = None  # e.g., "python", "typescript"


class LSPClient:
    """Async LSP client that communicates via JSON-RPC over stdio."""

    def __init__(self, config: ServerConfig, errlog: Optional[TextIO] = None):
        self.config = config
        self._errlog = errlog or sys.stderr
        self._process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._initialized = False
        self._capabilities: Optional[ServerCapabilities] = None
        self._diagnostics: Dict[str, List[Diagnostic]] = {}  # uri -> diagnostics
        self._open_documents: Dict[str, int] = {}  # uri -> version

    async def start(self) -> None:
        """Start the language server process."""
        env = {**os.environ, **(self.config.env or {})}

        self._process = await asyncio.create_subprocess_exec(
            self.config.command,
            *self.config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self._errlog if hasattr(self._errlog, 'fileno') else asyncio.subprocess.DEVNULL,
            env=env
        )

        # Start reader task
        self._reader_task = asyncio.create_task(self._read_messages())

        # Initialize the server
        await self._initialize()

    async def stop(self) -> None:
        """Stop the language server."""
        if self._initialized:
            try:
                await self._send_request("shutdown", {})
                await self._send_notification("exit", {})
            except Exception:
                pass

        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()

    async def _initialize(self) -> None:
        """Send initialize request to server."""
        root_uri = self.config.root_uri or f"file://{os.getcwd()}"

        params = {
            "processId": os.getpid(),
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {
                    "hover": {"contentFormat": ["plaintext", "markdown"]},
                    "completion": {"completionItem": {"snippetSupport": False}},
                    "definition": {},
                    "references": {},
                    "documentSymbol": {},
                    "publishDiagnostics": {"relatedInformation": True}
                },
                "workspace": {
                    "symbol": {}
                }
            }
        }

        result = await self._send_request("initialize", params)
        self._capabilities = ServerCapabilities.from_dict(result.get("capabilities", {}))

        # Send initialized notification
        await self._send_notification("initialized", {})
        self._initialized = True

    async def _send_request(self, method: str, params: Dict[str, Any]) -> Any:
        """Send a request and wait for response."""
        self._request_id += 1
        req_id = self._request_id

        message = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params
        }

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[req_id] = future

        await self._write_message(message)

        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            self._pending_requests.pop(req_id, None)
            raise TimeoutError(f"LSP request {method} timed out")

    async def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a notification (no response expected)."""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        await self._write_message(message)

    async def _write_message(self, message: Dict[str, Any]) -> None:
        """Write a JSON-RPC message to the server."""
        content = json.dumps(message)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        self._process.stdin.write(header.encode('utf-8'))
        self._process.stdin.write(content.encode('utf-8'))
        await self._process.stdin.drain()

    async def _read_messages(self) -> None:
        """Read messages from the server."""
        while True:
            try:
                # Read headers
                headers = {}
                while True:
                    line = await self._process.stdout.readline()
                    if not line:
                        return  # EOF
                    line = line.decode('utf-8').strip()
                    if not line:
                        break  # End of headers
                    if ':' in line:
                        key, value = line.split(':', 1)
                        headers[key.strip().lower()] = value.strip()

                # Read content
                content_length = int(headers.get('content-length', 0))
                if content_length > 0:
                    content = await self._process.stdout.read(content_length)
                    message = json.loads(content.decode('utf-8'))
                    await self._handle_message(message)
            except asyncio.CancelledError:
                break
            except Exception:
                break

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle an incoming message."""
        if 'id' in message and 'method' not in message:
            # Response to a request
            req_id = message['id']
            if req_id in self._pending_requests:
                future = self._pending_requests.pop(req_id)
                if 'error' in message:
                    future.set_exception(Exception(message['error'].get('message', 'Unknown error')))
                else:
                    future.set_result(message.get('result'))
        elif 'method' in message:
            # Notification or request from server
            method = message['method']
            params = message.get('params', {})

            if method == 'textDocument/publishDiagnostics':
                uri = params.get('uri', '')
                diagnostics = [Diagnostic.from_dict(d) for d in params.get('diagnostics', [])]
                self._diagnostics[uri] = diagnostics

    # Public API methods

    @property
    def capabilities(self) -> Optional[ServerCapabilities]:
        return self._capabilities

    def uri_from_path(self, path: str) -> str:
        """Convert a file path to a URI."""
        abs_path = os.path.abspath(path)
        if sys.platform == 'win32':
            return f"file:///{abs_path.replace(os.sep, '/')}"
        return f"file://{abs_path}"

    async def open_document(self, path: str, text: Optional[str] = None) -> None:
        """Notify the server that a document is open."""
        uri = self.uri_from_path(path)
        if text is None:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()

        lang_id = self.config.language_id or self._guess_language_id(path)
        version = self._open_documents.get(uri, 0) + 1
        self._open_documents[uri] = version

        await self._send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": lang_id,
                "version": version,
                "text": text
            }
        })

    async def close_document(self, path: str) -> None:
        """Notify the server that a document is closed."""
        uri = self.uri_from_path(path)
        self._open_documents.pop(uri, None)
        await self._send_notification("textDocument/didClose", {
            "textDocument": {"uri": uri}
        })

    def _guess_language_id(self, path: str) -> str:
        """Guess the language ID from file extension."""
        ext = os.path.splitext(path)[1].lower()
        mapping = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescriptreact',
            '.jsx': 'javascriptreact',
            '.go': 'go',
            '.rs': 'rust',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.lua': 'lua',
            '.sh': 'shellscript',
            '.bash': 'shellscript',
            '.zsh': 'shellscript',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.xml': 'xml',
            '.html': 'html',
            '.css': 'css',
            '.md': 'markdown',
            '.sql': 'sql',
        }
        return mapping.get(ext, 'plaintext')

    async def goto_definition(self, path: str, line: int, character: int) -> List[Location]:
        """Get definition location(s) for symbol at position."""
        uri = self.uri_from_path(path)
        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character}
        }

        result = await self._send_request("textDocument/definition", params)
        if result is None:
            return []

        if isinstance(result, dict):
            return [Location.from_dict(result)]
        elif isinstance(result, list):
            return [Location.from_dict(loc) for loc in result]
        return []

    async def find_references(
        self, path: str, line: int, character: int, include_declaration: bool = True
    ) -> List[Location]:
        """Find all references to symbol at position."""
        uri = self.uri_from_path(path)
        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
            "context": {"includeDeclaration": include_declaration}
        }

        result = await self._send_request("textDocument/references", params)
        if result is None:
            return []
        return [Location.from_dict(loc) for loc in result]

    async def hover(self, path: str, line: int, character: int) -> Optional[Hover]:
        """Get hover information for position."""
        uri = self.uri_from_path(path)
        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character}
        }

        result = await self._send_request("textDocument/hover", params)
        if result is None:
            return None
        return Hover.from_dict(result)

    async def get_completions(self, path: str, line: int, character: int) -> List[CompletionItem]:
        """Get completion items at position."""
        uri = self.uri_from_path(path)
        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character}
        }

        result = await self._send_request("textDocument/completion", params)
        if result is None:
            return []

        items = result.get('items', result) if isinstance(result, dict) else result
        return [CompletionItem.from_dict(item) for item in items]

    async def get_document_symbols(self, path: str) -> List[SymbolInformation]:
        """Get all symbols in a document."""
        uri = self.uri_from_path(path)
        params = {"textDocument": {"uri": uri}}

        result = await self._send_request("textDocument/documentSymbol", params)
        if result is None:
            return []
        return [SymbolInformation.from_dict(sym) for sym in result]

    async def workspace_symbols(self, query: str) -> List[SymbolInformation]:
        """Search for symbols in workspace."""
        params = {"query": query}
        result = await self._send_request("workspace/symbol", params)
        if result is None:
            return []
        return [SymbolInformation.from_dict(sym) for sym in result]

    async def rename(self, path: str, line: int, character: int, new_name: str) -> Dict[str, Any]:
        """Rename symbol at position."""
        uri = self.uri_from_path(path)
        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
            "newName": new_name
        }

        result = await self._send_request("textDocument/rename", params)
        return result or {}

    def get_diagnostics(self, path: str) -> List[Diagnostic]:
        """Get cached diagnostics for a file."""
        uri = self.uri_from_path(path)
        return self._diagnostics.get(uri, [])

    async def get_code_actions(
        self, path: str, start_line: int, start_char: int, end_line: int, end_char: int
    ) -> List[Dict[str, Any]]:
        """Get available code actions for a range."""
        uri = self.uri_from_path(path)
        params = {
            "textDocument": {"uri": uri},
            "range": {
                "start": {"line": start_line, "character": start_char},
                "end": {"line": end_line, "character": end_char}
            },
            "context": {"diagnostics": []}
        }

        result = await self._send_request("textDocument/codeAction", params)
        return result or []
