"""LSP client implementation using JSON-RPC over stdio."""

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TextIO, Tuple


def _make_pdeathsig_preexec():
    """Build a ``preexec_fn`` that sets ``PR_SET_PDEATHSIG = SIGKILL``.

    Returns the callable on Linux (so a spawned language server is
    SIGKILL'd by the kernel the instant its owning runner process dies —
    graceful exit, SIGKILL, OOM, or crash), or ``None`` elsewhere
    (``preexec_fn=None`` is a harmless no-op, keeping the spawn portable
    to platforms without ``prctl``).

    The forked child runs a single ``libc`` syscall between fork and
    exec — the documented-safe use of ``preexec_fn`` (no Python
    allocation in the child).  See #284 / #280 (jdtls OOM leak).
    """
    if sys.platform != "linux":
        return None
    try:
        import ctypes
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        return None

    import signal as _signal
    _PR_SET_PDEATHSIG = 1

    def _preexec() -> None:
        # If the parent already died before this runs, prctl still
        # succeeds but the signal won't fire — the window between this
        # spawn and here is negligible.
        _libc.prctl(_PR_SET_PDEATHSIG, _signal.SIGKILL)

    return _preexec


# Resolved once at import: the preexec_fn passed to every language-server
# spawn (None on non-Linux).
_set_pdeathsig_sigkill = _make_pdeathsig_preexec()


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
class TextEdit:
    """A textual edit applicable to a text document."""
    range: Range
    new_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"range": self.range.to_dict(), "newText": self.new_text}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TextEdit":
        return cls(
            range=Range.from_dict(d["range"]),
            new_text=d.get("newText", "")
        )


@dataclass
class WorkspaceEdit:
    """A workspace edit represents changes to many resources in the workspace.

    The edit should either provide `changes` or `documentChanges`. If the client
    can handle versioned document edits and if `documentChanges` are present,
    the latter are preferred over `changes`.
    """
    # Simple map of uri -> list of text edits
    changes: Dict[str, List[TextEdit]] = field(default_factory=dict)
    # More structured document changes (not always present)
    document_changes: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkspaceEdit":
        changes: Dict[str, List[TextEdit]] = {}

        # Parse 'changes' format: { uri: TextEdit[] }
        if "changes" in d and d["changes"]:
            for uri, edits in d["changes"].items():
                changes[uri] = [TextEdit.from_dict(e) for e in edits]

        # Parse 'documentChanges' format if present
        document_changes = d.get("documentChanges")
        if document_changes:
            for doc_change in document_changes:
                # Handle TextDocumentEdit format
                if "textDocument" in doc_change and "edits" in doc_change:
                    uri = doc_change["textDocument"]["uri"]
                    if uri not in changes:
                        changes[uri] = []
                    changes[uri].extend([TextEdit.from_dict(e) for e in doc_change["edits"]])

        return cls(changes=changes, document_changes=document_changes)

    def get_affected_files(self) -> List[str]:
        """Return list of URIs that would be affected by this edit."""
        return list(self.changes.keys())


@dataclass
class CodeAction:
    """A code action represents a change that can be performed in code.

    Examples: refactoring, quick fixes, extract method, etc.
    """
    title: str
    kind: Optional[str] = None  # e.g., "refactor.extract", "quickfix"
    diagnostics: List["Diagnostic"] = field(default_factory=list)
    is_preferred: bool = False
    disabled: Optional[str] = None  # Reason if disabled
    edit: Optional[WorkspaceEdit] = None
    command: Optional[Dict[str, Any]] = None  # Command to execute
    data: Optional[Any] = None  # Additional data for resolve

    # Known code action kinds (from LSP spec)
    KIND_QUICKFIX = "quickfix"
    KIND_REFACTOR = "refactor"
    KIND_REFACTOR_EXTRACT = "refactor.extract"
    KIND_REFACTOR_INLINE = "refactor.inline"
    KIND_REFACTOR_REWRITE = "refactor.rewrite"
    KIND_SOURCE = "source"
    KIND_SOURCE_ORGANIZE_IMPORTS = "source.organizeImports"
    KIND_SOURCE_FIX_ALL = "source.fixAll"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CodeAction":
        edit = None
        if d.get("edit"):
            edit = WorkspaceEdit.from_dict(d["edit"])

        disabled = None
        if d.get("disabled"):
            disabled = d["disabled"].get("reason", str(d["disabled"]))

        return cls(
            title=d.get("title", ""),
            kind=d.get("kind"),
            diagnostics=[],  # Not parsing diagnostics for simplicity
            is_preferred=d.get("isPreferred", False),
            disabled=disabled,
            edit=edit,
            command=d.get("command"),
            data=d.get("data")
        )

    def is_refactoring(self) -> bool:
        """Check if this is a refactoring action."""
        return self.kind is not None and self.kind.startswith("refactor")

    def is_quickfix(self) -> bool:
        """Check if this is a quick fix action."""
        return self.kind == self.KIND_QUICKFIX

    def to_summary(self) -> Dict[str, Any]:
        """Return a summary suitable for tool output."""
        summary: Dict[str, Any] = {
            "title": self.title,
            "kind": self.kind or "unknown"
        }
        if self.is_preferred:
            summary["preferred"] = True
        if self.disabled:
            summary["disabled"] = self.disabled
        if self.edit:
            summary["has_edit"] = True
            summary["affected_files"] = len(self.edit.changes)
        if self.command:
            summary["has_command"] = True
            summary["command_name"] = self.command.get("command", "unknown")
        return summary


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
    """Information about a symbol.

    Handles both SymbolInformation and DocumentSymbol LSP formats:
    - SymbolInformation: flat list with 'location' field (older servers)
    - DocumentSymbol: hierarchical with 'range'/'selectionRange' (modern servers like pyright)
    """
    name: str
    kind: int
    location: Location
    container_name: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any], file_uri: Optional[str] = None) -> "SymbolInformation":
        """Parse symbol from either SymbolInformation or DocumentSymbol format.

        Args:
            d: The symbol dictionary from LSP response
            file_uri: File URI to use when parsing DocumentSymbol format
                     (DocumentSymbol doesn't include URI, only range)
        """
        # Check if this is DocumentSymbol format (has 'range' but no 'location')
        if "range" in d and "location" not in d:
            # DocumentSymbol format - construct location from range
            if not file_uri:
                # Fall back to empty URI if not provided
                file_uri = ""
            location = Location(
                uri=file_uri,
                range=Range.from_dict(d.get("selectionRange", d["range"]))
            )
        else:
            # SymbolInformation format - use location directly
            location = Location.from_dict(d["location"])

        return cls(
            name=d["name"],
            kind=d["kind"],
            location=location,
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
    extra_paths_key: Optional[str] = None  # e.g., "pylsp.plugins.jedi.extra_paths"


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
        # Per-URI asyncio.Event signalled whenever the server pushes a
        # `textDocument/publishDiagnostics` notification for that URI.
        # Used by `await_diagnostics()` so callers can wait for the
        # first batch instead of sleeping a fixed interval.  The event
        # is recreated on each `await_diagnostics()` consume cycle so
        # subsequent didChange + await pairs work correctly.
        self._diagnostics_events: Dict[str, asyncio.Event] = {}
        self._open_documents: Dict[str, int] = {}  # uri -> version
        self._configured_extra_paths: Set[str] = set()  # paths already sent to server
        self._workspace_folders: Set[str] = set()  # workspace folder URIs added

    async def start(self) -> None:
        """Start the language server process.

        The child is spawned with a kernel-enforced parent-death signal
        (``PR_SET_PDEATHSIG = SIGKILL``, Linux) so the language server —
        a heavyweight process (jdtls routinely sits at 0.5-1GB) — is
        reaped the instant its owning runner dies, no matter HOW it dies:
        graceful shutdown, SIGKILL, OOM, or crash.  Without this, a
        runner that's killed (e.g. OOM, pool recycle, mid-cascade abort)
        leaks its language server, which then re-parents to the daemon
        (subreaper) and accumulates — one per cascade stage — until the
        daemon itself OOMs.  This is the kernel backstop; the graceful
        ``stop()`` path (thread-cleanup → ``disconnect_server``) handles
        the normal session-end case.  See #284 / #280.
        """
        env = {**os.environ, **(self.config.env or {})}

        self._process = await asyncio.create_subprocess_exec(
            self.config.command,
            *self.config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self._errlog if hasattr(self._errlog, 'fileno') else asyncio.subprocess.DEVNULL,
            env=env,
            preexec_fn=_set_pdeathsig_sigkill,
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
        root_uri = self.config.root_uri or "file:///"
        workspace_name = os.path.basename(root_uri.replace("file://", "")) or "workspace"

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
                    "symbol": {},
                    "workspaceFolders": True,
                    "didChangeConfiguration": {"dynamicRegistration": False},
                    "didChangeWatchedFiles": {"dynamicRegistration": False}
                }
            },
            "workspaceFolders": [{"uri": root_uri, "name": workspace_name}]
        }

        result = await self._send_request("initialize", params)
        self._capabilities = ServerCapabilities.from_dict(result.get("capabilities", {}))

        # Track the initial workspace folder
        self._workspace_folders.add(root_uri)

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
                # Signal any caller awaiting first batch via
                # `await_diagnostics(path, ...)`.  We lazily create the
                # Event so URIs the server pushes for spontaneously
                # (e.g. cross-file analysis ripples) also get a signal
                # if a future awaiter shows up.
                event = self._diagnostics_events.get(uri)
                if event is None:
                    event = asyncio.Event()
                    self._diagnostics_events[uri] = event
                event.set()

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
        """Notify the server that a document is open.

        If the document is already open, this is a no-op. Use update_document()
        to notify the server of content changes.

        For new files, this also:
        1. Configures extra_paths for the directory (for cross-file imports)
        2. Sends didChangeWatchedFiles to notify about file existence
        3. Sends didOpen with the file content
        """
        uri = self.uri_from_path(path)

        # Don't re-open documents that are already open
        if uri in self._open_documents:
            return

        if text is None:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()

        # Configure extra_paths for this file's directory BEFORE opening
        # This is critical for cross-file imports to work (like in debug_pylsp.py)
        directory = os.path.dirname(os.path.abspath(path))
        await self.configure_extra_paths([directory])

        # Add directory as workspace folder for better cross-file indexing
        await self.add_workspace_folder(directory)

        # Notify server about new file existence BEFORE opening
        # This is important for servers like pylsp/jedi to track cross-file references
        await self._send_notification("workspace/didChangeWatchedFiles", {
            "changes": [{"uri": uri, "type": 1}]  # 1 = Created
        })

        lang_id = self.config.language_id or self._guess_language_id(path)
        self._open_documents[uri] = 1  # Start at version 1

        await self._send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": lang_id,
                "version": 1,
                "text": text
            }
        })

    async def update_document(self, path: str, text: Optional[str] = None) -> None:
        """Notify the server that a document's content has changed.

        If the document isn't open yet, this will open it first.
        """
        uri = self.uri_from_path(path)

        # If not open, open it first
        if uri not in self._open_documents:
            await self.open_document(path, text)
            return

        if text is None:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()

        # Increment version
        version = self._open_documents[uri] + 1
        self._open_documents[uri] = version

        await self._send_notification("textDocument/didChange", {
            "textDocument": {"uri": uri, "version": version},
            "contentChanges": [{"text": text}]
        })

    async def close_document(self, path: str) -> None:
        """Notify the server that a document is closed."""
        uri = self.uri_from_path(path)
        self._open_documents.pop(uri, None)
        await self._send_notification("textDocument/didClose", {
            "textDocument": {"uri": uri}
        })

    async def notify_files_created(self, paths: List[str]) -> None:
        """Notify the server that files were created.

        This triggers the server to index the new files so they can be
        found by workspace-wide operations like find_references.

        Args:
            paths: List of file paths that were created.
        """
        changes = [
            {"uri": self.uri_from_path(p), "type": 1}  # 1 = Created
            for p in paths
        ]
        if changes:
            await self._send_notification("workspace/didChangeWatchedFiles", {
                "changes": changes
            })

    async def update_configuration(self, settings: Dict[str, Any]) -> None:
        """Send workspace/didChangeConfiguration notification.

        This updates the language server's configuration dynamically.
        Useful for setting extra_paths for Python servers like pylsp.

        Args:
            settings: The settings object to send to the server.
        """
        await self._send_notification("workspace/didChangeConfiguration", {
            "settings": settings
        })

    async def configure_extra_paths(self, extra_paths: List[str]) -> None:
        """Configure extra paths for module resolution using server-specific key.

        The key is read from the server's extraPathsKey configuration in .lsp.json.
        If no key is configured, this method does nothing. Paths that have already
        been configured are skipped to avoid redundant updates.

        Args:
            extra_paths: List of directory paths to add to the module search path.
        """
        if not self.config.extra_paths_key:
            return  # No extra_paths configuration for this server

        # Check if all paths are already configured
        new_paths = [p for p in extra_paths if p not in self._configured_extra_paths]
        if not new_paths:
            return  # All paths already configured

        # Update the cache
        self._configured_extra_paths.update(new_paths)

        # Send all configured paths (not just new ones) since LSP expects the full list
        all_paths = list(self._configured_extra_paths)

        # Convert dotted key (e.g., "pylsp.plugins.jedi.extra_paths") to nested structure
        # LSP servers expect nested dicts, not flat dotted keys
        key_parts = self.config.extra_paths_key.split('.')
        settings: Dict[str, Any] = {}
        current = settings
        for part in key_parts[:-1]:
            current[part] = {}
            current = current[part]
        current[key_parts[-1]] = all_paths

        await self.update_configuration(settings)

    async def add_workspace_folder(self, directory: str) -> None:
        """Add a directory as a workspace folder.

        This is useful when working with files outside the original rootUri.
        Some servers like pylsp need workspace folders to properly index cross-file references.
        """
        uri = self.uri_from_path(directory)
        name = os.path.basename(directory) or "workspace"

        # Track added workspace folders to avoid duplicates
        if uri in self._workspace_folders:
            return

        self._workspace_folders.add(uri)

        await self._send_notification("workspace/didChangeWorkspaceFolders", {
            "event": {
                "added": [{"uri": uri, "name": name}],
                "removed": []
            }
        })

    async def ensure_workspace_indexed(self, directory: str, extensions: Optional[List[str]] = None) -> None:
        """Open all files of supported types in a directory to ensure they're indexed.

        This is needed for find_references to work across files that
        haven't been explicitly opened yet.

        Args:
            directory: Directory to scan for files.
            extensions: List of file extensions to include (e.g., ['.py', '.pyi']).
                       If None, uses all extensions this server supports.
        """
        import glob

        # Configure extra_paths if the server supports it (via extraPathsKey in .lsp.json)
        await self.configure_extra_paths([directory])

        if extensions is None:
            # Use the language ID to determine supported extensions
            lang_to_exts = {
                'python': ['.py', '.pyi', '.pyw'],
                'javascript': ['.js', '.mjs', '.cjs'],
                'typescript': ['.ts', '.mts', '.cts'],
                'typescriptreact': ['.tsx'],
                'javascriptreact': ['.jsx'],
                'go': ['.go'],
                'rust': ['.rs'],
                'java': ['.java'],
                'c': ['.c', '.h'],
                'cpp': ['.cpp', '.hpp', '.cc', '.hh'],
                'ruby': ['.rb'],
                'php': ['.php'],
            }
            lang = self.config.language_id
            extensions = lang_to_exts.get(lang, []) if lang else []

        # Collect all files to index
        all_files = []
        for ext in extensions:
            pattern = os.path.join(directory, "**", f"*{ext}")
            all_files.extend(glob.glob(pattern, recursive=True))

        # Find files that aren't already open - these are "new" to the LSP server
        new_files = [f for f in all_files if self.uri_from_path(f) not in self._open_documents]

        # Open files that aren't already open
        # open_document handles: configure_extra_paths, didChangeWatchedFiles, didOpen
        for file_path in new_files:
            await self.open_document(file_path)

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
        """Get all symbols in a document.

        Handles both SymbolInformation[] and DocumentSymbol[] response formats.
        DocumentSymbol is hierarchical, so we flatten it to a list.
        """
        uri = self.uri_from_path(path)
        params = {"textDocument": {"uri": uri}}

        result = await self._send_request("textDocument/documentSymbol", params)
        if result is None:
            return []

        symbols = []
        self._collect_symbols(result, uri, symbols)
        return symbols

    def _collect_symbols(
        self,
        items: List[Dict[str, Any]],
        file_uri: str,
        out: List[SymbolInformation],
        container_name: Optional[str] = None
    ) -> None:
        """Recursively collect symbols from potentially hierarchical DocumentSymbol structure.

        Args:
            items: List of symbol dictionaries (SymbolInformation or DocumentSymbol)
            file_uri: The file URI for constructing locations
            out: Output list to append symbols to
            container_name: Name of the containing symbol (for nested symbols)
        """
        for item in items:
            # Debug logging for symbol parsing
            sel_range = item.get("selectionRange")
            range_data = item.get("range")
            used_range = sel_range if sel_range else range_data
            from shared.trace import trace as _trace_write
            _trace_write("LSP_CLIENT", f"{item.get('name')}: selectionRange={sel_range}, range={range_data}, using={used_range}")
            if used_range:
                start = used_range.get('start', {})
                _trace_write("LSP_CLIENT", f"{item.get('name')}: start.line={start.get('line')}, start.character={start.get('character')}")
            sym = SymbolInformation.from_dict(item, file_uri)
            if container_name:
                sym.container_name = container_name
            out.append(sym)

            # DocumentSymbol can have children - recurse into them
            children = item.get("children", [])
            if children:
                self._collect_symbols(children, file_uri, out, container_name=sym.name)

    async def workspace_symbols(self, query: str) -> List[SymbolInformation]:
        """Search for symbols in workspace."""
        params = {"query": query}
        result = await self._send_request("workspace/symbol", params)
        if result is None:
            return []
        return [SymbolInformation.from_dict(sym) for sym in result]

    async def rename(self, path: str, line: int, character: int, new_name: str) -> Optional[WorkspaceEdit]:
        """Rename symbol at position.

        Returns:
            WorkspaceEdit containing all changes needed for the rename,
            or None if the rename is not possible.
        """
        uri = self.uri_from_path(path)
        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
            "newName": new_name
        }

        result = await self._send_request("textDocument/rename", params)
        if result is None:
            return None
        return WorkspaceEdit.from_dict(result)

    async def prepare_rename(self, path: str, line: int, character: int) -> Optional[Dict[str, Any]]:
        """Check if rename is valid at position and get the text to be renamed.

        Returns:
            Dict with 'range' and optionally 'placeholder' if rename is possible,
            or None if rename is not supported at this position.
        """
        uri = self.uri_from_path(path)
        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character}
        }

        try:
            result = await self._send_request("textDocument/prepareRename", params)
            return result
        except Exception:
            # prepareRename is optional - server may not support it
            return None

    def get_diagnostics(self, path: str) -> List[Diagnostic]:
        """Get cached diagnostics for a file."""
        uri = self.uri_from_path(path)
        return self._diagnostics.get(uri, [])

    async def await_diagnostics(
        self,
        path: str,
        max_wait: float,
        min_wait: float = 0.0,
        convergence_window: float = 0.0,
    ) -> bool:
        """Block until the server pushes a `publishDiagnostics` batch for ``path``.

        Returns AS SOON AS the first diagnostic batch arrives — and,
        when ``convergence_window > 0``, keeps listening for follow-up
        batches that overwrite the cache until ``convergence_window``
        seconds pass without one.  Returns ``True`` if at least one
        ``publishDiagnostics`` notification was observed within the
        ``max_wait`` budget; ``False`` if the budget expired without
        any notification.

        ## Why a convergence loop?

        Eclipse JDT LS (jdtls) on a cold Maven workspace publishes
        multiple batches per URI as its multi-stage analysis pipeline
        progresses.  Empirically (2026-06-05 instrumented cascade,
        91 adjacent-publish races over 30 distinct ``.java`` URIs,
        ``/tmp/converge2.py`` analysis):

        - First publish is often an EMPTY batch (jdtls just accepted
          ``didOpen``, parsing is in progress).  Median delay from
          ``didOpen`` to first publish: ~500 ms.
        - Real analysis (intra-project resolution, import settlement)
          lands as a follow-up publish at ``+1500 ms`` median, ``+3 s``
          p75 from the first publish.  ``Customer.java`` in the
          concrete case study: ``12 errors`` at T+0 → ``0 errors`` at
          T+2.2 s.  Returning on first-publish reads the cache before
          jdtls converged — the agent sees phantom errors that don't
          exist after settling.

        Pre-server-0.6.193 behaviour (``convergence_window=0``) returns
        on first publish — keeps the legacy fast path for tests and
        for callers that want raw first-batch semantics.

        Args:
            path: file path the caller wants diagnostics for.
            max_wait: upper bound on TOTAL wait time (including
                ``min_wait`` and any convergence loop).
            min_wait: floor on the first-publish wait.  See
                pre-existing behaviour for multi-stage analysis
                rationale.
            convergence_window: seconds to keep listening for
                follow-up publishes AFTER the first publish arrives.
                Each follow-up resets the window timer (jdtls's
                re-publish cascade may span multiple cycles).
                ``0.0`` disables the loop (legacy first-publish
                semantics).  Recommended ``3.0`` for jdtls on Maven
                projects (catches the ~75 % cluster of fresh-render
                races per the 2026-06-05 analysis); raise for stacks
                with a heavier multi-stage tail.

        Returns:
            ``True`` if a ``publishDiagnostics`` notification arrived
            within ``max_wait``; ``False`` if the budget expired
            without any.  The caller is expected to
            ``get_diagnostics`` afterwards regardless — ``False``
            just tells them the cache may be stale.

        Implementation notes:
            - The Event is cleared on every observed publish so the
              loop sees only NEW notifications, not the one that
              ended the prior iteration.  The cache
              (``self._diagnostics[uri]``) is NOT cleared — readers
              still want the latest batch via ``get_diagnostics``.
            - ``max_wait`` of 0 disables the await entirely (legacy
              "read cache as-is" behavior).
            - ``max_wait`` is a HARD ceiling.  Even if the convergence
              loop is still seeing follow-up publishes, the call
              returns once the cumulative budget elapses — a stack
              that re-publishes forever won't hang the call site.
        """
        uri = self.uri_from_path(path)
        event = self._diagnostics_events.get(uri)
        if event is None:
            event = asyncio.Event()
            self._diagnostics_events[uri] = event

        if max_wait <= 0:
            return event.is_set()

        start = time.monotonic()
        deadline = start + max_wait

        # Phase 1 — min_wait floor.  Multi-stage analysis pipelines
        # may already have published one or more batches during this
        # sleep; the cache holds the latest one when we wake.
        if min_wait > 0:
            await asyncio.sleep(min_wait)

        # Phase 2 — wait for the first publish (or until the
        # ``max_wait`` deadline).  When the Event was set during
        # ``min_wait``, this returns immediately.
        first_publish_seen = False
        remaining = deadline - time.monotonic()
        if event.is_set():
            first_publish_seen = True
        elif remaining > 0:
            try:
                await asyncio.wait_for(event.wait(), timeout=remaining)
                first_publish_seen = True
            except asyncio.TimeoutError:
                event.clear()
                return False

        if not first_publish_seen:
            event.clear()
            return False

        # Phase 3 — convergence loop.  After the first publish, keep
        # listening for follow-up publishes that overwrite the cache.
        # Each observed publish RESETS the window timer because jdtls's
        # multi-stage cascade may span several re-publishes.  Exits
        # when ``convergence_window`` elapses without a new publish,
        # or when the cumulative ``max_wait`` deadline hits.
        while convergence_window > 0:
            event.clear()
            time_to_deadline = deadline - time.monotonic()
            if time_to_deadline <= 0:
                break
            window_budget = min(convergence_window, time_to_deadline)
            try:
                await asyncio.wait_for(event.wait(), timeout=window_budget)
                # Follow-up publish arrived — cache overwritten with
                # newer state.  Loop to check for more.
            except asyncio.TimeoutError:
                # ``convergence_window`` elapsed with no new publish:
                # jdtls has settled on this URI.
                break

        event.clear()
        return True

    async def get_code_actions(
        self,
        path: str,
        start_line: int,
        start_char: int,
        end_line: int,
        end_char: int,
        only_kinds: Optional[List[str]] = None
    ) -> List[CodeAction]:
        """Get available code actions for a range.

        Args:
            path: File path
            start_line: Start line (0-indexed)
            start_char: Start character (0-indexed)
            end_line: End line (0-indexed)
            end_char: End character (0-indexed)
            only_kinds: Optional list of code action kinds to filter
                       (e.g., ["refactor.extract", "refactor.inline"])

        Returns:
            List of available CodeAction objects.
        """
        uri = self.uri_from_path(path)
        context: Dict[str, Any] = {"diagnostics": []}
        if only_kinds:
            context["only"] = only_kinds

        params = {
            "textDocument": {"uri": uri},
            "range": {
                "start": {"line": start_line, "character": start_char},
                "end": {"line": end_line, "character": end_char}
            },
            "context": context
        }

        result = await self._send_request("textDocument/codeAction", params)
        if result is None:
            return []

        actions = []
        for item in result:
            # Server may return Command or CodeAction
            if "title" in item:
                actions.append(CodeAction.from_dict(item))
        return actions

    async def resolve_code_action(self, action: CodeAction) -> CodeAction:
        """Resolve a code action to get its edit/command if not already present.

        Some servers return minimal code actions that need to be resolved
        to get the actual workspace edit.
        """
        if action.edit is not None:
            return action  # Already resolved

        # Build the request payload from the action
        params = {
            "title": action.title,
            "kind": action.kind,
            "data": action.data
        }

        try:
            result = await self._send_request("codeAction/resolve", params)
            if result:
                return CodeAction.from_dict(result)
        except Exception:
            pass  # resolve is optional
        return action

    async def execute_command(self, command: str, arguments: Optional[List[Any]] = None) -> Any:
        """Execute a workspace command.

        This is used for code actions that return a command instead of (or
        in addition to) a workspace edit.

        Args:
            command: The command identifier
            arguments: Optional arguments for the command

        Returns:
            The result of the command execution (varies by command).
        """
        params: Dict[str, Any] = {"command": command}
        if arguments:
            params["arguments"] = arguments

        result = await self._send_request("workspace/executeCommand", params)
        return result
