"""Prompt library plugin for managing reusable prompts.

This plugin provides:
- User command: `prompt` for listing and using prompts
- Model tools: listPrompts, usePrompt, savePrompt

Storage locations (in priority order):
1. .jaato/prompts/ (project)
2. ~/.jaato/prompts/ (global)
3. .claude/skills/ (Claude Code interop, read-only)
4. .claude/commands/ (Claude Code legacy, read-only)
5. ~/.claude/skills/ (Claude Code global, read-only)

Prompt format:
- Single file: .jaato/prompts/review.md
- Directory with supporting files: .jaato/prompts/api-design/PROMPT.md

Template syntax:
- {{name}} - Named parameter
- {{name:default}} - Named parameter with default value
- {{$1}}, {{$2}}, etc. - Positional parameters (for Claude compat)
- {{$0}} - All arguments joined with spaces
- $ARGUMENTS - Claude Code compatibility (becomes {{$0}})
"""

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from ..permission.plugin import PermissionPlugin

from ..model_provider.types import ToolSchema
from ..base import UserCommand, CommandCompletion


# Entry point filename for directory-based prompts
PROMPT_ENTRY_FILE = "PROMPT.md"
# Claude Code entry point
SKILL_ENTRY_FILE = "SKILL.md"

# Maximum file size to read (100KB)
MAX_PROMPT_FILE_SIZE = 100_000

# Template patterns
# Named params: {{name}} or {{name:default}}
NAMED_PARAM_PATTERN = re.compile(r'\{\{([a-zA-Z_][a-zA-Z0-9_]*)(?::([^}]*))?\}\}')
# Positional params: {{$1}}, {{$2}}, {{$0}} (for Claude compat)
POSITIONAL_PARAM_PATTERN = re.compile(r'\{\{\$(\d+)(?::([^}]*))?\}\}')
# Claude Code $ARGUMENTS placeholder
ARGUMENTS_PLACEHOLDER = re.compile(r'\$ARGUMENTS\b')


@dataclass
class PromptParam:
    """Definition of a prompt parameter."""
    name: str
    required: bool = True
    default: Optional[str] = None
    description: str = ""
    enum: Optional[List[str]] = None


@dataclass
class PromptInfo:
    """Information about a prompt in the library."""
    name: str
    description: str
    source: str  # "project", "global", "claude-skills", "claude-commands", "claude-global"
    path: Path
    is_directory: bool = False
    tags: List[str] = field(default_factory=list)
    params: Dict[str, PromptParam] = field(default_factory=dict)


@dataclass
class PromptSource:
    """A source location for prompts."""
    path: Path
    source_name: str
    entry_file: str  # PROMPT.md or SKILL.md
    writable: bool = False


@dataclass
class FetchResult:
    """Result of a fetch operation."""
    success: bool
    prompts_fetched: List[str] = field(default_factory=list)
    error: Optional[str] = None
    source_type: str = ""
    source_params: str = ""


class PromptLibraryPlugin:
    """Plugin that provides prompt library functionality.

    Users can:
    - Type `prompt` to list available prompts
    - Type `prompt <name> [args...]` to use a prompt

    Model can:
    - Call listPrompts() to discover available prompts
    - Call usePrompt(name, params) to retrieve and expand a prompt
    - Call savePrompt(name, content, description) to create new prompts
    """

    def __init__(self):
        self._workspace_path: Optional[str] = None
        self._initialized = False
        self._agent_name: Optional[str] = None
        # Cache of discovered prompts (refreshed on each list)
        self._prompt_cache: Dict[str, PromptInfo] = {}
        # Confirmation callback for interactive prompts (set by client)
        self._confirm_callback: Optional[Callable[[str, List[str]], Optional[str]]] = None
        # Output callback for progress messages
        self._output_callback: Optional[Callable[[str], None]] = None

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        trace_path = os.environ.get(
            'JAATO_TRACE_LOG',
            os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
        )
        if trace_path:
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    agent_prefix = f"@{self._agent_name}" if self._agent_name else ""
                    f.write(f"[{ts}] [PROMPT_LIBRARY{agent_prefix}] {msg}\n")
                    f.flush()
            except (IOError, OSError):
                pass

    @property
    def name(self) -> str:
        return "prompt_library"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the prompt library plugin.

        Args:
            config: Optional dict with:
                - workspace_path: Override workspace path
                - agent_name: Agent name for trace logging
        """
        config = config or {}
        self._agent_name = config.get("agent_name")
        if 'workspace_path' in config:
            self._workspace_path = config['workspace_path']
        self._initialized = True
        self._trace(f"initialize: workspace={self._workspace_path}")

    def shutdown(self) -> None:
        """Shutdown the prompt library plugin."""
        self._trace("shutdown")
        self._initialized = False
        self._prompt_cache.clear()

    def set_workspace_path(self, path: str) -> None:
        """Set the workspace root path (called by registry)."""
        self._workspace_path = path
        self._trace(f"set_workspace_path: {path}")

    def set_confirm_callback(
        self,
        callback: Callable[[str, List[str]], Optional[str]]
    ) -> None:
        """Set callback for interactive confirmation prompts.

        Args:
            callback: Function that takes (message, options) and returns
                     the selected option or None if cancelled.
        """
        self._confirm_callback = callback

    def set_output_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for output/progress messages."""
        self._output_callback = callback

    def _emit(self, message: str) -> None:
        """Emit a message via output callback or print."""
        if self._output_callback:
            self._output_callback(message)
        else:
            print(message)

    def _confirm(self, message: str, options: List[str]) -> Optional[str]:
        """Ask user for confirmation.

        Returns selected option or None if cancelled/no callback.
        """
        if self._confirm_callback:
            return self._confirm_callback(message, options)
        # No callback - default to first option (usually 'yes' or safe default)
        return options[0] if options else None

    def _get_workspace(self) -> Path:
        """Get the current workspace path."""
        if self._workspace_path:
            return Path(self._workspace_path)
        return Path.cwd()

    def _get_prompt_sources(self) -> List[PromptSource]:
        """Get all prompt source locations in priority order."""
        workspace = self._get_workspace()
        home = Path.home()

        sources = [
            # Jaato native (writable)
            PromptSource(
                path=workspace / ".jaato" / "prompts",
                source_name="project",
                entry_file=PROMPT_ENTRY_FILE,
                writable=True,
            ),
            PromptSource(
                path=home / ".jaato" / "prompts",
                source_name="global",
                entry_file=PROMPT_ENTRY_FILE,
                writable=True,
            ),
            # Claude Code interop (read-only)
            PromptSource(
                path=workspace / ".claude" / "skills",
                source_name="claude-skills",
                entry_file=SKILL_ENTRY_FILE,
                writable=False,
            ),
            PromptSource(
                path=workspace / ".claude" / "commands",
                source_name="claude-commands",
                entry_file=SKILL_ENTRY_FILE,  # or plain files
                writable=False,
            ),
            PromptSource(
                path=home / ".claude" / "skills",
                source_name="claude-global",
                entry_file=SKILL_ENTRY_FILE,
                writable=False,
            ),
        ]

        return sources

    def _parse_frontmatter(self, content: str) -> tuple[Dict[str, Any], str]:
        """Parse YAML frontmatter from markdown content.

        Returns:
            Tuple of (frontmatter_dict, remaining_content)
        """
        if not content.startswith('---'):
            return {}, content

        # Find the closing ---
        end_match = re.search(r'\n---\s*\n', content[3:])
        if not end_match:
            return {}, content

        frontmatter_text = content[3:end_match.start() + 3]
        body = content[end_match.end() + 3:]

        try:
            frontmatter = yaml.safe_load(frontmatter_text) or {}
            return frontmatter, body.lstrip()
        except yaml.YAMLError:
            self._trace(f"Failed to parse frontmatter: {frontmatter_text[:100]}")
            return {}, content

    def _extract_description(self, content: str, frontmatter: Dict[str, Any]) -> str:
        """Extract description from frontmatter or first line of content."""
        # Try frontmatter first
        if 'description' in frontmatter:
            return frontmatter['description']

        # Fall back to first line (strip comment prefix if present)
        first_line = content.split('\n')[0].strip()
        if first_line.startswith('#'):
            # Remove markdown heading prefix
            first_line = first_line.lstrip('#').strip()
        elif first_line.startswith('//') or first_line.startswith('--'):
            first_line = first_line[2:].strip()

        return first_line[:100] if first_line else "No description"

    def _extract_params(self, content: str, frontmatter: Dict[str, Any]) -> Dict[str, PromptParam]:
        """Extract parameter definitions from frontmatter and content."""
        params: Dict[str, PromptParam] = {}

        # From frontmatter params section
        if 'params' in frontmatter:
            for name, spec in frontmatter['params'].items():
                if isinstance(spec, dict):
                    params[name] = PromptParam(
                        name=name,
                        required=spec.get('required', True),
                        default=spec.get('default'),
                        description=spec.get('description', ''),
                        enum=spec.get('enum'),
                    )
                else:
                    # Simple value is the default
                    params[name] = PromptParam(
                        name=name,
                        required=False,
                        default=str(spec),
                    )

        # Discover named params from content: {{name}} or {{name:default}}
        for match in NAMED_PARAM_PATTERN.finditer(content):
            name = match.group(1)
            default = match.group(2)
            if name not in params:
                params[name] = PromptParam(
                    name=name,
                    required=default is None,
                    default=default,
                )

        return params

    def _load_prompt_info(self, path: Path, source_name: str, entry_file: str) -> Optional[PromptInfo]:
        """Load prompt info from a file or directory."""
        try:
            if path.is_dir():
                # Directory-based prompt
                entry_path = path / entry_file
                # Also try SKILL.md for Claude compat
                if not entry_path.exists() and entry_file == PROMPT_ENTRY_FILE:
                    entry_path = path / SKILL_ENTRY_FILE
                # Also try just the directory name as a file (Claude commands)
                if not entry_path.exists():
                    return None

                content = entry_path.read_text(encoding='utf-8')
                frontmatter, body = self._parse_frontmatter(content)

                return PromptInfo(
                    name=frontmatter.get('name', path.name),
                    description=self._extract_description(body, frontmatter),
                    source=source_name,
                    path=path,
                    is_directory=True,
                    tags=frontmatter.get('tags', []),
                    params=self._extract_params(body, frontmatter),
                )

            elif path.is_file():
                # Check file size
                if path.stat().st_size > MAX_PROMPT_FILE_SIZE:
                    self._trace(f"Prompt file too large: {path}")
                    return None

                content = path.read_text(encoding='utf-8')
                frontmatter, body = self._parse_frontmatter(content)

                # Determine name (file stem, or from frontmatter)
                name = frontmatter.get('name', path.stem)

                return PromptInfo(
                    name=name,
                    description=self._extract_description(body, frontmatter),
                    source=source_name,
                    path=path,
                    is_directory=False,
                    tags=frontmatter.get('tags', []),
                    params=self._extract_params(body, frontmatter),
                )

        except Exception as e:
            self._trace(f"Error loading prompt from {path}: {e}")

        return None

    def _discover_prompts(self) -> Dict[str, PromptInfo]:
        """Discover all available prompts from all sources."""
        prompts: Dict[str, PromptInfo] = {}

        for source in self._get_prompt_sources():
            if not source.path.exists():
                continue

            self._trace(f"Scanning source: {source.path}")

            for item in source.path.iterdir():
                # Skip hidden files
                if item.name.startswith('.'):
                    continue

                info = self._load_prompt_info(item, source.source_name, source.entry_file)
                if info and info.name not in prompts:
                    # First source wins (priority order)
                    prompts[info.name] = info
                    self._trace(f"Discovered prompt: {info.name} from {info.source}")

        self._prompt_cache = prompts
        return prompts

    def _substitute_params(
        self,
        content: str,
        named_params: Dict[str, str],
        positional_args: List[str]
    ) -> tuple[str, List[str]]:
        """Substitute parameters in prompt content.

        Args:
            content: The prompt content with template variables
            named_params: Dict of named parameter values
            positional_args: List of positional arguments

        Returns:
            Tuple of (substituted_content, list of missing required params)
        """
        missing_params = []

        # Handle Claude Code $ARGUMENTS placeholder
        content = ARGUMENTS_PLACEHOLDER.sub('{{$0}}', content)

        # Substitute named params: {{name}} or {{name:default}}
        def replace_named(match: re.Match) -> str:
            name = match.group(1)
            default = match.group(2)

            if name in named_params:
                return named_params[name]
            elif default is not None:
                return default
            else:
                missing_params.append(name)
                return f'{{{{{name}}}}}'  # Leave unreplaced

        content = NAMED_PARAM_PATTERN.sub(replace_named, content)

        # Substitute positional params: {{$1}}, {{$2}}, {{$0}}
        def replace_positional(match: re.Match) -> str:
            index = int(match.group(1))
            default = match.group(2)

            if index == 0:
                # {{$0}} - all arguments joined
                return ' '.join(positional_args) if positional_args else (default or '')

            # 1-indexed positional parameter
            if index <= len(positional_args):
                return positional_args[index - 1]
            elif default is not None:
                return default
            else:
                missing_params.append(f'${index}')
                return f'{{{{${index}}}}}'  # Leave unreplaced

        content = POSITIONAL_PARAM_PATTERN.sub(replace_positional, content)

        return content, missing_params

    def _get_prompt_content(self, info: PromptInfo) -> str:
        """Get the raw content of a prompt."""
        if info.is_directory:
            # Read entry file from directory
            entry_path = info.path / PROMPT_ENTRY_FILE
            if not entry_path.exists():
                entry_path = info.path / SKILL_ENTRY_FILE
            if not entry_path.exists():
                raise FileNotFoundError(f"No entry file found in {info.path}")
            content = entry_path.read_text(encoding='utf-8')
        else:
            content = info.path.read_text(encoding='utf-8')

        # Strip frontmatter for execution
        _, body = self._parse_frontmatter(content)
        return body

    # ==================== Fetch Implementation ====================

    def _get_destination_dir(self, destination: str) -> Path:
        """Get the destination directory for fetched prompts."""
        if destination == "user":
            return Path.home() / ".jaato" / "prompts"
        else:  # project
            return self._get_workspace() / ".jaato" / "prompts"

    def _add_provenance(
        self,
        content: str,
        source_type: str,
        source_params: str
    ) -> str:
        """Add or update provenance metadata in prompt frontmatter."""
        frontmatter, body = self._parse_frontmatter(content)

        # Add provenance fields
        frontmatter['fetched_from'] = f"{source_type} {source_params}"
        frontmatter['fetched_at'] = datetime.now().isoformat()

        # Rebuild content with updated frontmatter
        fm_lines = ['---']
        for key, value in frontmatter.items():
            if isinstance(value, list):
                fm_lines.append(f"{key}: {value}")
            elif isinstance(value, bool):
                fm_lines.append(f"{key}: {'true' if value else 'false'}")
            else:
                # Quote strings that might need it
                if isinstance(value, str) and (':' in value or '\n' in value):
                    fm_lines.append(f'{key}: "{value}"')
                else:
                    fm_lines.append(f"{key}: {value}")
        fm_lines.append('---')
        fm_lines.append('')

        return '\n'.join(fm_lines) + body

    def _handle_conflict(
        self,
        name: str,
        dest_path: Path,
        new_content: str
    ) -> tuple[bool, Optional[Path]]:
        """Handle conflict when prompt already exists.

        Returns:
            Tuple of (should_proceed, actual_path_to_use)
            - (True, path) - proceed with this path
            - (False, None) - skip this prompt
        """
        if not dest_path.exists():
            return True, dest_path

        # Ask for confirmation
        message = f"Prompt '{name}' already exists at {dest_path}. What would you like to do?"
        options = ["overwrite", "rename", "skip"]

        choice = self._confirm(message, options)

        if choice == "overwrite":
            return True, dest_path
        elif choice == "rename":
            # Generate unique name
            counter = 1
            while True:
                new_name = f"{name}-{counter}"
                new_path = dest_path.parent / f"{new_name}.md"
                if not new_path.exists():
                    self._emit(f"Renamed to: {new_name}")
                    return True, new_path
                counter += 1
        else:  # skip
            return False, None

    def _fetch_from_npx(
        self,
        package: str,
        args: List[str],
        dest_dir: Path
    ) -> FetchResult:
        """Fetch prompts by running an npx command.

        The npx command should output prompt files to a temp directory,
        which we then copy to the destination.
        """
        self._trace(f"fetch npx: {package} {args}")

        # Create temp directory for npx output
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Run npx command
            cmd = ["npx", "-y", package] + args + ["--output", str(tmpdir_path)]
            self._emit(f"Running: npx {package} {' '.join(args)}...")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,  # 2 minute timeout
                    cwd=str(self._get_workspace())
                )

                if result.returncode != 0:
                    error_msg = result.stderr or result.stdout or "Unknown error"
                    return FetchResult(
                        success=False,
                        error=f"npx command failed: {error_msg}",
                        source_type="npx",
                        source_params=f"{package} {' '.join(args)}"
                    )

                # Look for prompt files in temp directory
                prompts_fetched = []
                for item in tmpdir_path.iterdir():
                    if item.is_file() and item.suffix in ('.md', '.yaml', '.yml'):
                        # Read and add provenance
                        content = item.read_text(encoding='utf-8')
                        content = self._add_provenance(
                            content, "npx", f"{package} {' '.join(args)}"
                        )

                        # Handle conflict
                        dest_path = dest_dir / item.name
                        proceed, actual_path = self._handle_conflict(
                            item.stem, dest_path, content
                        )

                        if proceed and actual_path:
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            actual_path.write_text(content, encoding='utf-8')
                            prompts_fetched.append(actual_path.stem)

                    elif item.is_dir():
                        # Directory-based prompt
                        entry_file = item / PROMPT_ENTRY_FILE
                        if not entry_file.exists():
                            entry_file = item / SKILL_ENTRY_FILE

                        if entry_file.exists():
                            # Handle conflict
                            dest_path = dest_dir / item.name
                            proceed, actual_path = self._handle_conflict(
                                item.name, dest_path, ""
                            )

                            if proceed and actual_path:
                                # Copy entire directory
                                if actual_path.exists():
                                    shutil.rmtree(actual_path)
                                shutil.copytree(item, actual_path)

                                # Add provenance to entry file
                                entry_in_dest = actual_path / entry_file.name
                                content = entry_in_dest.read_text(encoding='utf-8')
                                content = self._add_provenance(
                                    content, "npx", f"{package} {' '.join(args)}"
                                )
                                entry_in_dest.write_text(content, encoding='utf-8')
                                prompts_fetched.append(actual_path.name)

                return FetchResult(
                    success=True,
                    prompts_fetched=prompts_fetched,
                    source_type="npx",
                    source_params=f"{package} {' '.join(args)}"
                )

            except subprocess.TimeoutExpired:
                return FetchResult(
                    success=False,
                    error="npx command timed out after 120 seconds",
                    source_type="npx",
                    source_params=f"{package} {' '.join(args)}"
                )
            except FileNotFoundError:
                return FetchResult(
                    success=False,
                    error="npx not found. Please install Node.js.",
                    source_type="npx",
                    source_params=f"{package} {' '.join(args)}"
                )
            except Exception as e:
                return FetchResult(
                    success=False,
                    error=str(e),
                    source_type="npx",
                    source_params=f"{package} {' '.join(args)}"
                )

    def _fetch_from_git(self, repo_url: str, dest_dir: Path) -> FetchResult:
        """Fetch prompts from a git repository."""
        self._trace(f"fetch git: {repo_url}")
        self._emit(f"Cloning: {repo_url}...")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            try:
                # Clone the repository
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", repo_url, str(tmpdir_path / "repo")],
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode != 0:
                    return FetchResult(
                        success=False,
                        error=f"git clone failed: {result.stderr}",
                        source_type="git",
                        source_params=repo_url
                    )

                repo_path = tmpdir_path / "repo"

                # Look for prompts in various locations
                prompt_dirs = [
                    repo_path / ".jaato" / "prompts",
                    repo_path / ".claude" / "skills",
                    repo_path / ".claude" / "commands",
                    repo_path / "prompts",
                    repo_path / "skills",
                ]

                prompts_fetched = []
                for prompt_dir in prompt_dirs:
                    if not prompt_dir.exists():
                        continue

                    for item in prompt_dir.iterdir():
                        if item.name.startswith('.'):
                            continue

                        if item.is_file() and item.suffix in ('.md', '.yaml', '.yml'):
                            content = item.read_text(encoding='utf-8')
                            content = self._add_provenance(content, "git", repo_url)

                            dest_path = dest_dir / item.name
                            proceed, actual_path = self._handle_conflict(
                                item.stem, dest_path, content
                            )

                            if proceed and actual_path:
                                dest_dir.mkdir(parents=True, exist_ok=True)
                                actual_path.write_text(content, encoding='utf-8')
                                prompts_fetched.append(actual_path.stem)

                        elif item.is_dir():
                            entry_file = None
                            for ef in [PROMPT_ENTRY_FILE, SKILL_ENTRY_FILE]:
                                if (item / ef).exists():
                                    entry_file = item / ef
                                    break

                            if entry_file:
                                dest_path = dest_dir / item.name
                                proceed, actual_path = self._handle_conflict(
                                    item.name, dest_path, ""
                                )

                                if proceed and actual_path:
                                    if actual_path.exists():
                                        shutil.rmtree(actual_path)
                                    shutil.copytree(item, actual_path)

                                    # Add provenance
                                    for ef in [PROMPT_ENTRY_FILE, SKILL_ENTRY_FILE]:
                                        entry_in_dest = actual_path / ef
                                        if entry_in_dest.exists():
                                            content = entry_in_dest.read_text(encoding='utf-8')
                                            content = self._add_provenance(content, "git", repo_url)
                                            entry_in_dest.write_text(content, encoding='utf-8')
                                            break

                                    prompts_fetched.append(actual_path.name)

                if not prompts_fetched:
                    return FetchResult(
                        success=False,
                        error="No prompts found in repository",
                        source_type="git",
                        source_params=repo_url
                    )

                return FetchResult(
                    success=True,
                    prompts_fetched=prompts_fetched,
                    source_type="git",
                    source_params=repo_url
                )

            except subprocess.TimeoutExpired:
                return FetchResult(
                    success=False,
                    error="git clone timed out after 120 seconds",
                    source_type="git",
                    source_params=repo_url
                )
            except FileNotFoundError:
                return FetchResult(
                    success=False,
                    error="git not found. Please install git.",
                    source_type="git",
                    source_params=repo_url
                )
            except Exception as e:
                return FetchResult(
                    success=False,
                    error=str(e),
                    source_type="git",
                    source_params=repo_url
                )

    def _fetch_from_url(self, url: str, dest_dir: Path) -> FetchResult:
        """Fetch a single prompt from a URL."""
        self._trace(f"fetch url: {url}")
        self._emit(f"Fetching: {url}...")

        try:
            import urllib.request
            import urllib.error

            # Determine filename from URL
            from urllib.parse import urlparse
            parsed = urlparse(url)
            filename = Path(parsed.path).name
            if not filename or not filename.endswith('.md'):
                filename = "fetched-prompt.md"

            # Fetch the content
            req = urllib.request.Request(url, headers={'User-Agent': 'jaato-prompt-fetch/1.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read().decode('utf-8')

            # Add provenance
            content = self._add_provenance(content, "url", url)

            # Handle conflict
            name = Path(filename).stem
            dest_path = dest_dir / filename
            proceed, actual_path = self._handle_conflict(name, dest_path, content)

            if proceed and actual_path:
                dest_dir.mkdir(parents=True, exist_ok=True)
                actual_path.write_text(content, encoding='utf-8')
                return FetchResult(
                    success=True,
                    prompts_fetched=[actual_path.stem],
                    source_type="url",
                    source_params=url
                )
            else:
                return FetchResult(
                    success=True,
                    prompts_fetched=[],
                    source_type="url",
                    source_params=url
                )

        except urllib.error.HTTPError as e:
            return FetchResult(
                success=False,
                error=f"HTTP error: {e.code} {e.reason}",
                source_type="url",
                source_params=url
            )
        except urllib.error.URLError as e:
            return FetchResult(
                success=False,
                error=f"URL error: {e.reason}",
                source_type="url",
                source_params=url
            )
        except Exception as e:
            return FetchResult(
                success=False,
                error=str(e),
                source_type="url",
                source_params=url
            )

    def _fetch_from_github(self, repo_spec: str, dest_dir: Path) -> FetchResult:
        """Fetch prompts from a GitHub repository (shorthand).

        repo_spec can be:
        - owner/repo
        - owner/repo/path/to/prompts
        """
        self._trace(f"fetch github: {repo_spec}")

        parts = repo_spec.split('/')
        if len(parts) < 2:
            return FetchResult(
                success=False,
                error="Invalid GitHub repo spec. Use: owner/repo or owner/repo/path",
                source_type="github",
                source_params=repo_spec
            )

        owner = parts[0]
        repo = parts[1]
        repo_url = f"https://github.com/{owner}/{repo}.git"

        return self._fetch_from_git(repo_url, dest_dir)

    def _execute_fetch(
        self,
        source_type: str,
        source_params: List[str],
        destination: str = "project"
    ) -> str:
        """Execute the fetch subcommand.

        Args:
            source_type: One of 'npx', 'git', 'github', 'url'
            source_params: Parameters for the source
            destination: 'project' or 'user'

        Returns:
            Result message string
        """
        self._trace(f"fetch: type={source_type}, params={source_params}, dest={destination}")

        if not source_params:
            return f"Error: No source parameters provided for '{source_type}'"

        dest_dir = self._get_destination_dir(destination)

        # Execute based on source type
        if source_type == "npx":
            if len(source_params) < 1:
                return "Error: npx requires at least a package name"
            package = source_params[0]
            args = source_params[1:]
            result = self._fetch_from_npx(package, args, dest_dir)

        elif source_type == "git":
            repo_url = source_params[0]
            result = self._fetch_from_git(repo_url, dest_dir)

        elif source_type == "github":
            repo_spec = source_params[0]
            result = self._fetch_from_github(repo_spec, dest_dir)

        elif source_type == "url":
            url = source_params[0]
            result = self._fetch_from_url(url, dest_dir)

        else:
            return f"Error: Unknown source type '{source_type}'. Supported: npx, git, github, url"

        # Format result message
        if result.success:
            if result.prompts_fetched:
                prompts_list = ", ".join(result.prompts_fetched)
                return f"Fetched {len(result.prompts_fetched)} prompt(s): {prompts_list}\nDestination: {dest_dir}"
            else:
                return f"No prompts were fetched (all skipped or no prompts found)"
        else:
            return f"Fetch failed: {result.error}"

    # ==================== Model Tools ====================

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for model access."""
        return [
            ToolSchema(
                name='listPrompts',
                description=(
                    'List available prompts from the prompt library.\n\n'
                    'Returns prompts from:\n'
                    '- .jaato/prompts/ (project)\n'
                    '- ~/.jaato/prompts/ (global)\n'
                    '- .claude/skills/ (Claude Code interop)\n\n'
                    'Use this to discover reusable prompts before using them.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "tag": {
                            "type": "string",
                            "description": "Filter prompts by tag"
                        },
                        "search": {
                            "type": "string",
                            "description": "Search prompts by name or description"
                        }
                    },
                    "required": []
                },
                category="introspection",
                discoverability="discoverable",
            ),
            ToolSchema(
                name='usePrompt',
                description=(
                    'Retrieve and expand a prompt from the library.\n\n'
                    'The prompt content is returned with parameters substituted.\n'
                    'Follow the instructions in the returned content.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the prompt to use"
                        },
                        "params": {
                            "type": "object",
                            "description": "Named parameters to substitute (e.g., {\"file\": \"main.py\"})",
                            "additionalProperties": {"type": "string"}
                        },
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Positional arguments for {{$1}}, {{$2}}, etc."
                        }
                    },
                    "required": ["name"]
                },
                category="system",
                discoverability="discoverable",
            ),
            ToolSchema(
                name='savePrompt',
                description=(
                    'Save a new prompt to the library.\n\n'
                    'Use this when you notice the user performing repetitive tasks '
                    'that could be captured as a reusable prompt.\n\n'
                    'Prompts are saved to .jaato/prompts/ in the project.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name for the prompt (lowercase, hyphens allowed)"
                        },
                        "content": {
                            "type": "string",
                            "description": "The prompt content with optional {{param}} placeholders"
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief description of what the prompt does"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for categorization"
                        },
                        "global": {
                            "type": "boolean",
                            "description": "Save to ~/.jaato/prompts/ instead of project"
                        }
                    },
                    "required": ["name", "content", "description"]
                },
                category="system",
                discoverability="discoverable",
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executor mapping for model tools and user commands."""
        return {
            'listPrompts': self._execute_list_prompts,
            'usePrompt': self._execute_use_prompt,
            'savePrompt': self._execute_save_prompt,
            'prompt': self._execute_prompt_command,
        }

    def _execute_list_prompts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute listPrompts tool."""
        tag_filter = args.get('tag')
        search_query = args.get('search', '').lower()

        self._trace(f"listPrompts: tag={tag_filter}, search={search_query}")

        prompts = self._discover_prompts()

        results = []
        for info in prompts.values():
            # Apply tag filter
            if tag_filter and tag_filter not in info.tags:
                continue

            # Apply search filter
            if search_query:
                if (search_query not in info.name.lower() and
                    search_query not in info.description.lower()):
                    continue

            results.append({
                "name": info.name,
                "description": info.description,
                "source": info.source,
                "tags": info.tags,
                "params": {
                    name: {
                        "required": p.required,
                        "default": p.default,
                        "description": p.description,
                    }
                    for name, p in info.params.items()
                } if info.params else None,
            })

        return {
            "prompts": results,
            "total": len(results),
        }

    def _execute_use_prompt(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute usePrompt tool."""
        name = args.get('name', '').strip()
        named_params = args.get('params', {}) or {}
        positional_args = args.get('args', []) or []

        self._trace(f"usePrompt: name={name}, params={named_params}, args={positional_args}")

        if not name:
            return {
                'error': 'No prompt name provided',
                'hint': 'Use listPrompts() to see available prompts'
            }

        # Refresh cache and find prompt
        prompts = self._discover_prompts()

        if name not in prompts:
            available = list(prompts.keys())
            return {
                'error': f'Prompt not found: {name}',
                'available': available[:10],
                'hint': f'Available prompts: {", ".join(available[:5])}{"..." if len(available) > 5 else ""}'
            }

        info = prompts[name]

        try:
            content = self._get_prompt_content(info)
            substituted, missing = self._substitute_params(content, named_params, positional_args)

            result = {
                'name': name,
                'content': substituted,
                'source': info.source,
                'path': str(info.path),
            }

            if missing:
                result['missing_params'] = missing
                result['hint'] = f'Some parameters were not provided: {", ".join(missing)}'

            return result

        except Exception as e:
            return {
                'error': f'Failed to read prompt: {e}',
                'name': name
            }

    def _execute_save_prompt(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute savePrompt tool."""
        name = args.get('name', '').strip()
        content = args.get('content', '').strip()
        description = args.get('description', '').strip()
        tags = args.get('tags', [])
        save_global = args.get('global', False)

        self._trace(f"savePrompt: name={name}, global={save_global}")

        if not name:
            return {'error': 'No prompt name provided'}
        if not content:
            return {'error': 'No prompt content provided'}
        if not description:
            return {'error': 'No description provided'}

        # Validate name (lowercase, hyphens, underscores)
        if not re.match(r'^[a-z][a-z0-9_-]*$', name):
            return {
                'error': 'Invalid prompt name',
                'hint': 'Use lowercase letters, numbers, hyphens, and underscores. Must start with a letter.'
            }

        # Determine save location
        if save_global:
            prompts_dir = Path.home() / ".jaato" / "prompts"
        else:
            prompts_dir = self._get_workspace() / ".jaato" / "prompts"

        # Create directory if needed
        prompts_dir.mkdir(parents=True, exist_ok=True)

        # Build file content with frontmatter
        file_content = f"""---
description: {description}
"""
        if tags:
            file_content += f"tags: {tags}\n"
        file_content += f"""---

{content}
"""

        # Save file
        prompt_path = prompts_dir / f"{name}.md"

        # Check if exists
        if prompt_path.exists():
            return {
                'error': f'Prompt already exists: {name}',
                'path': str(prompt_path),
                'hint': 'Delete the existing prompt first or choose a different name'
            }

        try:
            prompt_path.write_text(file_content, encoding='utf-8')
            self._trace(f"Saved prompt to: {prompt_path}")

            return {
                'success': True,
                'name': name,
                'path': str(prompt_path),
                'message': f'Prompt "{name}" saved. Use `prompt {name}` or usePrompt("{name}") to invoke it.'
            }
        except Exception as e:
            return {
                'error': f'Failed to save prompt: {e}',
                'name': name
            }

    # ==================== User Commands ====================

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands."""
        return [
            UserCommand(
                name="prompt",
                description="Use a prompt from the library: prompt <name> [args...]",
                share_with_model=True,  # Output goes to model for execution
            )
        ]

    def get_command_completions(
        self, command: str, args: List[str]
    ) -> List[CommandCompletion]:
        """Return completion options for prompt command arguments."""
        if command != "prompt":
            return []

        # First argument: prompt name or subcommand
        if len(args) <= 1:
            prefix = args[0].lower() if args else ""
            completions = []

            # Add 'fetch' subcommand
            if not prefix or "fetch".startswith(prefix):
                completions.append(CommandCompletion(
                    "fetch",
                    "Fetch prompts from external source"
                ))

            # Add available prompts
            prompts = self._discover_prompts()
            for name, info in sorted(prompts.items()):
                if prefix and not name.lower().startswith(prefix):
                    continue
                completions.append(CommandCompletion(
                    name,
                    info.description[:60] + ("..." if len(info.description) > 60 else "")
                ))

            return completions

        # Handle 'fetch' subcommand completions
        if args[0].lower() == "fetch":
            if len(args) == 2:
                # Source type completions
                prefix = args[1].lower() if len(args) > 1 else ""
                source_types = [
                    ("npx", "Run npx command to fetch prompts"),
                    ("git", "Clone git repository"),
                    ("github", "Fetch from GitHub (owner/repo)"),
                    ("url", "Fetch single prompt from URL"),
                ]
                completions = []
                for name, desc in source_types:
                    if not prefix or name.startswith(prefix):
                        completions.append(CommandCompletion(name, desc))
                return completions

            # Show destination options as last argument
            last_arg = args[-1].lower() if args else ""
            if last_arg in ("", "p", "pr", "pro", "proj", "proje", "projec", "project",
                           "u", "us", "use", "user"):
                options = [
                    ("project", "Save to .jaato/prompts/ (default)"),
                    ("user", "Save to ~/.jaato/prompts/"),
                ]
                completions = []
                for name, desc in options:
                    if not last_arg or name.startswith(last_arg):
                        completions.append(CommandCompletion(name, desc))
                return completions

        # No specific completion for other arguments
        return []

    def execute_user_command(self, command: str, args: Dict[str, Any]) -> str:
        """Execute a user command and return the result as a string.

        For the 'prompt' command:
        - With no args: lists available prompts
        - With args: uses the specified prompt with given arguments

        Args:
            command: The command name (should be 'prompt')
            args: Parsed arguments containing positional args list

        Returns:
            String result - either the prompt content or a listing.
        """
        if command != "prompt":
            return f"Unknown command: {command}"

        # Extract args - could be a list or have 'args' key
        positional_args = args.get('args', [])
        if isinstance(positional_args, str):
            positional_args = positional_args.split() if positional_args else []

        return self._execute_prompt_command({'args': positional_args})

    def _execute_prompt_command(self, args: Dict[str, Any]) -> str:
        """Execute the prompt user command.

        Args:
            args: Dict with 'args' key containing list of positional arguments.
                  First arg is subcommand or prompt name.

        Subcommands:
            prompt                              - List available prompts
            prompt <name> [args...]             - Use a prompt
            prompt fetch <type> <params...>     - Fetch prompts from external source
            prompt fetch <type> <params...> user  - Fetch to user directory

        Returns:
            String with prompt content or listing.
        """
        positional_args = args.get('args', []) or []

        # No args - list available prompts
        if not positional_args:
            return self._list_prompts_formatted()

        first_arg = positional_args[0].lower()

        # Handle 'fetch' subcommand
        if first_arg == "fetch":
            return self._handle_fetch_subcommand(positional_args[1:])

        # Otherwise, first arg is prompt name
        prompt_name = positional_args[0]
        prompt_args = positional_args[1:]

        # Use the usePrompt executor
        result = self._execute_use_prompt({
            'name': prompt_name,
            'args': prompt_args,
        })

        if 'error' in result:
            error_msg = result['error']
            if 'hint' in result:
                error_msg += f"\n{result['hint']}"
            return error_msg

        # Return the content for the model to process
        content = result.get('content', '')
        if result.get('missing_params'):
            content += f"\n\n[Note: Missing parameters: {', '.join(result['missing_params'])}]"

        return content

    def _list_prompts_formatted(self) -> str:
        """List available prompts in formatted output."""
        prompts = self._discover_prompts()
        if not prompts:
            return "No prompts available.\n\nCreate prompts in .jaato/prompts/ or ~/.jaato/prompts/"

        lines = ["Available prompts:\n"]
        for name, info in sorted(prompts.items()):
            source_tag = f" [{info.source}]" if info.source != "project" else ""
            lines.append(f"  {name}{source_tag}")
            lines.append(f"    {info.description[:70]}")
        lines.append(f"\nUse: prompt <name> [args...]")
        lines.append("     prompt fetch <type> <params...> [user]")
        return "\n".join(lines)

    def _handle_fetch_subcommand(self, args: List[str]) -> str:
        """Handle the 'prompt fetch' subcommand.

        Args:
            args: Arguments after 'fetch', e.g., ['npx', 'molthub@latest', 'moltbook']

        Syntax:
            prompt fetch <source_type> <source_params...> [project|user]

        Examples:
            prompt fetch npx molthub@latest moltbook
            prompt fetch git https://github.com/user/prompts
            prompt fetch github user/prompts
            prompt fetch url https://example.com/review.md
            prompt fetch npx some-package user
        """
        if not args:
            return """Usage: prompt fetch <source_type> <source_params...> [project|user]

Source types:
  npx <package> [args...]   - Run npx command to fetch prompts
  git <repo_url>            - Clone git repository
  github <owner/repo>       - Fetch from GitHub (shorthand)
  url <url>                 - Fetch single prompt from URL

Destination (optional, default: project):
  project   Save to .jaato/prompts/
  user      Save to ~/.jaato/prompts/

Examples:
  prompt fetch npx molthub@latest moltbook
  prompt fetch github anthropics/prompt-library
  prompt fetch url https://example.com/review.md user"""

        # Parse destination - check if last arg is 'project' or 'user'
        destination = "project"
        source_params = []
        for arg in args[1:]:
            if arg.lower() == "user":
                destination = "user"
            elif arg.lower() == "project":
                destination = "project"
            else:
                source_params.append(arg)

        source_type = args[0].lower()

        # Validate source type
        valid_sources = ["npx", "git", "github", "url"]
        if source_type not in valid_sources:
            return f"Error: Unknown source type '{source_type}'. Supported: {', '.join(valid_sources)}"

        if not source_params:
            return f"Error: No source parameters provided for '{source_type}'"

        # Execute fetch with permission check
        return self._execute_fetch(source_type, source_params, destination)

    def get_auto_approved_tools(self) -> List[str]:
        """Return tools that should be auto-approved."""
        # listPrompts and usePrompt are read-only, safe to auto-approve
        # savePrompt creates files, so it should require permission
        return ['listPrompts', 'usePrompt', 'prompt']

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for prompt library."""
        prompts = self._discover_prompts()
        if not prompts:
            prompt_list = "(no prompts available yet)"
        else:
            prompt_list = ", ".join(sorted(prompts.keys())[:10])
            if len(prompts) > 10:
                prompt_list += f", ... ({len(prompts)} total)"

        return f"""You have access to a prompt library via listPrompts, usePrompt, and savePrompt tools.

Available prompts: {prompt_list}

## Using prompts
- Use listPrompts() to discover available prompts with their parameters
- Use usePrompt(name, params) to retrieve and expand a prompt, then follow its instructions
- The user can invoke prompts directly with `prompt <name> [args...]`

## Proactively suggest creating prompts
When you notice the user performing similar tasks repeatedly, suggest saving it as a reusable prompt:

Patterns to watch for:
- User asks for the same type of review/analysis multiple times
- User repeatedly gives similar formatting or style instructions
- User describes a workflow they want to reuse
- User mentions "like before" or "the same way" when giving instructions

When you notice a pattern (2-3 similar requests), proactively offer:
"I've noticed you've asked me to [describe pattern] a few times. Would you like me to save this as a reusable prompt? You could then use `prompt <suggested-name>` to invoke it."

If the user agrees, use savePrompt() with:
- A descriptive name (lowercase, hyphens)
- Clear instructions that capture their preferences
- Parameter placeholders for variable parts (e.g., {{{{file}}}}, {{{{focus}}}})

Example:
```
savePrompt(
  name="security-review",
  description="Review code for security vulnerabilities",
  content="Review {{{{file}}}} for security issues:\\n\\n1. Check for injection vulnerabilities\\n2. Look for authentication/authorization issues\\n3. Identify data exposure risks\\n\\nProvide specific line numbers and remediation suggestions."
)
```"""


def create_plugin() -> PromptLibraryPlugin:
    """Factory function to create the prompt library plugin instance."""
    return PromptLibraryPlugin()
