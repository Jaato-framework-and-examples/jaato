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

import base64
import json
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
from .validation import PromptValidator, format_validation_error

# Type alias for output callback: (source, text, mode) -> None
OutputCallback = Callable[[str, str, str], None]

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
        self._output_callback: Optional[OutputCallback] = None
        # Plugin registry reference (for tool change notifications)
        self._plugin_registry: Optional[Any] = None
        # Callback for notifying when prompts change (tools added/removed)
        self._on_tools_changed: Optional[Callable[[List[str]], None]] = None

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

        # Pre-discover prompts so they're available as tools immediately
        self._discover_prompts()
        self._trace(f"initialize: workspace={self._workspace_path}, discovered {len(self._prompt_cache)} prompts")

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

    def set_output_callback(self, callback: Optional[OutputCallback]) -> None:
        """Set callback for output/progress messages."""
        self._output_callback = callback

    def set_plugin_registry(self, registry: Any) -> None:
        """Set the plugin registry reference (called during expose_tool).

        This allows the plugin to notify the registry when tools change
        (e.g., after fetching new prompts).
        """
        self._plugin_registry = registry
        self._trace(f"set_plugin_registry: {type(registry).__name__}")

    def set_on_tools_changed(self, callback: Callable[[List[str]], None]) -> None:
        """Set callback for when prompt tools are added or removed.

        Args:
            callback: Function called with list of new tool names after fetch.
                     The TUI can use this to refresh completions and tool caches.
        """
        self._on_tools_changed = callback

    def _notify_tools_changed(self, new_tools: List[str]) -> None:
        """Notify that prompt tools have changed (new prompts fetched).

        This refreshes the prompt cache and notifies subscribers.
        """
        self._trace(f"_notify_tools_changed: {new_tools}")
        # Refresh the cache
        self._discover_prompts()
        # Notify callback if set
        if self._on_tools_changed:
            self._on_tools_changed(new_tools)

    def _emit(self, message: str, mode: str = "write") -> None:
        """Emit a message via output callback or print."""
        if self._output_callback:
            self._output_callback("prompt_library", message, mode)
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
            # Jaato native prompts (writable)
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
            # Jaato skills (writable, for ClawdHub installs)
            PromptSource(
                path=workspace / ".jaato" / "skills",
                source_name="project-skills",
                entry_file=SKILL_ENTRY_FILE,
                writable=True,
            ),
            PromptSource(
                path=home / ".jaato" / "skills",
                source_name="global-skills",
                entry_file=SKILL_ENTRY_FILE,
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
        """Fetch skills by running an npx command (e.g., ClawdHub).

        Runs the npx command in the .jaato/ directory. Tools like ClawdHub
        create a skills/ subdirectory automatically, matching Claude's pattern.
        """
        self._trace(f"fetch npx: {package} {args}")

        # Run in .jaato/ parent directory - ClawdHub creates skills/ subdirectory
        # dest_dir is .jaato/prompts/, we want .jaato/ as working dir
        jaato_dir = dest_dir.parent
        skills_dir = jaato_dir / "skills"

        # Track existing skills to detect new ones
        existing_skills = set()
        if skills_dir.exists():
            existing_skills = {d.name for d in skills_dir.iterdir() if d.is_dir()}

        # Ensure .jaato/ directory exists
        jaato_dir.mkdir(parents=True, exist_ok=True)

        cmd = ["npx", "-y", package] + args
        self._emit(f"Running: npx {package} {' '.join(args)}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                cwd=str(jaato_dir)
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                return FetchResult(
                    success=False,
                    error=f"npx command failed: {error_msg}",
                    source_type="npx",
                    source_params=f"{package} {' '.join(args)}"
                )

            # Find newly installed skills
            prompts_fetched = []
            validator = PromptValidator()

            if skills_dir.exists():
                for skill_dir in skills_dir.iterdir():
                    if skill_dir.is_dir() and skill_dir.name not in existing_skills:
                        # New skill installed
                        entry_file = skill_dir / SKILL_ENTRY_FILE
                        if not entry_file.exists():
                            entry_file = skill_dir / PROMPT_ENTRY_FILE

                        if entry_file.exists():
                            content = entry_file.read_text(encoding='utf-8')

                            # Validate content (log warning but don't reject npx installs)
                            validation = validator.validate(content)
                            if not validation.valid:
                                self._trace(f"Warning: npx skill {skill_dir.name} has validation issues: {validation.errors}")

                            # Add provenance to entry file
                            content = self._add_provenance(
                                content, "npx", f"{package} {' '.join(args)}"
                            )
                            entry_file.write_text(content, encoding='utf-8')
                            prompts_fetched.append(skill_dir.name)

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
                skipped_invalid = []
                validator = PromptValidator()

                for prompt_dir in prompt_dirs:
                    if not prompt_dir.exists():
                        continue

                    for item in prompt_dir.iterdir():
                        if item.name.startswith('.'):
                            continue

                        if item.is_file() and item.suffix in ('.md', '.yaml', '.yml'):
                            content = item.read_text(encoding='utf-8')

                            # Validate content before saving
                            validation = validator.validate(content)
                            if not validation.valid:
                                self._trace(f"Skipping invalid prompt {item.name}: {validation.errors}")
                                skipped_invalid.append(item.name)
                                continue

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
                                # Validate directory-based prompt entry file
                                content = entry_file.read_text(encoding='utf-8')
                                validation = validator.validate(content)
                                if not validation.valid:
                                    self._trace(f"Skipping invalid prompt dir {item.name}: {validation.errors}")
                                    skipped_invalid.append(item.name)
                                    continue

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

    def _validate_content(self, content: str, source_url: str = "") -> FetchResult:
        """Validate fetched content is a valid prompt/skill.

        Args:
            content: The content to validate
            source_url: URL for error hint (e.g., GitHub blob URL hint)

        Returns:
            FetchResult with success=False if validation fails, None otherwise
        """
        validator = PromptValidator()
        result = validator.validate(content, source_hint=source_url)

        if not result.valid:
            error_msg = format_validation_error(result, source_url)
            return FetchResult(
                success=False,
                error=error_msg,
                source_type="url" if source_url else "",
                source_params=source_url
            )

        return None  # Validation passed

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

            # Validate content before saving
            validation_error = self._validate_content(content, source_url=url)
            if validation_error:
                return validation_error

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
        - owner/repo (fetches all prompts from standard locations via git clone)
        - owner/repo/path/to/skill (fetches specific skill via gh CLI)
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
        path = '/'.join(parts[2:]) if len(parts) > 2 else None

        # If specific path requested, try gh CLI first
        if path:
            result = self._fetch_github_path(owner, repo, path, dest_dir)
            if result is not None:
                return result
            # Fall through to git clone if gh not available

        # Fall back to git clone for whole repo
        repo_url = f"https://github.com/{owner}/{repo}.git"
        return self._fetch_from_git(repo_url, dest_dir)

    def _fetch_github_path(
        self, owner: str, repo: str, path: str, dest_dir: Path
    ) -> Optional[FetchResult]:
        """Fetch a specific path from GitHub using gh CLI.

        Returns None if gh CLI is not available (caller should fall back to git).
        """
        # Check if gh is available
        try:
            subprocess.run(
                ["gh", "--version"],
                capture_output=True,
                check=True,
                timeout=10
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            self._trace("gh CLI not available, falling back to git clone")
            return None

        self._emit(f"Fetching: {owner}/{repo}/{path}...")
        source_ref = f"{owner}/{repo}/{path}"

        try:
            # Get contents at path
            result = subprocess.run(
                ["gh", "api", f"repos/{owner}/{repo}/contents/{path}"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                return FetchResult(
                    success=False,
                    error=f"GitHub API error: {error_msg.strip()}",
                    source_type="github",
                    source_params=source_ref
                )

            contents = json.loads(result.stdout)

            # Determine if this is a file or directory
            if isinstance(contents, list):
                # Directory - fetch as a skill to .jaato/skills/
                return self._fetch_github_directory(
                    owner, repo, path, contents, dest_dir.parent / "skills"
                )
            else:
                # Single file - fetch to prompts/
                return self._fetch_github_file(contents, dest_dir, source_ref)

        except subprocess.TimeoutExpired:
            return FetchResult(
                success=False,
                error="GitHub API request timed out",
                source_type="github",
                source_params=source_ref
            )
        except json.JSONDecodeError as e:
            return FetchResult(
                success=False,
                error=f"Invalid JSON response from GitHub: {e}",
                source_type="github",
                source_params=source_ref
            )
        except Exception as e:
            return FetchResult(
                success=False,
                error=str(e),
                source_type="github",
                source_params=source_ref
            )

    def _fetch_github_directory(
        self,
        owner: str,
        repo: str,
        path: str,
        contents: List[Dict[str, Any]],
        dest_dir: Path
    ) -> FetchResult:
        """Fetch a directory from GitHub as a skill."""
        source_ref = f"{owner}/{repo}/{path}"
        skill_name = path.rstrip('/').split('/')[-1]

        # Handle conflict
        skill_dest = dest_dir / skill_name
        proceed, actual_path = self._handle_conflict(skill_name, skill_dest, "")
        if not proceed or not actual_path:
            return FetchResult(
                success=True,
                prompts_fetched=[],
                source_type="github",
                source_params=source_ref
            )

        skill_dest = actual_path
        skill_dest.mkdir(parents=True, exist_ok=True)

        # Download directory contents recursively
        self._download_github_contents(contents, owner, repo, path, skill_dest)

        # Validate entry file if present
        validator = PromptValidator()
        entry_file = None
        for ef in [SKILL_ENTRY_FILE, PROMPT_ENTRY_FILE]:
            if (skill_dest / ef).exists():
                entry_file = skill_dest / ef
                break

        if entry_file:
            content = entry_file.read_text(encoding='utf-8')
            validation = validator.validate(content)
            if not validation.valid:
                shutil.rmtree(skill_dest)
                return FetchResult(
                    success=False,
                    error=format_validation_error(validation, f"github:{source_ref}"),
                    source_type="github",
                    source_params=source_ref
                )

            content = self._add_provenance(content, "github", source_ref)
            entry_file.write_text(content, encoding='utf-8')

        return FetchResult(
            success=True,
            prompts_fetched=[skill_name],
            source_type="github",
            source_params=source_ref
        )

    def _fetch_github_file(
        self, file_info: Dict[str, Any], dest_dir: Path, source_ref: str
    ) -> FetchResult:
        """Fetch a single file from GitHub."""
        filename = file_info.get('name', 'fetched-prompt.md')
        if not filename.endswith('.md'):
            return FetchResult(
                success=False,
                error=f"Expected .md file, got: {filename}",
                source_type="github",
                source_params=source_ref
            )

        # Download the file content
        content = self._get_github_file_content(file_info)
        if content is None:
            return FetchResult(
                success=False,
                error="Could not download file content",
                source_type="github",
                source_params=source_ref
            )

        # Validate content
        validation_error = self._validate_content(content, source_url=source_ref)
        if validation_error:
            return validation_error

        # Add provenance
        content = self._add_provenance(content, "github", source_ref)

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
                source_type="github",
                source_params=source_ref
            )
        else:
            return FetchResult(
                success=True,
                prompts_fetched=[],
                source_type="github",
                source_params=source_ref
            )

    def _download_github_contents(
        self,
        contents: List[Dict[str, Any]],
        owner: str,
        repo: str,
        base_path: str,
        dest: Path
    ) -> None:
        """Recursively download GitHub directory contents."""
        for item in contents:
            item_name = item.get('name', '')
            item_type = item.get('type', '')
            item_path = f"{base_path}/{item_name}"

            if item_type == 'file':
                content = self._get_github_file_content(item)
                if content is not None:
                    (dest / item_name).write_text(content, encoding='utf-8')
            elif item_type == 'dir':
                # Recurse into subdirectory
                subdir = dest / item_name
                subdir.mkdir(exist_ok=True)
                try:
                    result = subprocess.run(
                        ["gh", "api", f"repos/{owner}/{repo}/contents/{item_path}"],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        sub_contents = json.loads(result.stdout)
                        if isinstance(sub_contents, list):
                            self._download_github_contents(
                                sub_contents, owner, repo, item_path, subdir
                            )
                except (subprocess.TimeoutExpired, json.JSONDecodeError):
                    self._trace(f"Failed to fetch subdirectory: {item_path}")

    def _get_github_file_content(self, file_info: Dict[str, Any]) -> Optional[str]:
        """Get file content from GitHub file info.

        Tries download_url first, falls back to base64-encoded content.
        """
        import urllib.request
        import urllib.error

        # Try download_url first
        download_url = file_info.get('download_url')
        if download_url:
            try:
                req = urllib.request.Request(
                    download_url,
                    headers={'User-Agent': 'jaato-prompt-fetch/1.0'}
                )
                with urllib.request.urlopen(req, timeout=30) as response:
                    return response.read().decode('utf-8')
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                self._trace(f"Failed to download from {download_url}: {e}")

        # Fall back to base64-encoded content
        if 'content' in file_info:
            try:
                return base64.b64decode(file_info['content']).decode('utf-8')
            except Exception as e:
                self._trace(f"Failed to decode base64 content: {e}")

        return None

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
                # Notify that new prompt tools are available
                new_tools = [f"prompt.{name}" for name in result.prompts_fetched]
                self._notify_tools_changed(new_tools)

                prompts_list = ", ".join(result.prompts_fetched)
                # npx and github path fetches install to skills/ subdirectory
                is_github_path = (
                    source_type == "github" and
                    len(source_params[0].split('/')) > 2
                )
                uses_skills_dir = source_type == "npx" or is_github_path
                actual_dest = dest_dir.parent / "skills" if uses_skills_dir else dest_dir
                return f"Fetched {len(result.prompts_fetched)} prompt(s): {prompts_list}\nDestination: {actual_dest}"
            else:
                return f"No prompts were fetched (all skipped or no prompts found)"
        else:
            return f"Fetch failed: {result.error}"

    # ==================== Prompt-to-Tool Conversion ====================

    def _params_to_json_schema(self, params: Dict[str, PromptParam]) -> Dict[str, Any]:
        """Convert prompt template params to JSON Schema.

        Args:
            params: Dict of PromptParam objects from the prompt

        Returns:
            JSON Schema dict for tool parameters
        """
        properties = {}
        required = []

        for name, param in params.items():
            prop: Dict[str, Any] = {
                "type": "string",
                "description": param.description or f"Parameter: {name}"
            }
            if param.default is not None:
                prop["default"] = param.default
            if param.enum:
                prop["enum"] = param.enum
            properties[name] = prop

            if param.required:
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _prompt_to_tool_schema(self, info: PromptInfo) -> ToolSchema:
        """Convert a PromptInfo to a discoverable ToolSchema.

        Args:
            info: The prompt info to convert

        Returns:
            ToolSchema with 'prompt.' prefix namespace
        """
        return ToolSchema(
            name=f"prompt.{info.name}",
            description=info.description or f"Prompt: {info.name}",
            parameters=self._params_to_json_schema(info.params),
            category="prompt",
            discoverability="discoverable",
        )

    # ==================== Model Tools ====================

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for model access.

        Returns:
            - savePrompt tool for creating new prompts
            - Virtual tools for each discovered prompt (prompt.name)
        """
        schemas = [
            # Only savePrompt remains - listPrompts/usePrompt are redundant
            # since prompts are now first-class tools discoverable via list_tools
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
                category="prompt",
                discoverability="discoverable",
            ),
        ]

        # Add virtual tools for each discovered prompt
        for info in self._discover_prompts().values():
            schemas.append(self._prompt_to_tool_schema(info))

        return schemas

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executor mapping for model tools and user commands.

        Returns mapping for:
        - savePrompt: Create new prompts
        - prompt: User command
        - prompt.<name>: Dynamic executors for each discovered prompt
        """
        executors: Dict[str, Callable[[Dict[str, Any]], Any]] = {
            'savePrompt': self._execute_save_prompt,
            'prompt': self._execute_prompt_command,
        }

        # Add dynamic executors for each prompt tool
        for name in self._discover_prompts():
            executors[f"prompt.{name}"] = self._make_prompt_executor(name)

        return executors

    def _make_prompt_executor(self, prompt_name: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Factory for prompt tool executors.

        Args:
            prompt_name: Name of the prompt to create executor for

        Returns:
            Executor function for the prompt tool
        """
        def executor(args: Dict[str, Any]) -> Dict[str, Any]:
            return self._execute_prompt_tool(prompt_name, args)
        return executor

    def _execute_prompt_tool(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a prompt tool by name with given params.

        Args:
            name: The prompt name (without 'prompt.' prefix)
            params: Parameters to substitute in the prompt

        Returns:
            Dict with 'content' on success, 'error' on failure
        """
        self._trace(f"execute prompt.{name} with params={params}")

        prompts = self._discover_prompts()
        if name not in prompts:
            available = list(prompts.keys())
            return {
                'error': f'Prompt not found: {name}',
                'available': available[:10],
                'hint': f'Use list_tools(category="prompt") to see available prompts'
            }

        info = prompts[name]

        try:
            content = self._get_prompt_content(info)
            # Params from tool call are named params
            substituted, missing = self._substitute_params(content, params, [])

            # Determine the skill's root path for relative path resolution
            if info.is_directory:
                skill_root = str(info.path)
            else:
                skill_root = str(info.path.parent)

            result = {
                'name': name,
                'content': substituted,
                'source': info.source,
                'skill_path': skill_root,
                'instruction': (
                    'Execute the instructions in the content above. '
                    f'Relative paths are relative to: {skill_root}\n\n'
                    'EXECUTION BEHAVIOR:\n'
                    '- Execute silently. Do not narrate steps, explain what you are doing, or summarize the prompt.\n'
                    '- Only involve the user when necessary: missing information (use clarification tool), '
                    'or unrecoverable failures not addressed by the prompt.\n'
                    '- If a tool fails and the prompt provides fallback/recovery instructions, follow them silently.\n'
                    '- If a tool fails with no fallback in the prompt, report the error and stop.\n'
                    '- When complete, provide only the final result or outcome the user needs.'
                ),
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

            # Add subcommands
            subcommands = [
                ("fetch", "Fetch prompts from external source"),
                ("remove", "Remove a prompt from the library"),
            ]
            for name, desc in subcommands:
                if not prefix or name.startswith(prefix):
                    completions.append(CommandCompletion(name, desc))

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

        # Handle 'remove' subcommand completions
        if args[0].lower() == "remove":
            if len(args) == 2:
                # Show removable prompts (only from writable sources)
                prefix = args[1].lower() if len(args) > 1 else ""
                writable_sources = {"project", "global", "project-skills", "global-skills"}
                prompts = self._discover_prompts()
                completions = []
                for name, info in sorted(prompts.items()):
                    if info.source not in writable_sources:
                        continue  # Skip read-only prompts
                    if prefix and not name.lower().startswith(prefix):
                        continue
                    source_tag = f" [{info.source}]"
                    completions.append(CommandCompletion(
                        name,
                        info.description[:50] + source_tag
                    ))
                return completions
            return []

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
            prompt remove <name>                - Remove a prompt from the library

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

        # Handle 'remove' subcommand
        if first_arg == "remove":
            return self._handle_remove_subcommand(positional_args[1:])

        # Otherwise, first arg is prompt name
        prompt_name = positional_args[0]
        prompt_args = positional_args[1:]

        # Get the prompt and substitute parameters
        prompts = self._discover_prompts()
        if prompt_name not in prompts:
            available = list(prompts.keys())
            hint = f'Available prompts: {", ".join(available[:5])}{"..." if len(available) > 5 else ""}'
            return f'Prompt not found: {prompt_name}\n{hint}'

        info = prompts[prompt_name]

        try:
            content = self._get_prompt_content(info)
            # User command uses positional args
            substituted, missing = self._substitute_params(content, {}, prompt_args)

            if missing:
                substituted += f"\n\n[Note: Missing parameters: {', '.join(missing)}]"

            return substituted

        except Exception as e:
            return f'Failed to read prompt: {e}'

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
        lines.append("     prompt remove <name>")
        return "\n".join(lines)

    def _handle_fetch_subcommand(self, args: List[str]) -> str:
        """Handle the 'prompt fetch' subcommand.

        Args:
            args: Arguments after 'fetch', e.g., ['npx', 'clawdhub@latest', 'install', 'skill']

        Syntax:
            prompt fetch <source_type> <source_params...> [project|user]

        Examples:
            prompt fetch npx clawdhub@latest install some-skill
            prompt fetch git https://github.com/user/prompts
            prompt fetch github user/prompts
            prompt fetch github user/skills-repo/my-skill
            prompt fetch url https://example.com/review.md
            prompt fetch npx clawdhub@latest install some-skill user
        """
        if not args:
            return """Usage: prompt fetch <source_type> <source_params...> [project|user]

Source types:
  npx <package> [args...]       - Install skills via npx (to skills/)
  git <repo_url>                - Clone git repository
  github <owner/repo>           - Fetch all prompts from GitHub repo
  github <owner/repo/path>      - Fetch specific skill from GitHub (to skills/)
  url <url>                     - Fetch single prompt from URL

Destination (optional, default: project):
  project   skills: .jaato/skills/    prompts: .jaato/prompts/
  user      skills: ~/.jaato/skills/  prompts: ~/.jaato/prompts/

Examples:
  prompt fetch github anthropics/prompt-library
  prompt fetch github mkdev-me/claude-skills/gemini-image-generator
  prompt fetch npx clawdhub@latest install some-skill
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

    def _handle_remove_subcommand(self, args: List[str]) -> str:
        """Handle the 'prompt remove' subcommand.

        Args:
            args: Arguments after 'remove', e.g., ['my-prompt']

        Syntax:
            prompt remove <name>

        Examples:
            prompt remove old-review
            prompt remove gemini-image-generator
        """
        if not args:
            return """Usage: prompt remove <name>

Removes a prompt from the library.

Only prompts in writable locations can be removed:
  - .jaato/prompts/
  - .jaato/skills/
  - ~/.jaato/prompts/
  - ~/.jaato/skills/

Read-only locations (.claude/) cannot be modified.

Examples:
  prompt remove old-review
  prompt remove gemini-image-generator"""

        prompt_name = args[0]
        self._trace(f"remove: name={prompt_name}")

        # Find the prompt
        prompts = self._discover_prompts()
        if prompt_name not in prompts:
            available = list(prompts.keys())
            hint = f'Available prompts: {", ".join(available[:5])}{"..." if len(available) > 5 else ""}'
            return f'Prompt not found: {prompt_name}\n{hint}'

        info = prompts[prompt_name]

        # Check if the source is writable
        writable_sources = {"project", "global", "project-skills", "global-skills"}
        if info.source not in writable_sources:
            return f"Cannot remove '{prompt_name}': it's in a read-only location ({info.source}).\n" \
                   f"Path: {info.path}"

        # Confirm with user
        if info.is_directory:
            message = f"Remove prompt directory '{prompt_name}' and all its contents?"
        else:
            message = f"Remove prompt file '{prompt_name}'?"
        message += f"\nPath: {info.path}"

        confirmation = self._confirm(message, ["yes", "no"])
        if confirmation != "yes":
            return "Removal cancelled."

        # Perform the deletion
        try:
            if info.is_directory:
                shutil.rmtree(info.path)
            else:
                info.path.unlink()

            self._trace(f"Removed prompt: {prompt_name} from {info.path}")

            # Notify that tools changed (prompt removed)
            self._notify_tools_changed([f"prompt.{prompt_name}"])

            return f"Removed prompt '{prompt_name}' from {info.path}"

        except PermissionError:
            return f"Permission denied: cannot remove {info.path}"
        except Exception as e:
            return f"Failed to remove prompt: {e}"

    def get_auto_approved_tools(self) -> List[str]:
        """Return tools that should be auto-approved.

        Prompt tools (prompt.*) are read-only and safe to auto-approve.
        savePrompt creates files, so it requires permission.
        """
        auto_approved = ['prompt']  # User command

        # Auto-approve all prompt.* tools (read-only)
        for name in self._discover_prompts():
            auto_approved.append(f"prompt.{name}")

        return auto_approved

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for prompt library."""
        prompts = self._discover_prompts()
        if not prompts:
            return None  # No instructions if no prompts

        prompt_tools = [f"prompt.{name}" for name in sorted(prompts.keys())[:10]]
        more_indicator = f"... ({len(prompts)} total)" if len(prompts) > 10 else ""

        return f"""## Prompt Tools (category: prompt)

Reusable prompts are available as discoverable tools with 'prompt.' prefix.
Available: {', '.join(prompt_tools)}{more_indicator}

### Discovery
- list_tools(category="prompt") - see all prompt tools
- get_tool_schemas(names=["prompt.code-review"]) - get parameters for a prompt

### Usage
Call prompt tools directly with their parameters:
  prompt.code-review(file="main.py", focus="security")

**IMPORTANT**: When a prompt tool returns content, treat it as a user request and execute the instructions immediately. Use your available tools (CLI, file operations, etc.) to carry out what the prompt describes.

### Creating Prompts
Use savePrompt(name, content, description) to create new reusable prompts.
The user can also invoke prompts with `prompt <name> [args...]`

### Proactively suggest creating prompts
When you notice the user performing similar tasks repeatedly (2-3 times), suggest saving it as a reusable prompt:
"I've noticed you've asked me to [pattern]. Would you like me to save this as a reusable prompt?"

If they agree, use savePrompt() with:
- A descriptive name (lowercase, hyphens)
- Clear instructions capturing their preferences
- Parameter placeholders: {{{{file}}}}, {{{{focus}}}}"""


def create_plugin() -> PromptLibraryPlugin:
    """Factory function to create the prompt library plugin instance."""
    return PromptLibraryPlugin()
