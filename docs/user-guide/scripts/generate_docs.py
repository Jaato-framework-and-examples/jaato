#!/usr/bin/env python3
"""
Automated documentation generation for Jaato rich client user guide.

This script extracts content from the codebase to generate documentation:
- Command help text
- Environment variables
- Default configurations
- Keybindings
"""

import ast
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict


@dataclass
class Command:
    """Represents a user command."""
    name: str
    syntax: str
    description: str
    examples: List[str]
    category: str
    aliases: List[str] = None


@dataclass
class EnvVar:
    """Represents an environment variable."""
    name: str
    description: str
    default: str
    category: str
    required: bool = False


@dataclass
class Keybinding:
    """Represents a keybinding."""
    action: str
    keys: str
    description: str
    context: str = "global"


class DocGenerator:
    """Generates documentation content from codebase."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.rich_client_path = repo_root / "rich-client"
        self.shared_path = repo_root / "shared"

    def extract_commands(self) -> List[Command]:
        """Extract command definitions from rich client."""
        commands = []

        # Parse rich_client.py to find command registry
        rich_client_file = self.rich_client_path / "rich_client.py"
        if rich_client_file.exists():
            with open(rich_client_file) as f:
                content = f.read()

            # Find command handlers (methods starting with do_)
            # This is a simplified extraction - adjust based on actual code structure
            do_methods = re.findall(
                r'def (do_\w+)\(self.*?\):\s*"""(.*?)"""',
                content,
                re.DOTALL
            )

            for method_name, docstring in do_methods:
                command_name = method_name.replace('do_', '')
                commands.append(Command(
                    name=command_name,
                    syntax=self._extract_syntax(docstring),
                    description=self._extract_description(docstring),
                    examples=self._extract_examples(docstring),
                    category=self._categorize_command(command_name)
                ))

        return commands

    def extract_env_vars(self) -> List[EnvVar]:
        """Extract environment variable usage from codebase."""
        env_vars = {}

        # Search for os.getenv, os.environ calls
        for py_file in self.repo_root.rglob("*.py"):
            if "venv" in str(py_file) or ".venv" in str(py_file):
                continue

            with open(py_file) as f:
                content = f.read()

            # Find os.getenv calls
            getenv_calls = re.findall(
                r'os\.(?:getenv|environ\.get)\(["\'](\w+)["\'](?:,\s*["\']([^"\']*)["\'])?\)',
                content
            )

            for var_name, default_value in getenv_calls:
                if var_name not in env_vars:
                    env_vars[var_name] = EnvVar(
                        name=var_name,
                        description=self._infer_var_description(var_name, py_file),
                        default=default_value or "None",
                        category=self._categorize_env_var(var_name)
                    )

        return list(env_vars.values())

    def extract_keybindings(self) -> List[Keybinding]:
        """Extract default keybindings from config."""
        keybindings = []

        # Look for keybinding initialization in rich client
        config_file = self.rich_client_path / "keybinding_manager.py"
        if config_file.exists():
            with open(config_file) as f:
                content = f.read()

            # Extract default keybindings from code
            # This is simplified - adjust based on actual structure
            bindings = re.findall(
                r'["\'](\w+)["\']\s*:\s*["\']([^"\']+)["\']',
                content
            )

            for action, keys in bindings:
                keybindings.append(Keybinding(
                    action=action,
                    keys=keys,
                    description=self._describe_keybinding_action(action)
                ))

        return keybindings

    def generate_markdown_commands(self, commands: List[Command]) -> str:
        """Generate markdown documentation for commands."""
        md = "# Command Reference\n\n"

        # Group by category
        by_category = {}
        for cmd in commands:
            by_category.setdefault(cmd.category, []).append(cmd)

        for category, cmds in sorted(by_category.items()):
            md += f"## {category}\n\n"

            for cmd in sorted(cmds, key=lambda c: c.name):
                md += f"### `{cmd.name}`\n\n"
                md += f"{cmd.description}\n\n"
                md += f"**Syntax:** `{cmd.syntax}`\n\n"

                if cmd.examples:
                    md += "**Examples:**\n\n"
                    for example in cmd.examples:
                        md += f"```bash\n{example}\n```\n\n"

                md += "---\n\n"

        return md

    def generate_markdown_env_vars(self, env_vars: List[EnvVar]) -> str:
        """Generate markdown table for environment variables."""
        md = "# Environment Variables\n\n"

        # Group by category
        by_category = {}
        for var in env_vars:
            by_category.setdefault(var.category, []).append(var)

        for category, vars in sorted(by_category.items()):
            md += f"## {category}\n\n"
            md += "| Variable | Description | Default | Required |\n"
            md += "|----------|-------------|---------|----------|\n"

            for var in sorted(vars, key=lambda v: v.name):
                required = "Yes" if var.required else "No"
                md += f"| `{var.name}` | {var.description} | `{var.default}` | {required} |\n"

            md += "\n"

        return md

    def generate_markdown_keybindings(self, keybindings: List[Keybinding]) -> str:
        """Generate markdown table for keybindings."""
        md = "# Keybindings\n\n"
        md += "| Action | Keys | Description |\n"
        md += "|--------|------|-------------|\n"

        for kb in sorted(keybindings, key=lambda k: k.action):
            md += f"| `{kb.action}` | `{kb.keys}` | {kb.description} |\n"

        return md

    # Helper methods

    def _extract_syntax(self, docstring: str) -> str:
        """Extract syntax line from docstring."""
        lines = docstring.strip().split('\n')
        for line in lines:
            if 'syntax:' in line.lower() or 'usage:' in line.lower():
                return line.split(':', 1)[1].strip()
        return lines[0] if lines else ""

    def _extract_description(self, docstring: str) -> str:
        """Extract description from docstring."""
        lines = [l.strip() for l in docstring.strip().split('\n') if l.strip()]
        # Return first paragraph
        desc = []
        for line in lines:
            if line.startswith('Syntax:') or line.startswith('Usage:') or line.startswith('Example'):
                break
            desc.append(line)
        return ' '.join(desc)

    def _extract_examples(self, docstring: str) -> List[str]:
        """Extract examples from docstring."""
        examples = []
        in_example = False
        current_example = []

        for line in docstring.split('\n'):
            if 'example' in line.lower() and ':' in line:
                in_example = True
                continue
            if in_example:
                if line.strip().startswith('-') or not line.strip():
                    if current_example:
                        examples.append('\n'.join(current_example))
                        current_example = []
                    if not line.strip():
                        in_example = False
                else:
                    current_example.append(line.strip())

        if current_example:
            examples.append('\n'.join(current_example))

        return examples

    def _categorize_command(self, name: str) -> str:
        """Categorize command by name."""
        auth_commands = ['anthropic_auth', 'antigravity_auth', 'github_auth']
        session_commands = ['reset', 'model', 'exit', 'quit']
        permission_commands = ['permissions']
        vision_commands = ['screenshot']

        if any(auth in name for auth in auth_commands):
            return "Authentication"
        elif name in session_commands:
            return "Session Management"
        elif name in permission_commands:
            return "Permission Management"
        elif name in vision_commands:
            return "Vision & Capture"
        else:
            return "General"

    def _categorize_env_var(self, name: str) -> str:
        """Categorize environment variable by name."""
        if any(provider in name for provider in ['GOOGLE', 'ANTHROPIC', 'GITHUB', 'OLLAMA']):
            return "Provider Configuration"
        elif any(key in name for key in ['PROXY', 'HTTP']):
            return "Network Configuration"
        elif 'JAATO' in name:
            if 'GC' in name:
                return "Garbage Collection"
            elif 'TELEMETRY' in name or 'OTEL' in name:
                return "Telemetry"
            else:
                return "General Configuration"
        elif any(key in name for key in ['LEDGER', 'TOKEN']):
            return "Token Accounting"
        else:
            return "Other"

    def _infer_var_description(self, var_name: str, file_path: Path) -> str:
        """Infer description from variable name and context."""
        # This would be enhanced by reading comments near the usage
        descriptions = {
            'PROJECT_ID': 'GCP project ID',
            'LOCATION': 'Vertex AI region',
            'MODEL_NAME': 'Model name to use',
            'ANTHROPIC_API_KEY': 'Anthropic API key',
            'GITHUB_TOKEN': 'GitHub personal access token',
            'JAATO_PARALLEL_TOOLS': 'Enable parallel tool execution',
            'JAATO_DEFERRED_TOOLS': 'Enable deferred tool loading',
        }

        return descriptions.get(var_name, f"Configuration for {var_name}")

    def _describe_keybinding_action(self, action: str) -> str:
        """Generate description for keybinding action."""
        descriptions = {
            'submit': 'Submit current input',
            'cancel': 'Cancel current operation',
            'exit': 'Exit the client',
            'toggle_plan': 'Toggle plan panel visibility',
            'toggle_tools': 'Toggle tool output panel visibility',
            'open_editor': 'Open external editor',
            'search': 'Open search mode',
        }
        return descriptions.get(action, action.replace('_', ' ').title())


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent.parent.parent
    generator = DocGenerator(repo_root)

    output_dir = repo_root / "docs" / "user-guide" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting commands...")
    commands = generator.extract_commands()
    (output_dir / "commands.md").write_text(
        generator.generate_markdown_commands(commands)
    )

    print("Extracting environment variables...")
    env_vars = generator.extract_env_vars()
    (output_dir / "env-vars.md").write_text(
        generator.generate_markdown_env_vars(env_vars)
    )

    print("Extracting keybindings...")
    keybindings = generator.extract_keybindings()
    (output_dir / "keybindings.md").write_text(
        generator.generate_markdown_keybindings(keybindings)
    )

    # Export structured data for other uses
    (output_dir / "commands.json").write_text(
        json.dumps([asdict(c) for c in commands], indent=2)
    )
    (output_dir / "env-vars.json").write_text(
        json.dumps([asdict(e) for e in env_vars], indent=2)
    )
    (output_dir / "keybindings.json").write_text(
        json.dumps([asdict(k) for k in keybindings], indent=2)
    )

    print(f"Generated documentation in {output_dir}")


if __name__ == "__main__":
    main()
