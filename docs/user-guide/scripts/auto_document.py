#!/usr/bin/env python3
"""
Self-documenting automation for Jaato user guide.

This script uses Jaato itself (in headless mode) to:
1. Generate missing documentation chapters
2. Update existing chapters when code changes
3. Ensure documentation stays current with codebase

The AI writes the documentation by analyzing the codebase.
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.jaato_client import JaatoClient


@dataclass
class Chapter:
    """Represents a documentation chapter."""
    number: str
    filename: str
    title: str
    description: str
    depends_on: List[str] = None  # Files/dirs to monitor for changes
    prompt_template: str = None  # Custom prompt for AI generation


@dataclass
class DocumentationState:
    """Tracks documentation generation state."""
    last_updated: str
    chapter_hashes: Dict[str, str]  # filename -> content hash
    dependency_hashes: Dict[str, str]  # dependency path -> hash
    generation_log: List[Dict] = None


class AutoDocGenerator:
    """Generates and updates documentation using Jaato."""

    def __init__(
        self,
        chapters_dir: Path,
        state_file: Path
    ):
        self.chapters_dir = chapters_dir
        self.state_file = state_file
        self.repo_root = REPO_ROOT

        # Load or initialize state
        self.state = self._load_state()

        # Load chapter manifest
        self.chapters = self._load_chapter_manifest()

    def _load_state(self) -> DocumentationState:
        """Load documentation generation state."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                data = json.load(f)
                return DocumentationState(**data)
        return DocumentationState(
            last_updated=datetime.now().isoformat(),
            chapter_hashes={},
            dependency_hashes={},
            generation_log=[]
        )

    def _save_state(self):
        """Save documentation generation state."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=2)

    def _load_chapter_manifest(self) -> List[Chapter]:
        """Load chapter definitions from manifest."""
        manifest_file = self.chapters_dir.parent / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                data = json.load(f)
                return [Chapter(**ch) for ch in data['chapters']]
        return self._get_default_chapters()

    def _get_default_chapters(self) -> List[Chapter]:
        """Default chapter structure for user guide."""
        return [
            Chapter(
                number="01",
                filename="01-introduction.md",
                title="Introduction",
                description="Overview of Jaato and the rich client",
                depends_on=["README.md", "CLAUDE.md"],
                prompt_template=None  # Uses default
            ),
            Chapter(
                number="02",
                filename="02-installation.md",
                title="Installation",
                description="Installing dependencies and setting up Jaato",
                depends_on=["requirements.txt", "CLAUDE.md"],
            ),
            Chapter(
                number="03",
                filename="03-quickstart.md",
                title="Quick Start",
                description="First conversation and basic commands",
                depends_on=["rich-client/rich_client.py"],
            ),
            Chapter(
                number="04",
                filename="04-authentication.md",
                title="Authentication & Providers",
                description="Setting up providers and authentication",
                depends_on=[
                    "shared/plugins/model_provider/",
                    "rich-client/commands/auth_commands.py"
                ],
            ),
            Chapter(
                number="05",
                filename="05-user-interface.md",
                title="User Interface",
                description="TUI components and navigation",
                depends_on=["rich-client/output_buffer.py", "rich-client/rich_client.py"],
            ),
            Chapter(
                number="06",
                filename="06-basic-commands.md",
                title="Basic Commands",
                description="Essential commands for daily use",
                depends_on=["rich-client/commands/"],
            ),
            Chapter(
                number="07",
                filename="07-session-management.md",
                title="Session Management",
                description="Managing conversations and sessions",
                depends_on=["server/session_manager.py"],
            ),
            Chapter(
                number="08",
                filename="08-permission-system.md",
                title="Permission System",
                description="Controlling tool execution permissions",
                depends_on=["shared/plugins/permission/"],
            ),
            Chapter(
                number="09",
                filename="09-vision-capture.md",
                title="Vision Capture",
                description="Screenshot command and visual capture",
                depends_on=["rich-client/commands/screenshot.py"],
            ),
            Chapter(
                number="10",
                filename="10-keybindings.md",
                title="Keybindings",
                description="Keyboard shortcuts and customization",
                depends_on=["rich-client/keybinding_manager.py"],
            ),
            Chapter(
                number="11",
                filename="11-configuration.md",
                title="Configuration Files",
                description=".jaato directory and configuration hierarchy",
                depends_on=[".jaato/", "rich-client/config.py"],
            ),
            Chapter(
                number="12",
                filename="12-environment-variables.md",
                title="Environment Variables",
                description="Complete environment variable reference",
                depends_on=["shared/", "CLAUDE.md"],
            ),
            Chapter(
                number="13",
                filename="13-theming.md",
                title="Theming",
                description="Built-in themes and custom theme creation",
                depends_on=["rich-client/theme.py"],
            ),
            Chapter(
                number="14",
                filename="14-server-configuration.md",
                title="Server Configuration",
                description="Server modes, IPC, and WebSocket setup",
                depends_on=["server/"],
            ),
            Chapter(
                number="15",
                filename="15-headless-mode.md",
                title="Headless Mode",
                description="Programmatic usage and API reference",
                depends_on=["shared/jaato_client.py", "examples/"],
            ),
        ]

    def _compute_hash(self, path: Path) -> str:
        """Compute hash of file or directory."""
        if not path.exists():
            return ""

        if path.is_file():
            with open(path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        else:
            # Directory: hash all Python files
            hashes = []
            for py_file in sorted(path.rglob("*.py")):
                with open(py_file, 'rb') as f:
                    hashes.append(hashlib.sha256(f.read()).hexdigest())
            combined = ''.join(hashes)
            return hashlib.sha256(combined.encode()).hexdigest()

    def _has_dependencies_changed(self, chapter: Chapter) -> Tuple[bool, List[str]]:
        """Check if chapter dependencies have changed."""
        if not chapter.depends_on:
            return False, []

        changed = []
        for dep in chapter.depends_on:
            dep_path = self.repo_root / dep
            current_hash = self._compute_hash(dep_path)
            stored_hash = self.state.dependency_hashes.get(dep, "")

            if current_hash != stored_hash:
                changed.append(dep)
                self.state.dependency_hashes[dep] = current_hash

        return len(changed) > 0, changed

    def _chapter_exists(self, chapter: Chapter) -> bool:
        """Check if chapter file exists."""
        return (self.chapters_dir / chapter.filename).exists()

    def _generate_chapter(self, chapter: Chapter, mode: str = "create") -> bool:
        """Generate or update a chapter using Jaato."""
        print(f"\n{'Generating' if mode == 'create' else 'Updating'}: {chapter.title}")

        # Build prompt for AI
        if mode == "create":
            prompt = self._build_creation_prompt(chapter)
        else:
            prompt = self._build_update_prompt(chapter)

        print(f"  Prompt: {prompt[:100]}...")

        # Initialize Jaato client in headless mode
        # Framework reads JAATO_PROVIDER, MODEL_NAME, and provider-specific env vars
        import os
        client = JaatoClient()

        try:
            # Connect - framework handles all configuration from environment
            client.connect(model=os.getenv("MODEL_NAME"))

            # Don't configure tools - we want pure text generation
            # client.configure_tools() is NOT called

            # Send message and collect response
            print("  Waiting for AI response...")
            output_parts = []

            def collect_output(source: str, text: str, mode: str):
                if source == "model":
                    output_parts.append(text)
                    # Show progress
                    print(".", end="", flush=True)

            response = client.send_message(prompt, on_output=collect_output)
            print()  # New line after dots

            if not output_parts:
                print(f"  ✗ No response from AI")
                return False

            content = ''.join(output_parts)

            # Extract markdown if AI wrapped it in code blocks
            content = self._extract_markdown(content)

            # Write to file
            output_file = self.chapters_dir / chapter.filename
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                f.write(content)

            # Update state
            self.state.chapter_hashes[chapter.filename] = hashlib.sha256(
                content.encode()
            ).hexdigest()

            self.state.generation_log.append({
                "timestamp": datetime.now().isoformat(),
                "chapter": chapter.filename,
                "mode": mode,
                "success": True
            })

            print(f"  ✓ Generated: {output_file}")
            return True

        except KeyboardInterrupt:
            print(f"\n  ✗ Interrupted by user")
            self.state.generation_log.append({
                "timestamp": datetime.now().isoformat(),
                "chapter": chapter.filename,
                "mode": mode,
                "success": False,
                "error": "Interrupted by user"
            })
            raise  # Re-raise to stop the whole process

        except Exception as e:
            # Log detailed error information
            error_type = type(e).__name__
            error_msg = str(e)
            print(f"  ✗ {error_type}: {error_msg}")

            # Save error to generation log
            self.state.generation_log.append({
                "timestamp": datetime.now().isoformat(),
                "chapter": chapter.filename,
                "mode": mode,
                "success": False,
                "error": error_msg,
                "error_type": error_type
            })

            # Save state even on failure so we don't lose error log
            self._save_state()

            return False

        finally:
            # Clean up
            pass

    def _build_creation_prompt(self, chapter: Chapter) -> str:
        """Build prompt for creating a new chapter."""
        if chapter.prompt_template:
            return chapter.prompt_template

        return f"""You are writing a chapter for the Jaato Rich Client User Guide.

**Chapter {chapter.number}: {chapter.title}**

{chapter.description}

**Requirements:**
1. Write in clear, accessible markdown
2. Use academic/technical documentation style
3. Include practical examples where relevant
4. Use these formatting conventions:
   - Commands: `command`
   - File paths: `/path/to/file`
   - Environment variables: `$VARIABLE_NAME`
   - Keyboard shortcuts: Ctrl+C (plain text)
5. Structure with proper heading hierarchy (# for chapter, ## for sections)
6. Include code examples in appropriate language code blocks
7. Add tip/warning boxes where helpful (we'll convert these to LaTeX boxes)

**Context:**
Analyze the following files/directories to understand what to document:
{chr(10).join(f"- {dep}" for dep in (chapter.depends_on or []))}

Read the codebase using your tools to understand the implementation, then write comprehensive user-facing documentation.

**Target Audience:** Both beginners and advanced users

**Style:** Formal but accessible, example-driven, progressive disclosure (simple to complex)

Generate ONLY the markdown content for this chapter. Do not include any preamble or explanation - output pure markdown that can be directly saved to a file.
"""

    def _build_update_prompt(self, chapter: Chapter) -> str:
        """Build prompt for updating an existing chapter."""
        chapter_file = self.chapters_dir / chapter.filename

        with open(chapter_file) as f:
            current_content = f.read()

        return f"""You are updating a chapter in the Jaato Rich Client User Guide.

**Chapter {chapter.number}: {chapter.title}**

The following dependencies have changed and the chapter needs to be updated:
{chr(10).join(f"- {dep}" for dep in (chapter.depends_on or []))}

**Current Chapter Content:**
```markdown
{current_content}
```

**Task:**
1. Read the updated files/directories to understand what changed
2. Update the chapter to reflect the current state of the codebase
3. Maintain the existing structure and style
4. Add new sections if needed for new features
5. Remove outdated information
6. Keep all good existing content

**Requirements:**
- Preserve the same formatting conventions
- Maintain the same level of detail
- Keep the same tone and style
- Only change what needs updating

Generate the COMPLETE updated chapter as markdown. Output pure markdown that can directly replace the existing file.
"""

    def _extract_markdown(self, content: str) -> str:
        """Extract markdown from AI response (remove code block wrappers if present)."""
        content = content.strip()

        # Remove markdown code block wrappers
        if content.startswith("```markdown") or content.startswith("```md"):
            lines = content.split('\n')
            # Remove first and last lines (```markdown and ```)
            content = '\n'.join(lines[1:-1])

        elif content.startswith("```"):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1])

        return content.strip()

    def generate_missing_chapters(self) -> int:
        """Generate all missing chapters."""
        generated = 0

        for chapter in self.chapters:
            if not self._chapter_exists(chapter):
                print(f"\n📝 Missing: {chapter.title}")
                if self._generate_chapter(chapter, mode="create"):
                    generated += 1
                    self._save_state()

        return generated

    def update_stale_chapters(self) -> int:
        """Update chapters whose dependencies have changed."""
        updated = 0

        for chapter in self.chapters:
            if not self._chapter_exists(chapter):
                continue

            has_changed, changed_deps = self._has_dependencies_changed(chapter)
            if has_changed:
                print(f"\n🔄 Stale: {chapter.title}")
                print(f"  Changed: {', '.join(changed_deps)}")

                if self._generate_chapter(chapter, mode="update"):
                    updated += 1
                    self._save_state()

        return updated

    def generate_all(self, force: bool = False):
        """Generate all chapters (missing + stale)."""
        print("=" * 60)
        print("Jaato Documentation Auto-Generator")
        print("=" * 60)

        if force:
            print("\n⚠️  Force mode: regenerating ALL chapters")
            # Clear state to force regeneration
            self.state.chapter_hashes.clear()
            self.state.dependency_hashes.clear()

        # Generate missing
        print("\n" + "=" * 60)
        print("Phase 1: Generate Missing Chapters")
        print("=" * 60)
        generated = self.generate_missing_chapters()
        print(f"\n✓ Generated {generated} new chapters")

        # Update stale
        print("\n" + "=" * 60)
        print("Phase 2: Update Stale Chapters")
        print("=" * 60)
        updated = self.update_stale_chapters()
        print(f"\n✓ Updated {updated} stale chapters")

        # Summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Generated: {generated}")
        print(f"  Updated:   {updated}")
        print(f"  Total:     {generated + updated}")

        self.state.last_updated = datetime.now().isoformat()
        self._save_state()

        print(f"\n✓ State saved to: {self.state_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-generate Jaato documentation using Jaato itself"
    )
    parser.add_argument(
        "--chapters-dir",
        type=Path,
        default=Path(__file__).parent.parent / "chapters",
        help="Directory for chapter markdown files"
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path(__file__).parent.parent / "generated" / "doc-state.json",
        help="State file for tracking generation"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate all chapters"
    )
    parser.add_argument(
        "--chapter",
        help="Generate/update specific chapter by filename"
    )

    args = parser.parse_args()

    generator = AutoDocGenerator(
        chapters_dir=args.chapters_dir,
        state_file=args.state_file
    )

    if args.chapter:
        # Generate specific chapter
        chapter = next(
            (ch for ch in generator.chapters if ch.filename == args.chapter),
            None
        )
        if not chapter:
            print(f"Error: Chapter '{args.chapter}' not found in manifest")
            sys.exit(1)

        mode = "create" if not generator._chapter_exists(chapter) else "update"
        success = generator._generate_chapter(chapter, mode=mode)
        sys.exit(0 if success else 1)
    else:
        # Generate all
        generator.generate_all(force=args.force)


if __name__ == "__main__":
    main()
