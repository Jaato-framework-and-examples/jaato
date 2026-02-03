#!/usr/bin/env python3
"""
Automated screenshot generation for Jaato documentation.

Uses pexpect to drive a rich client session and capture screenshots at key moments.
Generates SVG terminal output that can be converted to PNG if needed.
"""

import os
import pexpect
import time
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Screenshot:
    """Represents a screenshot to capture."""
    name: str
    description: str
    setup_commands: List[str]
    wait_after: float = 2.0  # Seconds to wait before capturing


class ScreenshotGenerator:
    """Generates screenshots by driving rich client with pexpect."""

    def __init__(self, output_dir: Path, venv_python: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.venv_python = venv_python
        self.vision_dir = output_dir / "captures"
        self.vision_dir.mkdir(exist_ok=True)

    def start_server(self) -> pexpect.spawn:
        """Start the Jaato server in daemon mode."""
        socket_path = "/tmp/jaato_screenshot_test.sock"

        # Clean up any existing socket
        if os.path.exists(socket_path):
            os.remove(socket_path)

        # Start server
        cmd = f"{self.venv_python} -m server --ipc-socket {socket_path} --daemon"
        print(f"Starting server: {cmd}")
        os.system(cmd)
        time.sleep(3)  # Wait for server to start

        return socket_path

    def stop_server(self):
        """Stop the server."""
        cmd = f"{self.venv_python} -m server --stop"
        print(f"Stopping server: {cmd}")
        os.system(cmd)

    def start_client(self, socket_path: str) -> pexpect.spawn:
        """Start rich client connected to server."""
        # Set environment for vision capture
        env = os.environ.copy()
        env['JAATO_VISION_DIR'] = str(self.vision_dir)
        env['TERM'] = 'xterm-256color'

        cmd = f"{self.venv_python} rich-client/rich_client.py --connect {socket_path}"
        print(f"Starting client: {cmd}")

        child = pexpect.spawn(
            cmd,
            env=env,
            encoding='utf-8',
            dimensions=(40, 120)  # rows, cols
        )

        # Wait for client to be ready
        child.expect(['.*>', '.*$'], timeout=10)

        return child

    def capture_screenshot(
        self,
        child: pexpect.spawn,
        name: str,
        format: str = "svg"
    ) -> Optional[Path]:
        """Capture a screenshot using the screenshot command."""
        # Send screenshot command
        child.sendline(f"screenshot nosend format {format}")

        # Wait for capture
        time.sleep(1)

        # Look for the most recent file
        captures = list(self.vision_dir.glob(f"*.{format}"))
        if captures:
            latest = max(captures, key=lambda p: p.stat().st_mtime)
            # Rename to our desired name
            target = self.output_dir / f"{name}.{format}"
            latest.rename(target)
            print(f"Captured: {target}")
            return target

        return None

    def generate_scenario(
        self,
        socket_path: str,
        screenshot: Screenshot
    ) -> Optional[Path]:
        """Generate a screenshot for a specific scenario."""
        print(f"\nGenerating screenshot: {screenshot.name}")
        print(f"Description: {screenshot.description}")

        child = self.start_client(socket_path)

        try:
            # Execute setup commands
            for i, cmd in enumerate(screenshot.setup_commands):
                print(f"  Command {i+1}: {cmd}")
                child.sendline(cmd)

                # Wait for command to complete
                # This is simplified - may need adjustment based on command
                time.sleep(1)

                # Check if we need to handle permission prompts
                if 'permission' in cmd.lower():
                    # Auto-approve for screenshot purposes
                    child.expect(['.*[y/n/a/t/i].*'], timeout=5)
                    child.sendline('y')
                    time.sleep(0.5)

            # Wait before capturing
            time.sleep(screenshot.wait_after)

            # Capture the screenshot
            result = self.capture_screenshot(child, screenshot.name)

            return result

        except Exception as e:
            print(f"Error generating screenshot: {e}")
            return None

        finally:
            # Clean exit
            child.sendline("exit")
            child.close()

    def generate_all(self, scenarios: List[Screenshot]):
        """Generate all screenshots."""
        print("Starting screenshot generation...")

        # Start server once
        socket_path = self.start_server()

        try:
            results = []
            for scenario in scenarios:
                result = self.generate_scenario(socket_path, scenario)
                if result:
                    results.append((scenario.name, result))

            print(f"\n✓ Generated {len(results)} screenshots")
            for name, path in results:
                print(f"  - {name}: {path}")

        finally:
            self.stop_server()


def get_documentation_screenshots() -> List[Screenshot]:
    """Define all screenshots needed for documentation."""
    return [
        Screenshot(
            name="01-first-launch",
            description="Rich client on first launch",
            setup_commands=[],
            wait_after=1.0
        ),
        Screenshot(
            name="02-basic-interaction",
            description="Basic conversation with the model",
            setup_commands=[
                "What are the main features of Jaato?"
            ],
            wait_after=3.0
        ),
        Screenshot(
            name="03-model-switch",
            description="Switching models",
            setup_commands=[
                "model gemini-2.5-pro"
            ],
            wait_after=1.0
        ),
        Screenshot(
            name="04-plan-panel",
            description="Plan panel showing task breakdown",
            setup_commands=[
                # This would need a prompt that triggers plan generation
                "Create a Python script that processes CSV files and generates a summary report"
            ],
            wait_after=5.0
        ),
        Screenshot(
            name="05-permissions-show",
            description="Viewing permission status",
            setup_commands=[
                "permissions show"
            ],
            wait_after=1.0
        ),
        Screenshot(
            name="06-theme-dark",
            description="Dark theme (default)",
            setup_commands=[
                "/theme dark"
            ],
            wait_after=1.0
        ),
        Screenshot(
            name="07-theme-light",
            description="Light theme",
            setup_commands=[
                "/theme light"
            ],
            wait_after=1.0
        ),
        Screenshot(
            name="08-anthropic-auth",
            description="Anthropic authentication command",
            setup_commands=[
                "anthropic-auth status"
            ],
            wait_after=1.0
        ),
        Screenshot(
            name="09-github-auth",
            description="GitHub authentication command",
            setup_commands=[
                "github-auth status"
            ],
            wait_after=1.0
        ),
        Screenshot(
            name="10-reset-session",
            description="Resetting the session",
            setup_commands=[
                "Tell me about Python",
                "reset"
            ],
            wait_after=1.0
        ),
        Screenshot(
            name="11-help-command",
            description="Getting help",
            setup_commands=[
                "help"
            ],
            wait_after=2.0
        ),
        Screenshot(
            name="12-screenshot-command",
            description="Using the screenshot command",
            setup_commands=[
                "screenshot help"
            ],
            wait_after=1.0
        ),
    ]


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate screenshots for Jaato documentation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "assets" / "screenshots",
        help="Output directory for screenshots"
    )
    parser.add_argument(
        "--venv-python",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent / ".venv" / "bin" / "python",
        help="Path to venv Python interpreter"
    )
    parser.add_argument(
        "--scenarios",
        nargs='+',
        help="Specific scenarios to generate (by name)"
    )

    args = parser.parse_args()

    generator = ScreenshotGenerator(args.output_dir, args.venv_python)
    scenarios = get_documentation_screenshots()

    # Filter scenarios if specified
    if args.scenarios:
        scenarios = [s for s in scenarios if s.name in args.scenarios]

    generator.generate_all(scenarios)


if __name__ == "__main__":
    main()
