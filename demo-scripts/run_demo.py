#!/usr/bin/env python3
"""
Demo runner for recording plugin demos with termsvg.

This script uses pexpect to interact with the real simple client,
running demo scripts defined in YAML files.

Usage:
    # Run a demo script
    python run_demo.py shared/plugins/cli/demo.yaml

    # Record with termtosvg
    termtosvg -c "python run_demo.py shared/plugins/cli/demo.yaml" -g 100x40 demo.svg

Requirements:
    pip install pexpect pyyaml
"""

import sys
import time
import argparse
from pathlib import Path

# Determine project root relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

try:
    import pexpect
except ImportError:
    print("Error: pexpect is required. Install with: pip install pexpect")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("Error: pyyaml is required. Install with: pip install pyyaml")
    sys.exit(1)


def type_slowly(child, text, delay=0):
    """Type text to the child process.

    Args:
        child: pexpect child process
        text: Text to send
        delay: Per-character delay in seconds. If > 0, waits (delay * len(text))
               before sending the text all at once. Default 0 = instant.
    """
    if delay and delay > 0:
        # Wait proportional to text length, then send all at once
        total_delay = delay * len(text)
        time.sleep(total_delay)
    child.send(text + '\n')


# Regex to match optional ANSI escape codes
ANSI = r'(?:\x1b\[[0-9;]*m)*'

# Pattern indices for expect
PATTERN_PROMPT = 0
PATTERN_PERMISSION = 1
PATTERN_CLARIFY_SINGLE = 2
PATTERN_CLARIFY_MULTI = 3
PATTERN_CLARIFY_FREE = 4
PATTERN_REFERENCE_SELECT = 5


def wait_for_prompt(child, timeout=60):
    """Wait for the 'You>' prompt (with optional ANSI color codes)."""
    child.expect(rf'{ANSI}You>{ANSI}', timeout=timeout)


def wait_for_permission_or_prompt(child, response='y', timeout=60):
    """Wait for either permission prompt, clarification prompt, or next You> prompt.

    Handles:
    - Permission prompts (Options:) - responds with the given response
    - Clarification prompts - auto-responds with first choice or default text
    - You> prompt - returns when ready for next input
    """
    # First wait for the client to acknowledge the command
    child.expect(r'\[client\]', timeout=timeout)

    # Now wait for various prompts in a loop until we get back to You>
    patterns = [
        rf'{ANSI}You>{ANSI}',           # 0: Ready for next input
        r'Options:',                     # 1: Permission prompt
        r'Enter choice \[[\d-]+\]:',     # 2: Single choice clarification
        r'Enter choices:',               # 3: Multiple choice clarification
        r'  > ',                          # 4: Free text clarification
        r"'none' or empty to skip",      # 5: Reference selection prompt
    ]

    while True:
        index = child.expect(patterns, timeout=timeout)

        if index == PATTERN_PROMPT:
            # Back to You> prompt, we're done
            return

        elif index == PATTERN_PERMISSION:
            # Permission was requested, send response
            time.sleep(0.3)
            child.send(f'{response}\n')

        elif index == PATTERN_CLARIFY_SINGLE:
            # Single choice - send "1" (first option)
            time.sleep(0.3)
            child.send('1\n')

        elif index == PATTERN_CLARIFY_MULTI:
            # Multiple choice - send "1" (first option)
            time.sleep(0.3)
            child.send('1\n')

        elif index == PATTERN_CLARIFY_FREE:
            # Free text - send auto response
            time.sleep(0.3)
            child.send('auto-response\n')

        elif index == PATTERN_REFERENCE_SELECT:
            # Reference selection - select all available
            time.sleep(0.3)
            child.send('all\n')


def run_demo(script_path: Path):
    """Run a demo from a YAML script file."""
    # Load the script
    with open(script_path) as f:
        script = yaml.safe_load(f)

    name = script.get('name', script_path.stem)
    timeout = script.get('timeout', 120)
    steps = script.get('steps', [])
    setup = script.get('setup')

    print(f"Starting {name}...")

    # Run setup commands if specified
    if setup:
        import subprocess
        for cmd in setup:
            subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)

    # Spawn the client
    child = pexpect.spawn(
        'python', ['simple-client/interactive_client.py', '--env-file', '.env'],
        encoding='utf-8',
        timeout=timeout,
        cwd=str(PROJECT_ROOT)
    )
    child.logfile_read = sys.stdout

    # Wait for initial prompt
    wait_for_prompt(child)
    time.sleep(0.3)

    # Execute each step
    for step in steps:
        if isinstance(step, str):
            # Simple string = just type it
            text = step
            is_local = False
            permission = 'y'
            delay = 0  # Send instantly by default
        else:
            # Dict with type and optional settings
            text = step.get('type', '')
            permission = step.get('permission', 'y')
            delay = step.get('delay', 0)  # Send instantly by default
            is_local = step.get('local', False)

        # Sanitize text: replace embedded newlines with spaces
        # This prevents multi-line YAML strings from being interpreted
        # as multiple separate commands by the terminal
        if text and '\n' in text:
            text = ' '.join(text.split())

        type_slowly(child, text, delay=delay)

        if text.lower() == 'quit':
            # Quit doesn't need any waiting
            pass
        elif is_local:
            # Local commands (like "plan") just wait for next prompt
            wait_for_prompt(child, timeout=timeout)
            time.sleep(0.3)
        else:
            # Model commands wait for [client] then permission/prompt
            wait_for_permission_or_prompt(child, response=permission, timeout=timeout)
            time.sleep(0.3)

    # Wait for EOF
    child.expect(pexpect.EOF, timeout=10)
    print(f"\n{name} completed.")


def main():
    parser = argparse.ArgumentParser(
        description='Run demo scripts for plugin recordings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_demo.py shared/plugins/cli/demo.yaml
    python run_demo.py shared/plugins/file_edit/demo.yaml

Script format (YAML):
    name: CLI Plugin Demo
    timeout: 120
    setup:
      - mkdir -p /tmp/demo
    steps:
      - type: "List the Python files"
        permission: "y"
      - type: "Show git status"
        permission: "a"
      - type: "plan"
        local: true       # Plugin commands that don't go to model
      - "quit"
        """
    )
    parser.add_argument('script', type=Path, help='Path to demo YAML script')

    args = parser.parse_args()

    if not args.script.exists():
        # Try relative to project root
        script_path = PROJECT_ROOT / args.script
        if not script_path.exists():
            print(f"Error: Script not found: {args.script}")
            sys.exit(1)
    else:
        script_path = args.script

    run_demo(script_path)


if __name__ == '__main__':
    main()
