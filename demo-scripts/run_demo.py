#!/usr/bin/env python3
"""
Demo runner for recording plugin demos with termsvg.

This script uses pexpect to interact with the real simple client,
feeding it prompts and responses via PTY to produce authentic recordings.

Usage:
    # Record CLI plugin demo
    termtosvg -c "python run_demo.py cli" -g 100x40 cli_demo.svg

    # Record file_edit plugin demo
    termtosvg -c "python run_demo.py file_edit" -g 100x45 file_edit_demo.svg

    # Record all demos
    ./record_all.sh

Requirements:
    pip install pexpect

    Also requires a valid .env file with API credentials.
"""

import sys
import os
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


def type_slowly(child, text, delay=0.05):
    """Type text character by character for visual effect."""
    for char in text:
        child.send(char)
        time.sleep(delay)
    child.send('\n')


# Regex to match optional ANSI escape codes
ANSI = r'(?:\x1b\[[0-9;]*m)*'

# Pattern indices for expect
PATTERN_PROMPT = 0
PATTERN_PERMISSION = 1


def wait_for_prompt(child, timeout=60):
    """Wait for the 'You>' prompt (with optional ANSI color codes)."""
    child.expect(rf'{ANSI}You>{ANSI}', timeout=timeout)


def wait_for_permission_or_prompt(child, response='y', timeout=60):
    """Wait for either permission prompt or next You> prompt.

    If permission is requested, send the response and wait for prompt.
    If prompt appears directly (permission already granted), just return.
    """
    patterns = [rf'{ANSI}You>{ANSI}', r'Options:']
    index = child.expect(patterns, timeout=timeout)

    if index == PATTERN_PERMISSION:
        # Permission was requested, send response
        time.sleep(0.3)
        child.send(f'{response}\n')
        # Now wait for the prompt after permission
        wait_for_prompt(child, timeout=timeout)


def run_cli_demo():
    """Run CLI plugin demo - shell command execution."""
    print("Starting CLI plugin demo...")

    child = pexpect.spawn(
        'python', ['simple-client/interactive_client.py', '--env-file', '.env'],
        encoding='utf-8',
        timeout=120,
        cwd=str(PROJECT_ROOT)
    )
    child.logfile_read = sys.stdout

    # Wait for initialization
    wait_for_prompt(child)
    time.sleep(0.5)

    # First command: list Python files
    type_slowly(child, "List the Python files in the current directory")

    # Wait for permission (or prompt if already granted) and approve with 'y'
    wait_for_permission_or_prompt(child, response='y')
    time.sleep(0.5)

    # Second command: git status
    type_slowly(child, "Show me the git status")

    # Wait for permission (or prompt if already granted) and approve with 'a' (always)
    wait_for_permission_or_prompt(child, response='a')
    time.sleep(1.0)

    # Exit
    type_slowly(child, "quit", delay=0.08)

    child.expect(pexpect.EOF, timeout=5)
    print("\nCLI demo completed.")


def run_file_edit_demo():
    """Run file_edit plugin demo - file modification with diff."""
    print("Starting file_edit plugin demo...")

    # Create a test file first
    import os
    test_dir = "/tmp/jaato_demo"
    os.makedirs(test_dir, exist_ok=True)
    test_file = f"{test_dir}/config.py"

    with open(test_file, 'w') as f:
        f.write('''# Configuration file
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
ENABLE_LOGGING = True
DEBUG_MODE = False
''')

    child = pexpect.spawn(
        'python', ['simple-client/interactive_client.py', '--env-file', '.env'],
        encoding='utf-8',
        timeout=120,
        cwd=str(PROJECT_ROOT)
    )
    child.logfile_read = sys.stdout

    wait_for_prompt(child)
    time.sleep(0.5)

    # Ask to update the file
    type_slowly(child, f"Update {test_file} to change MAX_RETRIES from 3 to 5")

    # readFile is auto-approved, wait for updateFile permission (or prompt if granted)
    wait_for_permission_or_prompt(child, response='y')
    time.sleep(1.0)

    # Exit
    type_slowly(child, "quit", delay=0.08)

    child.expect(pexpect.EOF, timeout=5)
    print("\nfile_edit demo completed.")


def run_web_search_demo():
    """Run web_search plugin demo - web search."""
    print("Starting web_search plugin demo...")

    child = pexpect.spawn(
        'python', ['simple-client/interactive_client.py', '--env-file', '.env'],
        encoding='utf-8',
        timeout=120,
        cwd=str(PROJECT_ROOT)
    )
    child.logfile_read = sys.stdout

    wait_for_prompt(child)
    time.sleep(0.5)

    # Search query
    type_slowly(child, "Search the web for Python asyncio best practices")

    # Wait for permission (or prompt if already granted)
    wait_for_permission_or_prompt(child, response='y')
    time.sleep(1.0)

    type_slowly(child, "quit", delay=0.08)

    child.expect(pexpect.EOF, timeout=5)
    print("\nweb_search demo completed.")


def run_todo_demo():
    """Run TODO plugin demo - plan creation and tracking."""
    print("Starting TODO plugin demo...")

    child = pexpect.spawn(
        'python', ['simple-client/interactive_client.py', '--env-file', '.env'],
        encoding='utf-8',
        timeout=120,
        cwd=str(PROJECT_ROOT)
    )
    child.logfile_read = sys.stdout

    wait_for_prompt(child)
    time.sleep(0.5)

    # Ask for a task that triggers plan creation
    type_slowly(child, "Help me refactor the authentication module. Create a plan first.")

    # createPlan might need permission depending on config
    wait_for_permission_or_prompt(child, response='y', timeout=60)
    time.sleep(0.5)

    # Check plan status
    type_slowly(child, "plan", delay=0.08)

    wait_for_prompt(child)
    time.sleep(1.0)

    type_slowly(child, "quit", delay=0.08)

    child.expect(pexpect.EOF, timeout=5)
    print("\nTODO demo completed.")


def run_references_demo():
    """Run references plugin demo - documentation loading."""
    print("Starting references plugin demo...")

    child = pexpect.spawn(
        'python', ['simple-client/interactive_client.py', '--env-file', '.env'],
        encoding='utf-8',
        timeout=120,
        cwd=str(PROJECT_ROOT)
    )
    child.logfile_read = sys.stdout

    wait_for_prompt(child)
    time.sleep(0.5)

    # List available references
    type_slowly(child, "listReferences", delay=0.08)

    wait_for_prompt(child)
    time.sleep(0.5)

    # Ask something that might trigger reference selection
    type_slowly(child, "I need to add a new API endpoint. What standards should I follow?")

    # Handle permission if selectReferences is called (or prompt if granted/skipped)
    wait_for_permission_or_prompt(child, response='y', timeout=60)
    time.sleep(1.0)

    type_slowly(child, "quit", delay=0.08)

    child.expect(pexpect.EOF, timeout=5)
    print("\nreferences demo completed.")


def run_subagent_demo():
    """Run subagent plugin demo - spawning specialized agents."""
    print("Starting subagent plugin demo...")

    child = pexpect.spawn(
        'python', ['simple-client/interactive_client.py', '--env-file', '.env'],
        encoding='utf-8',
        timeout=180,  # Subagents take longer
        cwd=str(PROJECT_ROOT)
    )
    child.logfile_read = sys.stdout

    wait_for_prompt(child)
    time.sleep(0.5)

    # List available profiles
    type_slowly(child, "profiles", delay=0.08)

    wait_for_prompt(child)
    time.sleep(0.5)

    # Ask for analysis that might spawn a subagent
    type_slowly(child, "Analyze this codebase structure and identify the main components")

    # spawn_subagent requires permission (use longer timeout for subagent execution)
    wait_for_permission_or_prompt(child, response='y', timeout=120)
    time.sleep(1.0)

    type_slowly(child, "quit", delay=0.08)

    child.expect(pexpect.EOF, timeout=5)
    print("\nsubagent demo completed.")


DEMOS = {
    'cli': run_cli_demo,
    'file_edit': run_file_edit_demo,
    'web_search': run_web_search_demo,
    'todo': run_todo_demo,
    'references': run_references_demo,
    'subagent': run_subagent_demo,
}


def main():
    parser = argparse.ArgumentParser(
        description='Run plugin demos for termsvg recording',
        epilog='''
Examples:
    # Record single demo
    termtosvg -c "python run_demo.py cli" -g 100x40 cli_demo.svg

    # Run demo directly (for testing)
    python run_demo.py cli

    # List available demos
    python run_demo.py --list
'''
    )
    parser.add_argument('demo', nargs='?', choices=list(DEMOS.keys()),
                        help='Demo to run')
    parser.add_argument('--list', action='store_true',
                        help='List available demos')

    args = parser.parse_args()

    if args.list:
        print("Available demos:")
        for name in DEMOS:
            print(f"  - {name}")
        return

    if not args.demo:
        parser.print_help()
        return

    # Change to project root
    os.chdir(PROJECT_ROOT)

    DEMOS[args.demo]()


if __name__ == '__main__':
    main()
