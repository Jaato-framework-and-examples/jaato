#!/usr/bin/env python3
"""jaato interactive installer.

A standalone TUI installer for the jaato framework. Guides the user through
virtual environment creation, install source selection, and granular feature
selection for the three jaato packages (jaato-sdk, jaato-server, jaato-tui).

Zero external dependencies — runs with any Python 3.10+.

Usage:
    python install.py
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Terminal abstraction
# ---------------------------------------------------------------------------

_IS_WINDOWS = platform.system() == "Windows"
_NO_COLOR = os.environ.get("NO_COLOR") is not None or os.environ.get("TERM") == "dumb"


def _supports_color() -> bool:
    if _NO_COLOR:
        return False
    if _IS_WINDOWS:
        return os.environ.get("WT_SESSION") is not None or "ANSICON" in os.environ
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOR = _supports_color()


class _Ansi:
    """ANSI escape code helpers."""

    RESET = "\033[0m" if _COLOR else ""
    BOLD = "\033[1m" if _COLOR else ""
    DIM = "\033[2m" if _COLOR else ""
    GREEN = "\033[32m" if _COLOR else ""
    CYAN = "\033[36m" if _COLOR else ""
    YELLOW = "\033[33m" if _COLOR else ""
    RED = "\033[31m" if _COLOR else ""
    WHITE = "\033[37m" if _COLOR else ""
    BG_BLUE = "\033[44m" if _COLOR else ""
    CLEAR_SCREEN = "\033[2J\033[H" if _COLOR else ""
    HIDE_CURSOR = "\033[?25l" if _COLOR else ""
    SHOW_CURSOR = "\033[?25h" if _COLOR else ""


class _Key(Enum):
    UP = "up"
    DOWN = "down"
    ENTER = "enter"
    SPACE = "space"
    ESC = "esc"
    BACKSPACE = "backspace"
    CHAR = "char"
    QUIT = "quit"


@dataclass
class _KeyEvent:
    key: _Key
    char: str = ""


def _read_key() -> _KeyEvent:
    """Read a single keypress from the terminal (blocking)."""
    if _IS_WINDOWS:
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            return _KeyEvent(_Key.ENTER)
        if ch == " ":
            return _KeyEvent(_Key.SPACE)
        if ch == "\x1b":
            return _KeyEvent(_Key.ESC)
        if ch == "\x08":
            return _KeyEvent(_Key.BACKSPACE)
        if ch == "q":
            return _KeyEvent(_Key.QUIT)
        if ch in ("\x00", "\xe0"):
            ch2 = msvcrt.getwch()
            if ch2 == "H":
                return _KeyEvent(_Key.UP)
            if ch2 == "P":
                return _KeyEvent(_Key.DOWN)
            return _KeyEvent(_Key.CHAR, ch2)
        return _KeyEvent(_Key.CHAR, ch)
    else:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\r" or ch == "\n":
                return _KeyEvent(_Key.ENTER)
            if ch == " ":
                return _KeyEvent(_Key.SPACE)
            if ch == "\x7f" or ch == "\x08":
                return _KeyEvent(_Key.BACKSPACE)
            if ch == "q":
                return _KeyEvent(_Key.QUIT)
            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "A":
                        return _KeyEvent(_Key.UP)
                    if ch3 == "B":
                        return _KeyEvent(_Key.DOWN)
                    return _KeyEvent(_Key.CHAR, ch3)
                if ch2 == "":
                    return _KeyEvent(_Key.ESC)
                return _KeyEvent(_Key.ESC)
            if ch == "\x03":  # Ctrl+C
                return _KeyEvent(_Key.QUIT)
            return _KeyEvent(_Key.CHAR, ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _clear() -> None:
    sys.stdout.write(_Ansi.CLEAR_SCREEN)
    sys.stdout.flush()


def _write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def _get_terminal_width() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


# ---------------------------------------------------------------------------
# UI data model
# ---------------------------------------------------------------------------


@dataclass
class RadioOption:
    """A single option in a radio group."""

    label: str
    value: str
    available: bool = True


@dataclass
class CheckboxItem:
    """A toggleable feature item.

    Attributes:
        label: Display text (e.g. "Google GenAI SDK (Gemini, Vertex AI)").
        extra: The pyproject.toml extra name (e.g. "google").
        checked: Current state.
        locked: If True, item is always checked and cursor skips it.
        package: Which package this extra belongs to ("server" or "tui").
    """

    label: str
    extra: str
    checked: bool = False
    locked: bool = False
    package: str = "server"


@dataclass
class Section:
    """A group of checkbox items with a header."""

    title: str
    items: List[CheckboxItem] = field(default_factory=list)


class InstallSource(Enum):
    EDITABLE = "editable"
    PYPI = "pypi"
    TEST_PYPI = "test_pypi"


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------


def _build_sections() -> List[Section]:
    """Build the feature selection sections."""
    return [
        Section("Packages", [
            CheckboxItem("jaato-sdk (required)", "", locked=True, checked=True),
            CheckboxItem("jaato-server (required)", "", locked=True, checked=True),
            CheckboxItem("jaato-tui", "", checked=True),
            CheckboxItem("jaato-tui vision support (screenshot export)", "vision", package="tui"),
        ]),
        Section("Model Providers", [
            CheckboxItem("Google GenAI SDK (Gemini, Vertex AI)", "google", checked=True),
            CheckboxItem("Anthropic SDK (Claude) — included in core", "", locked=True, checked=True),
            CheckboxItem("Azure AI Inference SDK (GitHub Models)", "github-models"),
            CheckboxItem("OpenAI-compatible SDK (NVIDIA NIM, ZhipuAI)", "nim"),
        ]),
        Section("Plugins", [
            CheckboxItem("Web fetching & search (web_fetch, web_search)", "web", checked=True),
            CheckboxItem("Interactive terminal sessions (interactive_shell)", "interactive", checked=True),
            CheckboxItem("AST-based code pattern search (ast_search)", "ast"),
            CheckboxItem("Jupyter notebook support (notebook)", "notebook"),
            CheckboxItem("Diagram rendering in terminal (mermaid_formatter)", "diagrams"),
            CheckboxItem("Handlebars template engine (template)", "templates"),
        ]),
        Section("Extras", [
            CheckboxItem("Kerberos proxy authentication", "kerberos"),
            CheckboxItem("Telemetry (OpenTelemetry)", "telemetry"),
            CheckboxItem("Kaggle integration", "kaggle"),
        ]),
    ]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_header() -> str:
    """Render the installer header."""
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    lines = [
        f"{_Ansi.BOLD}{_Ansi.CYAN}  jaato installer{_Ansi.RESET}",
        "",
        f"  Python {py_ver} detected",
        "",
    ]
    return "\n".join(lines)


def _render_radio(
    title: str,
    options: List[RadioOption],
    selected: int,
    text_input: Optional[str] = None,
    text_input_active: bool = False,
) -> str:
    """Render a radio button group."""
    lines = [f"  {_Ansi.BOLD}{title}{_Ansi.RESET}", ""]
    for i, opt in enumerate(options):
        if not opt.available:
            continue
        bullet = f"{_Ansi.GREEN}●{_Ansi.RESET}" if i == selected else "○"
        highlight = _Ansi.BOLD if i == selected else ""
        dim = _Ansi.DIM if not opt.available else ""
        lines.append(f"  {dim}{bullet} {highlight}{opt.label}{_Ansi.RESET}")
        if text_input is not None and i == selected and text_input_active:
            lines.append(f"    > {text_input}_")
    lines.append("")
    return "\n".join(lines)


def _render_sections(
    sections: List[Section],
    cursor: int,
    flat_items: List[Tuple[int, int]],
) -> str:
    """Render the feature selection checkbox list.

    Args:
        sections: The section definitions.
        cursor: Current cursor position in flat_items.
        flat_items: List of (section_idx, item_idx) for navigable items.
    """
    lines = []
    flat_pos = 0
    for si, section in enumerate(sections):
        # Section header
        lines.append(f"  {_Ansi.BOLD}{_Ansi.CYAN}── {section.title} {'─' * max(1, 45 - len(section.title))}{_Ansi.RESET}")
        for ii, item in enumerate(section.items):
            is_cursor = False
            if flat_pos < len(flat_items) and flat_items[flat_pos] == (si, ii):
                is_cursor = flat_pos == cursor
                flat_pos += 1
            elif item.locked:
                # Locked items aren't in flat_items
                pass

            if item.locked:
                check = f"{_Ansi.DIM}[✓]{_Ansi.RESET}"
                label = f"{_Ansi.DIM}{item.label}{_Ansi.RESET}"
            elif item.checked:
                check = f"{_Ansi.GREEN}[✓]{_Ansi.RESET}"
                label = item.label
            else:
                check = "[ ]"
                label = item.label

            pointer = f"{_Ansi.CYAN}▸{_Ansi.RESET} " if is_cursor else "  "
            highlight = _Ansi.BOLD if is_cursor else ""
            lines.append(f"  {pointer}{check} {highlight}{label}{_Ansi.RESET}")
        lines.append("")
    return "\n".join(lines)


def _render_footer(screen: str) -> str:
    """Render navigation hints for the current screen."""
    hints = {
        "venv": "↑↓ navigate  Enter confirm  q quit",
        "source": "↑↓ navigate  Enter confirm  Esc back  q quit",
        "features": "↑↓ navigate  Space toggle  Enter confirm  Esc back  q quit",
        "review": "Enter install  Esc back  q quit",
    }
    hint = hints.get(screen, "")
    return f"\n  {_Ansi.DIM}{hint}{_Ansi.RESET}\n"


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------


def screen_venv() -> Optional[Tuple[bool, str]]:
    """Screen 1: Virtual environment setup.

    Returns:
        (create_venv, path) tuple, or None if user quit.
    """
    options = [
        RadioOption("Yes, at ./.venv", "default"),
        RadioOption("Yes, custom path...", "custom"),
        RadioOption("No, install into current environment", "none"),
    ]
    selected = 0
    custom_path = ""
    typing_path = False

    while True:
        _clear()
        _write(_render_header())
        _write(_render_radio(
            "Create a virtual environment?",
            options,
            selected,
            text_input=custom_path if options[selected].value == "custom" else None,
            text_input_active=typing_path,
        ))
        _write(_render_footer("venv"))

        key = _read_key()

        if key.key == _Key.QUIT:
            return None

        if typing_path:
            if key.key == _Key.ENTER:
                if custom_path.strip():
                    return (True, custom_path.strip())
                typing_path = False
            elif key.key == _Key.ESC:
                typing_path = False
                custom_path = ""
            elif key.key == _Key.BACKSPACE:
                custom_path = custom_path[:-1]
            elif key.key == _Key.CHAR:
                custom_path += key.char
            continue

        if key.key == _Key.UP:
            selected = (selected - 1) % len(options)
        elif key.key == _Key.DOWN:
            selected = (selected + 1) % len(options)
        elif key.key == _Key.ENTER:
            val = options[selected].value
            if val == "default":
                return (True, ".venv")
            elif val == "custom":
                typing_path = True
            elif val == "none":
                return (False, "")


def screen_source(is_source_checkout: bool) -> Optional[InstallSource]:
    """Screen 2: Install source selection.

    Args:
        is_source_checkout: Whether local source dirs exist.

    Returns:
        InstallSource, or None if user quit/back.
    """
    options = []
    if is_source_checkout:
        options.append(RadioOption("Editable from local sources (-e)", "editable"))
    options.append(RadioOption("From PyPI (stable release)", "pypi"))
    options.append(RadioOption("From Test PyPI (pre-release / staging)", "test_pypi"))

    selected = 0

    while True:
        _clear()
        _write(_render_header())
        _write(_render_radio("Install source", options, selected))
        _write(_render_footer("source"))

        key = _read_key()

        if key.key == _Key.QUIT:
            return None
        if key.key == _Key.ESC:
            return None  # Signal "go back"
        if key.key == _Key.UP:
            selected = (selected - 1) % len(options)
        elif key.key == _Key.DOWN:
            selected = (selected + 1) % len(options)
        elif key.key == _Key.ENTER:
            return InstallSource(options[selected].value)


def screen_features(sections: List[Section]) -> Optional[List[Section]]:
    """Screen 3: Feature selection.

    Returns:
        Updated sections, or None if user quit/back.
    """
    # Build flat list of navigable (non-locked) items
    flat_items: List[Tuple[int, int]] = []
    for si, section in enumerate(sections):
        for ii, item in enumerate(section.items):
            if not item.locked:
                flat_items.append((si, ii))

    cursor = 0

    while True:
        _clear()
        _write(_render_header())
        _write(_render_sections(sections, cursor, flat_items))
        _write(_render_footer("features"))

        key = _read_key()

        if key.key == _Key.QUIT:
            return None
        if key.key == _Key.ESC:
            return None  # Signal "go back"
        if key.key == _Key.UP:
            cursor = (cursor - 1) % len(flat_items)
        elif key.key == _Key.DOWN:
            cursor = (cursor + 1) % len(flat_items)
        elif key.key == _Key.SPACE:
            si, ii = flat_items[cursor]
            item = sections[si].items[ii]
            item.checked = not item.checked
        elif key.key == _Key.ENTER:
            return sections


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------


def _build_pip_commands(
    sections: List[Section],
    source: InstallSource,
    pip_path: str,
) -> List[str]:
    """Build the pip install command(s) from feature selections.

    Returns:
        List of command strings to execute.
    """
    # Collect extras by package
    server_extras: List[str] = []
    tui_extras: List[str] = []
    install_tui = True

    for section in sections:
        for item in section.items:
            if not item.extra:
                # Special: check if jaato-tui is toggled
                if "jaato-tui" in item.label and not item.locked:
                    install_tui = item.checked
                continue
            if not item.checked:
                continue
            if item.package == "tui":
                tui_extras.append(item.extra)
            else:
                server_extras.append(item.extra)

    server_extras_str = f"[{','.join(server_extras)}]" if server_extras else ""
    tui_extras_str = f"[{','.join(tui_extras)}]" if tui_extras else ""

    if source == InstallSource.EDITABLE:
        parts = [pip_path, "install"]
        parts.append(f'-e "jaato-sdk/."')
        parts.append(f'-e "jaato-server/.{server_extras_str}"')
        if install_tui:
            parts.append(f'-e "jaato-tui/.{tui_extras_str}"')
        return [" ".join(parts)]

    elif source == InstallSource.PYPI:
        parts = [pip_path, "install"]
        parts.append('"jaato-sdk"')
        parts.append(f'"jaato-server{server_extras_str}"')
        if install_tui:
            parts.append(f'"jaato-tui{tui_extras_str}"')
        return [" ".join(parts)]

    else:  # TEST_PYPI
        parts = [pip_path, "install"]
        parts.append("--index-url https://test.pypi.org/simple/")
        parts.append("--extra-index-url https://pypi.org/simple/")
        parts.append('"jaato-sdk"')
        parts.append(f'"jaato-server{server_extras_str}"')
        if install_tui:
            parts.append(f'"jaato-tui{tui_extras_str}"')
        return [" ".join(parts)]


def screen_review(
    commands: List[str],
    venv_path: Optional[str],
    source: InstallSource,
) -> Optional[bool]:
    """Screen 4: Review and confirm.

    Returns:
        True to install, None to go back or quit.
    """
    source_labels = {
        InstallSource.EDITABLE: "Editable (local sources)",
        InstallSource.PYPI: "PyPI (stable release)",
        InstallSource.TEST_PYPI: "Test PyPI (pre-release)",
    }

    while True:
        _clear()
        _write(_render_header())
        _write(f"  {_Ansi.BOLD}{_Ansi.CYAN}── Review {'─' * 40}{_Ansi.RESET}\n")
        _write(f"\n  Source: {_Ansi.BOLD}{source_labels[source]}{_Ansi.RESET}\n")
        if venv_path:
            _write(f"  Venv:   {_Ansi.BOLD}{venv_path}{_Ansi.RESET}\n")
        _write("\n")

        for cmd in commands:
            # Format the command for display with line wrapping
            _write(f"  {_Ansi.GREEN}${_Ansi.RESET} {cmd}\n")
        _write("\n")
        _write(_render_footer("review"))

        key = _read_key()

        if key.key == _Key.QUIT:
            return None
        if key.key == _Key.ESC:
            return None  # go back
        if key.key == _Key.ENTER:
            return True


# ---------------------------------------------------------------------------
# Installation execution
# ---------------------------------------------------------------------------


def _create_venv(path: str) -> str:
    """Create a virtual environment and return the pip path inside it."""
    venv_path = Path(path).resolve()
    _write(f"\n  Creating virtual environment at {venv_path}...\n")
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_path)],
        check=True,
    )

    if _IS_WINDOWS:
        pip_path = str(venv_path / "Scripts" / "pip")
    else:
        pip_path = str(venv_path / "bin" / "pip")

    # Upgrade pip in the new venv
    _write("  Upgrading pip...\n")
    subprocess.run(
        [pip_path, "install", "--upgrade", "pip"],
        check=True,
        capture_output=True,
    )

    return pip_path


def _run_install(commands: List[str]) -> bool:
    """Execute the pip install commands.

    Returns:
        True if all commands succeeded.
    """
    for cmd in commands:
        _write(f"\n  {_Ansi.GREEN}${_Ansi.RESET} {cmd}\n\n")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            _write(f"\n  {_Ansi.RED}Installation failed (exit code {result.returncode}){_Ansi.RESET}\n")
            return False
    return True


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def _detect_source_checkout() -> bool:
    """Check if we're running from a jaato source checkout."""
    script_dir = Path(__file__).resolve().parent
    return all(
        (script_dir / d).is_dir()
        for d in ("jaato-sdk", "jaato-server", "jaato-tui")
    )


def main() -> None:
    """Main installer entry point."""
    if sys.version_info < (3, 10):
        print(f"jaato requires Python 3.10+, found {sys.version_info.major}.{sys.version_info.minor}")
        sys.exit(1)

    _write(_Ansi.HIDE_CURSOR)
    try:
        _main_flow()
    except KeyboardInterrupt:
        _write(f"\n\n  {_Ansi.DIM}Cancelled.{_Ansi.RESET}\n")
    finally:
        _write(_Ansi.SHOW_CURSOR)


def _main_flow() -> None:
    """The main screen flow with back navigation."""
    is_checkout = _detect_source_checkout()
    sections = _build_sections()
    venv_choice: Optional[Tuple[bool, str]] = None
    source_choice: Optional[InstallSource] = None

    current_screen = 0  # 0=venv, 1=source, 2=features, 3=review

    while True:
        if current_screen == 0:
            result = screen_venv()
            if result is None:
                _write(f"\n  {_Ansi.DIM}Cancelled.{_Ansi.RESET}\n")
                return
            venv_choice = result
            current_screen = 1

        elif current_screen == 1:
            result = screen_source(is_checkout)
            if result is None:
                current_screen = 0  # go back
                continue
            source_choice = result
            current_screen = 2

        elif current_screen == 2:
            result = screen_features(sections)
            if result is None:
                current_screen = 1  # go back
                continue
            sections = result
            current_screen = 3

        elif current_screen == 3:
            create_venv, venv_path = venv_choice

            # Determine pip path
            if create_venv and venv_path:
                resolved = Path(venv_path).resolve()
                if _IS_WINDOWS:
                    pip_path = str(resolved / "Scripts" / "pip")
                else:
                    pip_path = str(resolved / "bin" / "pip")
            else:
                pip_path = f"{sys.executable} -m pip"
                venv_path = None

            commands = _build_pip_commands(sections, source_choice, pip_path)

            result = screen_review(commands, venv_path, source_choice)
            if result is None:
                current_screen = 2  # go back
                continue

            # Execute installation
            _clear()
            _write(_render_header())
            _write(f"  {_Ansi.BOLD}{_Ansi.CYAN}── Installing {'─' * 37}{_Ansi.RESET}\n")

            # Create venv if requested
            if create_venv and venv_path:
                resolved = Path(venv_path).resolve()
                if not resolved.exists():
                    pip_path = _create_venv(str(resolved))
                    # Rebuild commands with actual pip path
                    commands = _build_pip_commands(sections, source_choice, pip_path)
                else:
                    _write(f"\n  Venv already exists at {resolved}, reusing.\n")

            _write(_Ansi.SHOW_CURSOR)
            success = _run_install(commands)

            if success:
                _write(f"\n  {_Ansi.GREEN}{_Ansi.BOLD}Installation complete!{_Ansi.RESET}\n")
                if venv_path:
                    if _IS_WINDOWS:
                        activate = str(Path(venv_path).resolve() / "Scripts" / "activate")
                    else:
                        activate = f"source {Path(venv_path).resolve() / 'bin' / 'activate'}"
                    _write(f"\n  Activate the virtual environment:\n")
                    _write(f"  {_Ansi.GREEN}${_Ansi.RESET} {activate}\n")
            _write("\n")
            return


if __name__ == "__main__":
    main()
