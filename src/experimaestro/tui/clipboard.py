"""Clipboard utilities for TUI that work in remote/tmux environments.

This module provides clipboard functionality using OSC 52 escape sequences,
which work through SSH and tmux. Falls back to pyperclip if OSC 52 fails.

OSC 52 is supported by most modern terminals:
- iTerm2 (macOS)
- kitty
- alacritty
- Windows Terminal
- tmux (requires configuration, see below)
- many others

For tmux, add to ~/.tmux.conf:
    set -g set-clipboard on
    set -g allow-passthrough on   # Required for tmux 3.2+
"""

from __future__ import annotations

import base64
import os
import sys
from typing import Literal

# Terminal capability detection
_osc52_supported: bool | None = None


def _detect_osc52_support() -> bool:
    """Detect if the terminal likely supports OSC 52.

    This is a heuristic - OSC 52 support cannot be reliably detected,
    so we check for known-supporting terminals.
    """
    term = os.environ.get("TERM", "")
    term_program = os.environ.get("TERM_PROGRAM", "")
    colorterm = os.environ.get("COLORTERM", "")

    # Known supporting terminals
    supporting_terms = {
        "xterm",
        "xterm-256color",
        "screen",
        "screen-256color",
        "tmux",
        "tmux-256color",
        "alacritty",
        "kitty",
        "foot",
    }

    supporting_programs = {
        "iTerm.app",
        "Apple_Terminal",
        "WezTerm",
        "vscode",
        "Hyper",
        "mintty",
    }

    # Check if we're in a known supporting terminal
    if term in supporting_terms:
        return True
    if term_program in supporting_programs:
        return True
    if colorterm in ("truecolor", "24bit"):
        return True

    # Check for tmux or screen (which can forward OSC 52)
    if os.environ.get("TMUX"):
        return True
    if term.startswith("screen"):
        return True

    # Default to trying OSC 52 - most modern terminals support it
    return True


def _write_osc52(text: str, target: Literal["c", "p", "s"] = "c") -> bool:
    """Write text to clipboard using OSC 52 escape sequence.

    Args:
        text: Text to copy to clipboard
        target: Clipboard target:
            - "c" = system clipboard (default)
            - "p" = primary selection (X11)
            - "s" = secondary selection

    Returns:
        True if the escape sequence was written successfully
    """
    try:
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")

        # Build OSC 52 sequence
        # Format: ESC ] 52 ; target ; base64-data ST
        # ST (String Terminator) = ESC \ (\033\\)
        # Some terminals also accept BEL (\a) but ST is more universal
        osc52 = f"\033]52;{target};{encoded}\033\\"

        # For tmux, we need to wrap in a DCS passthrough sequence
        # Format: DCS tmux ; <escaped_content> ST
        # All ESC characters in content must be doubled
        if os.environ.get("TMUX"):
            # Double all ESC characters in the OSC 52 sequence
            escaped = osc52.replace("\033", "\033\033")
            sequence = f"\033Ptmux;{escaped}\033\\"
        else:
            sequence = osc52

        # Write to stdout/tty
        # Try to write directly to the tty to bypass any buffering
        try:
            with open("/dev/tty", "w") as tty:
                tty.write(sequence)
                tty.flush()
        except OSError:
            # Fall back to stdout
            sys.stdout.write(sequence)
            sys.stdout.flush()

        return True
    except Exception:
        return False


def _copy_pyperclip(text: str) -> bool:
    """Copy text using pyperclip.

    Returns:
        True if copy succeeded
    """
    try:
        import pyperclip

        pyperclip.copy(text)
        return True
    except Exception:
        return False


def copy(text: str) -> bool:
    """Copy text to clipboard.

    Tries pyperclip first (reliable for local), falls back to OSC 52
    for remote/tmux environments where pyperclip doesn't work.

    Args:
        text: Text to copy to clipboard

    Returns:
        True if copy likely succeeded (OSC 52 success cannot be confirmed)
    """
    global _osc52_supported

    # Try pyperclip first (works reliably when local clipboard is available)
    if _copy_pyperclip(text):
        return True

    # Fall back to OSC 52 for remote/tmux scenarios
    if _osc52_supported is None:
        _osc52_supported = _detect_osc52_support()

    if _osc52_supported:
        return _write_osc52(text)

    return False


def copy_with_fallback(text: str) -> tuple[bool, str | None]:
    """Copy text to clipboard with detailed result.

    Args:
        text: Text to copy to clipboard

    Returns:
        Tuple of (success, method_used)
        method_used is "pyperclip", "osc52", or None if failed
    """
    global _osc52_supported

    # Try pyperclip first
    if _copy_pyperclip(text):
        return True, "pyperclip"

    # Fall back to OSC 52
    if _osc52_supported is None:
        _osc52_supported = _detect_osc52_support()

    if _osc52_supported and _write_osc52(text):
        return True, "osc52"

    return False, None
