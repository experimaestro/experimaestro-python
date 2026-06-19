"""Shared log-tailing utilities.

Pure file logic (no UI dependency) used by both the Textual TUI
(``tui/log_viewer.py``) and the web UI log endpoint (``webui/routes/logs.py``)
so that ``\\r``/tqdm collapsing and tailing behave identically everywhere.
"""

from pathlib import Path
from typing import Optional


# Default chunk size for reading file (64KB)
CHUNK_SIZE = 64 * 1024
# How many bytes to read from end of file initially
INITIAL_TAIL_SIZE = 256 * 1024  # 256KB


def apply_carriage_returns(line: str) -> str:
    """Collapse a line as a terminal would: keep only content after the last ``\\r``.

    tqdm-like progress bars rewrite the same line via ``\\r``; treat that as an
    overwrite rather than as a new line.
    """
    if "\r" in line:
        return line.rsplit("\r", 1)[-1]
    return line


# Backwards-compatible alias (previously a private helper in tui/log_viewer.py)
_apply_carriage_returns = apply_carriage_returns


class LogFile:
    """Efficient log file reader that tracks position and watches for changes"""

    def __init__(self, path: str):
        self.path = Path(path)
        self.position = 0
        self.size = 0
        # Buffer for the last incomplete line (no trailing \n yet).
        # Stored already collapsed for any \r updates seen so far.
        self._partial_line = ""
        self._update_size()

    def _update_size(self) -> None:
        """Update the known file size"""
        try:
            self.size = self.path.stat().st_size
        except OSError:
            self.size = 0

    def _process_content(self, content: str) -> tuple[list[str], str]:
        """Split a chunk into completed lines plus a trailing partial line.

        Carriage returns are handled like a terminal would: within each
        \\n-terminated line, only the substring after the last \\r is kept.
        """
        combined = self._partial_line + content
        parts = combined.split("\n")
        # parts[-1] is the partial trailing line (empty if combined ended in \n)
        complete = [apply_carriage_returns(line) for line in parts[:-1]]
        partial = apply_carriage_returns(parts[-1])
        self._partial_line = partial
        return complete, partial

    def read_tail(self, max_bytes: int = INITIAL_TAIL_SIZE) -> tuple[list[str], str]:
        """Read the last N bytes of the file.

        Returns ``(complete_lines, partial_line)``. ``partial_line`` is the
        in-progress last line (no trailing \\n yet) and may be empty.
        """
        if not self.path.exists():
            return [], ""

        self._update_size()
        if self.size == 0:
            return [], ""

        try:
            with open(self.path, "r", errors="replace", newline="") as f:
                # Start from max_bytes before end, or beginning
                start_pos = max(0, self.size - max_bytes)
                f.seek(start_pos)

                # If we're not at the start, skip to the next newline
                if start_pos > 0:
                    f.readline()  # Skip partial line

                content = f.read()
                self.position = f.tell()
                # Reset partial buffer since we're reading from a known boundary
                self._partial_line = ""
                return self._process_content(content)
        except Exception:
            return [], ""

    def read_new_content(self) -> tuple[list[str], str]:
        """Read any new content since last read.

        Returns ``(new_complete_lines, partial_line)``. ``partial_line`` is the
        current in-progress last line (which may have changed even when no new
        complete lines were produced).
        """
        if not self.path.exists():
            return [], self._partial_line

        self._update_size()

        # File was truncated or rotated
        if self.size < self.position:
            self.position = 0
            self._partial_line = ""

        if self.position >= self.size:
            return [], self._partial_line

        try:
            with open(self.path, "r", errors="replace", newline="") as f:
                f.seek(self.position)
                content = f.read()
                self.position = f.tell()
                return self._process_content(content)
        except Exception:
            return [], self._partial_line

    def has_new_content(self) -> bool:
        """Check if there's new content without reading it"""
        self._update_size()
        return self.size > self.position


def read_log_slice(
    path: Path | str,
    offset: Optional[int] = None,
    tail_bytes: int = INITIAL_TAIL_SIZE,
) -> dict:
    """Stateless log read suitable for HTTP polling.

    Args:
        path: log file path
        offset: byte offset to read from. ``None`` or a negative value means
            "tail": read the last ``tail_bytes`` (aligned to a line boundary).
        tail_bytes: how many bytes to read when tailing.

    Returns a dict ``{content, offset, size}`` where ``content`` is the decoded
    text (with ``\\r`` progress lines collapsed) and ``offset`` is the byte
    offset to pass on the next poll.
    """
    p = Path(path)
    if not p.exists():
        return {"content": "", "offset": 0, "size": 0}

    try:
        size = p.stat().st_size
    except OSError:
        return {"content": "", "offset": 0, "size": 0}

    # Handle truncation/rotation: an offset past EOF restarts from the tail.
    tail = offset is None or offset < 0 or offset > size

    try:
        with open(p, "r", errors="replace", newline="") as f:
            if tail:
                start = max(0, size - tail_bytes)
                f.seek(start)
                if start > 0:
                    f.readline()  # align to a line boundary
            else:
                f.seek(offset)
            content = f.read()
            new_offset = f.tell()
    except Exception:
        return {"content": "", "offset": offset or 0, "size": size}

    # Collapse \r progress lines per completed line, preserve a trailing partial.
    lines = content.split("\n")
    collapsed = "\n".join(apply_carriage_returns(line) for line in lines)
    return {"content": collapsed, "offset": new_offset, "size": size}
