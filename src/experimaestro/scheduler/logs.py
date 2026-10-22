"""Shared log-tailing utilities.

Pure file logic (no UI dependency) used by both the Textual TUI
(``tui/log_viewer.py``) and the web UI log endpoint (``webui/routes/logs.py``)
so that progress bars (tqdm) and tailing behave identically everywhere.

Logs are rendered as a terminal would show them (:func:`render_terminal`):
``\\r`` rewrites a line, cursor moves (``ESC[A``, used by nested tqdm bars)
go back to previous lines, and the other escape sequences (colors) are
dropped.
"""

import re
from pathlib import Path


# Default chunk size for reading file (64KB)
CHUNK_SIZE = 64 * 1024
# How many bytes to read from end of file initially
INITIAL_TAIL_SIZE = 256 * 1024  # 256KB
#: Beyond this size, the lines the cursor can go back to are no longer
#: re-read (they are shown as completed lines)
MAX_PENDING_SIZE = 256 * 1024


def apply_carriage_returns(line: str) -> str:
    """Collapse a line as a terminal would: keep only content after the last ``\\r``.

    tqdm-like progress bars rewrite the same line via ``\\r``; treat that as an
    overwrite rather than as a new line. Trailing ``\\r`` only move the cursor
    back (e.g. ``\\r\\n`` line ends), so the text before them is kept.
    """
    line = line.rstrip("\r")
    if "\r" in line:
        return line.rsplit("\r", 1)[-1]
    return line


# Backwards-compatible alias (previously a private helper in tui/log_viewer.py)
_apply_carriage_returns = apply_carriage_returns


#: CSI sequences (``ESC [ parameters final``), a CSI sequence cut at the end of
#: the data (completed by a later read), other escape sequences and the
#: control characters (but tabulations)
_CONTROL = re.compile(
    r"\x1b\[([0-9;?]*)[ -/]*([@-~])|\x1b\[[0-9;?]*[ -/]*$|\x1b[@-_]?"
    r"|[\x00-\x08\x0b-\x1f\x7f]"
)


class _Screen:
    """The lines of a terminal and its cursor"""

    def __init__(self):
        self.rows: list[str] = [""]
        #: Byte offset where the cursor first entered each row
        self.starts: list[int] = [0]
        self.row = 0
        self.col = 0

    def newline(self, offset: int) -> None:
        self.row += 1
        self.col = 0
        if self.row == len(self.rows):
            self.rows.append("")
            self.starts.append(offset)

    def write(self, text: str) -> None:
        """Write at the cursor; the end of the row is cleared (progress bars
        rewrite whole lines)"""
        if text:
            self.rows[self.row] = self.rows[self.row][: self.col].ljust(self.col) + text
            self.col += len(text)

    def control(self, match: re.Match) -> None:
        sequence = match.group(0)
        if sequence == "\r":
            self.col = 0
        elif sequence == "\b":
            self.col = max(0, self.col - 1)
        elif final := match.group(2):
            params = match.group(1)
            n = int(params) if params.isdigit() else 0
            match final:
                case "A" | "F":
                    self.row = max(0, self.row - max(n, 1))
                case "B" | "E":
                    self.row = min(len(self.rows) - 1, self.row + max(n, 1))
                case "C":
                    self.col += max(n, 1)
                case "D":
                    self.col = max(0, self.col - max(n, 1))
                case "G":
                    self.col = max(n, 1) - 1
                case "K":
                    line = self.rows[self.row]
                    match n:
                        case 0:
                            self.rows[self.row] = line[: self.col]
                        case 1:
                            self.rows[self.row] = " " * self.col + line[self.col :]
                        case _:
                            self.rows[self.row] = ""
            if final in "EF":
                self.col = 0

    def feed(self, text: str) -> None:
        """Process a line of text (without ``\\n``)"""
        position = 0
        for match in _CONTROL.finditer(text):
            self.write(text[position : match.start()])
            self.control(match)
            position = match.end()
        self.write(text[position:])


def render_terminal(data: bytes) -> tuple[list[str], list[str], int]:
    """Render log data as a terminal would show it

    Returns ``(completed, pending, offset)``: the completed lines, the lines
    still being written (the line of the cursor and the ones below it, which
    progress bars rewrite) and the byte offset where the pending lines start
    (rendering again from there gives them with their next updates).
    """
    screen = _Screen()
    start = 0
    for index, segment in enumerate(data.split(b"\n")):
        if index:
            screen.newline(start)
        screen.feed(segment.decode(errors="replace"))
        start += len(segment) + 1

    row = screen.row
    if len(data) - screen.starts[row] > MAX_PENDING_SIZE:
        # Too much to read again: only the last line stays pending
        row = len(screen.rows) - 1
    return screen.rows[:row], screen.rows[row:], screen.starts[row]


class LogFile:
    """Efficient log file reader that tracks position and watches for changes"""

    def __init__(self, path: str):
        self.path = Path(path)
        self.position = 0
        self.size = 0
        # Data of the lines still being written (see render_terminal), read
        # again with the new content
        self._pending = b""
        # The pending lines, as rendered
        self._partial = ""
        self._update_size()

    def _update_size(self) -> None:
        """Update the known file size"""
        try:
            self.size = self.path.stat().st_size
        except OSError:
            self.size = 0

    def _process_content(self, content: bytes) -> tuple[list[str], str]:
        """Render the pending data followed by new content

        Returns the newly completed lines and the lines still being written
        (joined with ``\\n``).
        """
        data = self._pending + content
        completed, pending, offset = render_terminal(data)
        self._pending = data[offset:]
        self._partial = "\n".join(pending)
        return completed, self._partial

    def read_tail(self, max_bytes: int = INITIAL_TAIL_SIZE) -> tuple[list[str], str]:
        """Read the last N bytes of the file.

        Returns ``(complete_lines, partial)``. ``partial`` holds the lines
        still being written (no trailing \\n yet, or rewritten by progress
        bars), joined with ``\\n``, and may be empty.
        """
        if not self.path.exists():
            return [], ""

        self._update_size()
        if self.size == 0:
            return [], ""

        try:
            with open(self.path, "rb") as f:
                # Start from max_bytes before end, or beginning
                start_pos = max(0, self.size - max_bytes)
                f.seek(start_pos)

                # If we're not at the start, skip to the next newline
                if start_pos > 0:
                    f.readline()  # Skip partial line

                content = f.read()
                self.position = f.tell()
                # Reset the pending data since we're reading from a known
                # boundary
                self._pending = b""
                return self._process_content(content)
        except Exception:
            return [], ""

    def read_new_content(self) -> tuple[list[str], str]:
        """Read any new content since last read.

        Returns ``(new_complete_lines, partial)``. ``partial`` holds the lines
        still being written (which may have changed even when no new complete
        lines were produced).
        """
        if not self.path.exists():
            return [], self._partial

        self._update_size()

        # File was truncated or rotated
        if self.size < self.position:
            self.position = 0
            self._pending = b""
            self._partial = ""

        if self.position >= self.size:
            return [], self._partial

        try:
            with open(self.path, "rb") as f:
                f.seek(self.position)
                content = f.read()
                self.position = f.tell()
                return self._process_content(content)
        except Exception:
            return [], self._partial

    def has_new_content(self) -> bool:
        """Check if there's new content without reading it"""
        self._update_size()
        return self.size > self.position


def read_log_slice(
    path: Path | str,
    offset: int | None = None,
    tail_bytes: int = INITIAL_TAIL_SIZE,
) -> dict:
    """Stateless log read suitable for HTTP polling.

    Args:
        path: log file path
        offset: byte offset to read from. ``None`` or a negative value means
            "tail": read the last ``tail_bytes`` (aligned to a line boundary).
        tail_bytes: how many bytes to read when tailing.

    Returns a dict ``{content, partial, offset, size}`` (the log being
    rendered as a terminal would, see :func:`render_terminal`):

    - ``content`` holds the completed lines (each ending with ``\\n``):
      clients append it to what they show;
    - ``partial`` holds the lines still being written (the last one, or the
      ones progress bars go back to), joined with ``\\n``: clients show it
      after the content and *replace* it on each poll;
    - ``offset`` is the byte offset to pass on the next poll. It stops at the
      start of the partial lines, which are read again (with their new
      updates) by the next poll.
    """
    empty = {"content": "", "partial": "", "offset": 0, "size": 0}
    p = Path(path)
    if not p.exists():
        return empty

    try:
        size = p.stat().st_size
    except OSError:
        return empty

    # Handle truncation/rotation: an offset past EOF restarts from the tail.
    tail = offset is None or offset < 0 or offset > size

    try:
        with open(p, "rb") as f:
            if tail:
                start = max(0, size - tail_bytes)
                f.seek(start)
                if start > 0:
                    f.readline()  # align to a line boundary
            else:
                f.seek(offset)
            start = f.tell()
            data = f.read()
    except Exception:
        return {**empty, "offset": offset or 0, "size": size}

    completed, pending, pending_offset = render_terminal(data)
    return {
        "content": "".join(f"{line}\n" for line in completed),
        "partial": "\n".join(pending),
        "offset": start + pending_offset,
        "size": size,
    }
