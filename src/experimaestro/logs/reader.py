"""Shared log-tailing utilities.

Pure logic (no UI dependency) used by both the Textual TUI
(``tui/log_viewer.py``) and the web UI log endpoint (``webui/routes/logs.py``)
so that progress bars (tqdm) and tailing behave identically everywhere.

Logs are rendered as a terminal would show them (:func:`render_terminal`):
``\\r`` rewrites a line, cursor moves (``ESC[A``, used by nested tqdm bars)
go back to previous lines, and the other escape sequences (colors) are
dropped.

Logs are read from a :class:`~experimaestro.logs.source.LogSource`
(local file, or remote file read by blocks): their end first, then the new
content (following) or the earlier one (scrolling up), by pages of
:data:`PAGE_SIZE` bytes.
"""

import re
from pathlib import Path

from experimaestro.logs.source import LocalLogSource, LogSource

#: Bytes read when opening a log (its end), and when loading earlier content
PAGE_SIZE = 128 * 1024
#: Beyond this many new bytes, readers following a log skip to its end
MAX_FORWARD_SIZE = 1024 * 1024
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


def _source(source: LogSource | Path | str) -> LogSource:
    return source if isinstance(source, LogSource) else LocalLogSource(source)


def _align(data: bytes, start: int) -> tuple[bytes, int]:
    """Start the data at a line (unless at the start of the file)"""
    if start > 0 and (newline := data.find(b"\n")) >= 0:
        return data[newline + 1 :], start + newline + 1
    return data, start


def _tail(source: LogSource, size: int, max_bytes: int) -> tuple[bytes, int]:
    """The end of a log (from a line start), and where it starts"""
    start = max(0, size - max_bytes)
    return _align(source.read(start, size - start), start)


def _lines_before(source: LogSource, end: int) -> tuple[list[str], int]:
    """The lines of the page ending at ``end`` (a line start), and where they
    start"""
    start = max(0, end - PAGE_SIZE)
    data, start = _align(source.read(start, end - start), start)
    completed, pending, _ = render_terminal(data)
    lines = completed + pending
    if lines and not lines[-1]:
        # The data ends with a new line
        lines.pop()
    return lines, start


class LogFile:
    """Reads a log, keeping track of what it read (used by the TUI)

    The end of the log is read first (:meth:`read_tail`), then its new content
    (:meth:`read_new_content`) or its earlier one (:meth:`read_before`). When
    the file is replaced (see :attr:`LogSource.generation`), its end is read
    again: readers compare :attr:`generation` before and after reading to
    know that the lines they show are stale.
    """

    def __init__(self, source: LogSource | Path | str):
        self.source = _source(source)
        #: The generation of the source of what was read
        self.generation = self.source.generation
        #: Where the next read starts (the lines still being written)
        self.position = 0
        #: Where what was read starts (earlier content is before)
        self.start = 0
        self._size: int | None = None
        self._partial = ""

    @property
    def exists(self) -> bool:
        """Whether the log file existed when last read"""
        return self._size is not None

    def _render(self, data: bytes) -> tuple[list[str], str]:
        completed, pending, offset = render_terminal(data)
        self.position += offset
        self._partial = "\n".join(pending)
        return completed, self._partial

    def read_tail(self, max_bytes: int = PAGE_SIZE) -> tuple[list[str], str]:
        """Read the end of the log

        Returns ``(complete_lines, partial)``. ``partial`` holds the lines
        still being written (no trailing \\n yet, or rewritten by progress
        bars), joined with ``\\n``, and may be empty.
        """
        self.source.touch()
        size = self._size = self.source.size()
        self.generation = self.source.generation
        self._partial = ""
        self.position = self.start = 0
        if not size:
            return [], ""
        data, self.start = _tail(self.source, size, max_bytes)
        self.position = self.start
        return self._render(data)

    def read_new_content(self) -> tuple[list[str], str]:
        """Read the new content of the log

        Returns ``(new_complete_lines, partial)``. ``partial`` holds the lines
        still being written (which may have changed even when no new complete
        lines were produced). Reads the end of the log again (see
        :meth:`read_tail`) when the file was replaced, or when there is too
        much new content.
        """
        size = self.source.size()
        if (
            self.source.generation != self.generation
            or (size or 0) - self.position > MAX_FORWARD_SIZE
        ):
            return self.read_tail()
        if size == self._size or size is None:
            self._size = size
            return [], self._partial
        self._size = size
        return self._render(self.source.read(self.position, size - self.position))

    def read_before(self) -> list[str]:
        """Read the lines before what was read (empty at the start)"""
        if self.start == 0:
            return []
        self.source.touch()
        lines, self.start = _lines_before(self.source, self.start)
        return lines


def empty_log_slice(size: int = 0, generation: int = 0, exists: bool = False) -> dict:
    """The answer of :func:`read_log_slice` for an empty (or missing) log"""
    return {
        "exists": exists,
        "content": "",
        "partial": "",
        "offset": 0,
        "start": 0,
        "size": size,
        "generation": generation,
        "tail": True,
    }


def read_log_slice(
    source: LogSource | Path | str,
    offset: int | None = None,
    *,
    before: int | None = None,
    generation: int | None = None,
    tail_bytes: int = PAGE_SIZE,
) -> dict:
    """Stateless log read suitable for HTTP polling.

    Args:
        source: the log
        offset: byte offset to read from (the ``offset`` returned by the
            previous read). ``None`` or a negative value means "tail": read
            the last ``tail_bytes`` (aligned to a line boundary).
        before: read the lines before this offset (the ``start`` returned by a
            previous read) instead
        generation: the ``generation`` returned by the previous read: the end
            of the log is read when the file was replaced since
        tail_bytes: how many bytes to read when tailing.

    Returns a dict ``{content, partial, offset, start, size, generation,
    tail}`` (the log being rendered as a terminal would, see
    :func:`render_terminal`):

    - ``content`` holds the completed lines (each ending with ``\\n``):
      clients append it to what they show (or prepend it when reading
      ``before``), or replace what they show with it when ``tail`` is true
      (the end of the log was read);
    - ``partial`` holds the lines still being written (the last one, or the
      ones progress bars go back to), joined with ``\\n``: clients show it
      after the content and *replace* it on each poll;
    - ``offset`` is the byte offset to pass on the next poll. It stops at the
      start of the partial lines, which are read again (with their new
      updates) by the next poll;
    - ``start`` is where the content read starts (to pass as ``before``);
    - ``exists`` tells whether the log file exists.
    """
    source = _source(source)
    if offset is None or offset < 0:
        source.touch()
    size = source.size()
    stale = generation is not None and generation != source.generation
    result = empty_log_slice(size or 0, source.generation, size is not None)
    if not size:
        return result

    if before is not None:
        if stale:
            return result
        lines, start = _lines_before(source, min(before, size))
        return {
            **result,
            "content": "".join(f"{line}\n" for line in lines),
            "offset": before,
            "start": start,
            "tail": False,
        }

    tail = (
        offset is None
        or offset < 0
        or offset > size
        or stale
        or size - offset > MAX_FORWARD_SIZE
    )
    if tail:
        data, start = _tail(source, size, tail_bytes)
    else:
        start = offset
        data = source.read(offset, size - offset)
    completed, pending, pending_offset = render_terminal(data)
    return {
        **result,
        "content": "".join(f"{line}\n" for line in completed),
        "partial": "\n".join(pending),
        "offset": start + pending_offset,
        "start": start,
        "tail": tail,
    }
