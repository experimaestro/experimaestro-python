"""Tests for the TUI log viewer's carriage-return handling"""

from experimaestro.logs.reader import (
    apply_carriage_returns,
    read_log_slice,
    render_terminal,
)
from experimaestro.logs.reader import LogFile, _apply_carriage_returns


def test_apply_carriage_returns_keeps_last_segment():
    assert _apply_carriage_returns(" 10%\r 20%\r 30%") == " 30%"


def test_apply_carriage_returns_no_cr():
    assert _apply_carriage_returns("hello") == "hello"


def test_log_file_handles_tqdm_progress(tmp_path):
    """A typical tqdm sequence should collapse into a single completed line."""
    log = tmp_path / "task.out"
    log.write_text("\rStep 0%\rStep 50%\rStep 100%\nDone\n")

    reader = LogFile(str(log))
    complete, partial = reader.read_tail()
    assert complete == ["Step 100%", "Done"]
    assert partial == ""


def test_log_file_partial_line_buffering(tmp_path):
    """Partial last line (no \\n) should be returned as partial, not as a full line."""
    log = tmp_path / "task.out"
    log.write_text("first\n\rprog 10%")

    reader = LogFile(str(log))
    complete, partial = reader.read_tail()
    assert complete == ["first"]
    assert partial == "prog 10%"


def test_log_file_progress_across_reads(tmp_path):
    """Progress updates split across reads should overwrite, not stack."""
    log = tmp_path / "task.out"
    log.write_text("start\n\rprog 10%")

    reader = LogFile(str(log))
    complete, partial = reader.read_tail()
    assert complete == ["start"]
    assert partial == "prog 10%"

    # Append more progress updates and a final newline + next line
    with open(log, "a") as f:
        f.write("\rprog 50%\rprog 100%\nnext line\n")

    complete, partial = reader.read_new_content()
    assert complete == ["prog 100%", "next line"]
    assert partial == ""


def test_log_file_truncation_resets_partial(tmp_path):
    log = tmp_path / "task.out"
    log.write_text("start\n\rprog 10%")

    reader = LogFile(str(log))
    reader.read_tail()
    assert reader._partial == "prog 10%"

    # Truncate
    log.write_text("")
    complete, partial = reader.read_new_content()
    assert complete == []
    assert partial == ""
    assert reader.position == 0


def test_log_file_partial_within_chunk(tmp_path):
    """Multiple \\r updates inside one read chunk collapse to the last."""
    log = tmp_path / "task.out"
    log.write_text("")

    reader = LogFile(str(log))
    reader.read_tail()

    # Write a chunk containing multiple \r updates and no trailing \n
    with open(log, "a") as f:
        f.write("\rA\rB\rC")

    complete, partial = reader.read_new_content()
    assert complete == []
    assert partial == "C"


# --- Shared log reading (logs/reader.py)


def test_apply_carriage_returns_trailing_cr():
    """``\\r`` only moves the cursor: a terminal still shows the text before it
    (e.g. ``\\r\\n`` line ends, or a progress bar ending its updates with ``\\r``)"""
    assert apply_carriage_returns(" 50%\r") == " 50%"
    assert apply_carriage_returns("done\r") == "done"


def test_read_log_slice_progress_across_polls(tmp_path):
    """Polls read from the returned offset can be appended (as the web client
    does) and show a rewritten line once: the line being written is returned
    apart (``partial``, replaced by the next poll) and read again next time"""
    log = tmp_path / "task.err"
    log.write_text("start\n\r 10%")
    first = read_log_slice(log)
    assert (first["content"], first["partial"]) == ("start\n", " 10%")
    with log.open("a", newline="") as out:
        out.write("\r 20%\r 30%")
    second = read_log_slice(log, first["offset"])
    assert (second["content"], second["partial"]) == ("", " 30%")
    with log.open("a", newline="") as out:
        out.write("\ndone\n")
    third = read_log_slice(log, second["offset"])
    assert first["content"] + second["content"] + third["content"] == (
        "start\n 30%\ndone\n"
    )
    assert third["partial"] == ""


def test_log_file_progress_ending_with_cr(tmp_path):
    """A progress bar ending its updates with ``\\r`` (instead of starting
    them with it) is overwritten by the next update, also across reads"""
    log = tmp_path / "task.err"
    log.write_text(" 10%\r")
    reader = LogFile(str(log))
    assert reader.read_tail() == ([], " 10%")
    with log.open("a", newline="") as out:
        out.write(" 20%\r")
    assert reader.read_new_content() == ([], " 20%")
    with log.open("a", newline="") as out:
        out.write("\n")
    # Kept pending until another line (progress bars can go back to it)
    assert reader.read_new_content() == ([], " 20%\n")
    with log.open("a", newline="") as out:
        out.write("done\n")
    assert reader.read_new_content() == ([" 20%", "done"], "")


def test_render_terminal_cursor_up():
    """Nested tqdm bars: the inner bar goes back up after each update, the
    outer one is rewritten from there with ``\\r``"""
    data = (
        "start\nouter:   0%\n"
        "inner:  50%\x1b[A\n"
        "inner: 100%\x1b[A\router:  50%\n"
        "\x1b[Kinner:  50%\x1b[A"
    ).encode()
    completed, pending, offset = render_terminal(data)
    assert completed == ["start"]
    assert pending == ["outer:  50%", "inner:  50%"]
    assert data[offset:].startswith(b"outer:   0%")


def test_render_terminal_drops_escape_sequences():
    """Colors are dropped, an escape sequence cut by a read is not shown"""
    assert render_terminal(b"\x1b[31mred\x1b[0m\ncut\x1b[") == (
        ["red"],
        ["cut"],
        13,
    )


def test_read_log_slice_nested_progress_across_polls(tmp_path):
    """The lines a progress bar goes back to are read again by the next poll"""
    log = tmp_path / "task.err"
    log.write_text("start\n")
    first = read_log_slice(log)
    contents = [first["content"]]
    offset = first["offset"]
    for update in ("10", "20", "30"):
        with log.open("a", newline="") as out:
            out.write(f"\nEpoch: {update}%\x1b[A")
        poll = read_log_slice(log, offset)
        contents.append(poll["content"])
        offset = poll["offset"]
        assert poll["partial"] == f"\nEpoch: {update}%"
    with log.open("a", newline="") as out:
        out.write("\n\ndone\n")
    last = read_log_slice(log, offset)
    assert "".join(contents) + last["content"] == "start\n\nEpoch: 30%\ndone\n"
    assert last["partial"] == ""


def test_log_file_nested_progress(tmp_path):
    """The TUI reader shows the lines a progress bar goes back to as pending"""
    log = tmp_path / "task.err"
    log.write_text("start\n\nEpoch: 10%\x1b[A")
    reader = LogFile(str(log))
    assert reader.read_tail() == (["start"], "\nEpoch: 10%")
    with log.open("a", newline="") as out:
        out.write("\nEpoch: 20%\x1b[A")
    assert reader.read_new_content() == ([], "\nEpoch: 20%")
    with log.open("a", newline="") as out:
        out.write("\n\ndone\n")
    assert reader.read_new_content() == (["", "Epoch: 20%", "done"], "")


def _nested_tqdm_writes() -> list[str]:
    """The writes of nested tqdm bars (each flushed on its own): the inner bar
    is drawn on the next line (``\\n``, bar, ``ESC[A``), the outer one is
    rewritten in place, a log line is printed in between"""
    writes = ["start\n", "\router: 0/3"]
    for epoch in range(3):
        for step in range(2):
            writes += ["\n", f"\rinner {epoch}: {step}/2", "\x1b[A"]
        writes += ["\n", "\r\x1b[K", "\x1b[A"]
        writes.append(f"\router: {epoch + 1}/3")
        if epoch == 0:
            writes.append("INFO: logged\n")
    return writes + ["\n", "done\n"]


def test_read_log_slice_nested_progress_cut_anywhere(tmp_path):
    """Polling between any two writes of nested bars shows each line once"""
    log = tmp_path / "task.err"
    log.write_bytes(b"")
    writes = _nested_tqdm_writes()
    expected, pending, _ = render_terminal("".join(writes).encode())
    shown, partial, offset = "", "", None
    for write in writes:
        with log.open("a", newline="") as out:
            out.write(write)
        poll = read_log_slice(log, offset)
        shown = poll["content"] if poll["tail"] else shown + poll["content"]
        partial, offset = poll["partial"], poll["offset"]
    assert (shown + partial).split("\n") == expected + pending


def test_log_file_nested_progress_cut_anywhere(tmp_path):
    """The TUI reader shows each line of nested bars once"""
    log = tmp_path / "task.err"
    log.write_bytes(b"")
    writes = _nested_tqdm_writes()
    expected, pending, _ = render_terminal("".join(writes).encode())
    reader = LogFile(str(log))
    shown, partial = reader.read_tail()
    for write in writes:
        with log.open("a", newline="") as out:
            out.write(write)
        lines, partial = reader.read_new_content()
        shown += lines
    assert shown + partial.split("\n") == expected + pending


def test_render_terminal_crlf_lines_are_completed():
    """Lines ending with ``\\r\\n`` are not progress bars"""
    assert render_terminal(b"one\r\ntwo\r\n") == (["one", "two"], [""], 10)


def test_read_log_slice_rereads_tail_when_completed_lines_rewritten(tmp_path):
    """A poll going back up to lines already sent as completed reads the end
    of the log again (the client replaces what it shows)"""
    log = tmp_path / "task.err"
    log.write_bytes(b"start\nouter: 1\ninner")
    offset = len(b"start\nouter: 1\n")
    with log.open("a", newline="") as out:
        out.write("\x1b[A\router: 2")
    poll = read_log_slice(log, offset)
    assert poll["tail"]
    assert (poll["content"], poll["partial"]) == ("start\n", "outer: 2\ninner")
