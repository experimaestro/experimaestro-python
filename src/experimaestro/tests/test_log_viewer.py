"""Tests for the TUI log viewer's carriage-return handling"""

from experimaestro.scheduler.logs import apply_carriage_returns, read_log_slice
from experimaestro.tui.log_viewer import LogFile, _apply_carriage_returns


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
    assert reader._partial_line == "prog 10%"

    # Truncate
    log.write_text("")
    complete, partial = reader.read_new_content()
    assert complete == []
    assert partial == ""
    assert reader._partial_line == ""


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


# --- Shared log reading (scheduler/logs.py)


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
    assert reader.read_new_content() == ([" 20%"], "")
