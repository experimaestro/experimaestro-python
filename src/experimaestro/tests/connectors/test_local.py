from pathlib import Path
import sys
from experimaestro.connectors import Redirect
from experimaestro.connectors.local import LocalProcessBuilder
from experimaestro.tests.connectors.utils import OutputCaptureHandler
from experimaestro.tests.utils import TemporaryDirectory


def test_local_simpole():
    builder = LocalProcessBuilder()

    with TemporaryDirectory() as tmp:
        builder.command = [
            sys.executable,
            Path(__file__).parent / "bin" / "executable.py",
        ]
        builder.workingDirectory = tmp
        outpath = tmp / "stdout.txt"
        builder.stdout = Redirect.file(outpath)

        p = builder.start()
        p.wait()

        assert outpath.read_text().strip() == "hello world"


def test_local_pipe():
    builder = LocalProcessBuilder()

    with TemporaryDirectory() as tmp:
        builder.command = [
            sys.executable,
            Path(__file__).parent / "bin" / "executable.py",
        ]
        builder.workingDirectory = tmp

        handler = OutputCaptureHandler()
        builder.stdout = Redirect.pipe(handler)

        p = builder.start()
        assert p.wait() == 0

        assert handler.output.strip() == b"hello world"
