# Fake experiment to test the web interface

import tempfile
import logging
from pathlib import Path
import click
import os
import time
import sys
from experimaestro import experiment, Task, progress
from experimaestro.core.arguments import Param


class ControlledTask(Task):
    path: Param[Path]

    def execute(self):
        self.path.unlink(missing_ok=True)
        os.mkfifo(self.path)

        with self.path.open("r") as fp:
            for line in fp:
                try:
                    command, *args = line.strip().split()
                    if command == "exit":
                        sys.exit(int(args[0]))
                    elif command == "progress":
                        value = float(args[0])
                        level = int(args[1]) if len(args) >= 2 else 0
                        desc = "".join(args[2:]) if len(args) >= 3 else None
                        progress(value, level=level, desc=desc)
                except Exception:
                    logging.exception("Error while intrepreting command: %s", line)


@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--keep", is_flag=True, help="Keep temporary directory")
@click.option("--port", type=int, default=12345, help="Port for experimaestro server")
@click.command()
def cli(keep: bool, debug: bool, port: int):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

    maindir = tempfile.TemporaryDirectory()
    if keep:
        maindir._finalizer.detach()

    with maindir:
        maindir = Path(maindir.name)
        xpdir = maindir / "xpm"
        logging.info("Running in %s", xpdir)
        with experiment(Path(xpdir) / "", "web-interface", port=port) as xp:
            socketpath = maindir / "socket1"
            logging.info("Controlled task with %s", socketpath)
            ControlledTask(path=socketpath).submit()

            while not socketpath.exists():
                time.sleep(0.1)

            print("\n\n --- Ready to take commands ---")
            with (socketpath).open("w") as fp:
                while True:
                    command = input("Command: ")
                    fp.write(command)
                    fp.write("\n")
                    fp.flush()


if __name__ == "__main__":
    cli()
