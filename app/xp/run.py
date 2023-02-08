# Fake experiment to test the web interface

import tempfile
import logging
from pathlib import Path
import threading
import click
import os
import time
import sys
from experimaestro import (
    experiment,
    Config,
    Task,
    progress,
    pathgenerator,
    Annotated,
    tagspath,
)
from experimaestro.core.arguments import Param
from experimaestro.scheduler.services import WebService
from experimaestro.utils import cleanupdir


class ControlledTask(Task):
    path: Param[Path]
    tensorboard: Param[bool]
    logdir: Annotated[Path, pathgenerator("runs")]

    def execute(self):
        self.path.unlink(missing_ok=True)
        os.mkfifo(self.path)

        with self.path.open("r") as fp:
            if self.tensorboard:
                import torch
                from torch.utils.tensorboard import SummaryWriter

                writer = SummaryWriter(self.logdir)
                for it in range(100):
                    writer.add_scalar("loss", (0.999 * it) * torch.randn((1,)), it)
                writer.flush()

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
                    logging.exception("Error while interpreting command: %s", line)


class TensorboardService(WebService):
    id = "tensorboard"

    def __init__(self, path: Path):
        super().__init__()

        self.path = path
        cleanupdir(self.path)
        self.path.mkdir(exist_ok=True, parents=True)
        logging.info("You can monitor learning with:")
        logging.info("tensorboard --logdir=%s", self.path)
        self.url = None

    def add(self, config: Config, path: Path):
        (self.path / tagspath(config)).symlink_to(path)

    def description(self):
        return "Tensorboard service"

    def close(self):
        if self.server:
            self.server.shutdown()

    def _serve(self, running: threading.Event):
        import tensorboard as tb

        try:
            self.program = tb.program.TensorBoard()
            self.program.configure(
                host="localhost",
                logdir=str(self.path.absolute()),
                path_prefix=f"/services/{self.id}",
                port=0,
            )
            self.server = self.program._make_server()

            self.url = self.server.get_url()
            running.set()
            self.server.serve_forever()
        except Exception:
            logging.exception("Error while starting tensorboard")
            running.set()


@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--keep", is_flag=True, help="Keep temporary directory")
@click.option("--tensorboard", is_flag=True, help="Adds a tensorboard service")
@click.option("--port", type=int, default=12345, help="Port for experimaestro server")
@click.command()
def cli(keep: bool, debug: bool, port: int, tensorboard: bool):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

    maindir = tempfile.TemporaryDirectory()
    if keep:
        maindir._finalizer.detach()

    with maindir:
        maindir = Path(maindir.name)
        xpdir = maindir / "xpm"
        logging.info("Running in %s", xpdir)
        with experiment(
            Path(xpdir) / "", "web-interface", port=port, token="1283jdasfjadse8134"
        ) as xp:
            if tensorboard:
                tb = xp.add_service(TensorboardService(xp.workdir / "runs"))

            socketpath = maindir / "socket1"
            logging.info("Controlled task with %s", socketpath)
            task = (
                ControlledTask(path=socketpath, tensorboard=tensorboard)
                .tag("model", "mine")
                .submit()
            )
            if tensorboard:
                tb.add(task, task.logdir)

            while not socketpath.exists():
                time.sleep(0.1)

            print("\n\n --- Ready to take commands ---")  # noqa: T201
            with (socketpath).open("w") as fp:
                while True:
                    command = input("Command: ")
                    fp.write(command)
                    fp.write("\n")
                    fp.flush()


if __name__ == "__main__":
    cli()
