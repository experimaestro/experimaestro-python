import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, get_type_hints
from experimaestro.connectors.local import LocalConnector
import re
import logging
from contextlib import contextmanager
from experimaestro.utils import ThreadingCondition
from experimaestro.tests.connectors.utils import OutputCaptureHandler
from experimaestro.utils.asyncio import asyncThreadcheck
from experimaestro.compat import cached_property
from . import Launcher
from experimaestro.scriptbuilder import ShScriptBuilder
from experimaestro.connectors import (
    Connector,
    ProcessBuilder,
    Process,
    Redirect,
    RedirectType,
)

logger = logging.getLogger("xpm.slurm")


class SlurmJobState:
    start: str
    end: str
    status: str

    def __init__(self, status, start, end):
        self.status = status
        self.start = start
        self.end = end

    def finished(self):
        """Returns true if the job has finished"""
        return (
            self.status
            in [
                "COMPLETED",
                "FAILED",
                "DEADLINE",
                "NODE_FAIL",
                "REVOKED",
                "TIMEOUT",
                "BOOT_FAIL",
            ]
            or self.status.startswith("CANCELLED")
        )

    def __repr__(self):
        return f"{self.status} ({self.start}-{self.end})"


class SlurmProcessWatcher(threading.Thread):
    WATCHERS: Dict[Tuple[Tuple[str, Any]], "SlurmProcessWatcher"] = {}

    def __init__(self, launcher: "SlurmLauncher"):
        super().__init__()
        self.launcher = launcher
        self.count = 1
        self.jobs: Dict[str, SlurmJobState] = {}

        self.cv = ThreadingCondition()
        self.start()

    @staticmethod
    @contextmanager
    def get(launcher: "SlurmLauncher"):
        watcher = SlurmProcessWatcher.WATCHERS.get(launcher.key, None)
        if watcher is None:
            watcher = SlurmProcessWatcher(launcher)
            SlurmProcessWatcher.WATCHERS[launcher.key] = watcher
        else:
            with watcher.cv:
                watcher.count += 1
        yield watcher
        watcher.count -= 1
        with watcher.cv:
            watcher.cv.notify()

    def getjob(self, jobid):
        """Allows to share the calls to sacct"""
        with self.cv:
            self.cv.wait()
            return self.jobs.get(jobid)

    def run(self):
        while self.count > 0:
            builder = self.launcher.connector.processbuilder()
            builder.command = [
                f"{self.launcher.binpath}/sacct",
                "-n",
                "-p",
                "--format=JobID,State,Start,End",
            ]
            handler = OutputCaptureHandler()
            builder.detach = False
            builder.stdout = Redirect.pipe(handler)
            builder.environ = self.launcher.launcherenv
            logger.debug("Checking SLURM state with sacct")
            builder.start()

            with self.cv:
                self.jobs = {}
                output = handler.output.decode("utf-8")
                for line in output.split("\n"):
                    line = line.strip()
                    if line:
                        try:
                            jobid, state, start, end, *_ = line.split("|")
                            self.jobs[jobid] = SlurmJobState(state, start, end)
                            logger.debug("Parsed line: %s", line)
                        except ValueError as e:
                            logger.error("Could not parse line %s", line)
                logger.debug("Jobs %s", self.jobs)
                self.cv.notify_all()

                self.cv.wait_for(
                    lambda: self.count == 0, timeout=self.launcher.interval
                )
                if self.count == 0:
                    logger.debug("Stopping SLURM watcher process")
                    del SlurmProcessWatcher.WATCHERS[self.launcher.key]
                    break


class BatchSlurmProcess(Process):
    """A batch slurm process"""

    def __init__(self, launcher: "SlurmLauncher", jobid: str):
        self.launcher = launcher
        self.jobid = jobid

    def wait(self):
        with SlurmProcessWatcher.get(self.launcher) as watcher:
            while True:
                state = watcher.getjob(self.jobid)
                if state and state.finished():
                    return 0 if state.status == "COMPLETED" else 1

    async def aio_isrunning(self):
        def check():
            with SlurmProcessWatcher.get(self.launcher) as watcher:
                jobinfo = watcher.getjob(self.jobid)
                return jobinfo is not None and not jobinfo.finished()

        return await asyncThreadcheck("slurm.aio_isrunning", check)

    def kill(self):
        logger.warning("Killing slurm job %s", self.jobid)
        builder = self.launcher.connector.processbuilder()
        builder.command = [f"{self.launcher.binpath}/scancel", f"{self.jobid}"]
        builder.start()

    def __repr__(self):
        return f"slurm:{self.jobid}"

    def tospec(self):
        return {"type": "slurm", "pid": self.jobid, "options": self.launcher.key}

    @classmethod
    def fromspec(cls, connector: Connector, spec: Dict[str, Any]):
        options = {k: v for k, v in spec.get("options", ())}
        launcher = SlurmLauncher(connector=connector, **options)
        return BatchSlurmProcess(launcher, spec["pid"])


def addstream(command: List[str], option: str, redirect: Redirect):
    if redirect.type == RedirectType.FILE:
        command.extend([option, redirect.path])
    elif redirect.type == RedirectType.INHERIT:
        pass
    else:
        raise NotImplementedError("For %s", redirect)


class SlurmProcessBuilder(ProcessBuilder):
    def __init__(self, launcher: "SlurmLauncher"):
        super().__init__()
        self.launcher = launcher

    def start(self) -> BatchSlurmProcess:
        """Start the process"""
        builder = self.launcher.connector.processbuilder()
        builder.workingDirectory = self.workingDirectory
        builder.environ = self.launcher.launcherenv
        builder.detach = False

        if not self.detach:
            # Simplest case: we wait for the output
            builder.command = [f"{self.launcher.binpath}/srun"]
            builder.command.extend(self.launcher.options.args())
            builder.command.extend(self.command)
            builder.stdin = self.stdin
            builder.stdout = self.stdout
            builder.stderr = self.stderr
            builder.environ = self.environ
            builder.detach = False
            return builder.start()

        builder.command = [f"{self.launcher.binpath}/sbatch", "--parsable"]
        builder.command.extend(self.launcher.options.args())

        addstream(builder.command, "-e", self.stderr)
        addstream(builder.command, "-o", self.stdout)
        addstream(builder.command, "-i", self.stdin)

        builder.command.extend(self.command)
        logger.info("slurm sbatch command: %s", builder.command)
        handler = OutputCaptureHandler()
        builder.stdout = Redirect.pipe(handler)
        p = builder.start()
        if p.wait() != 0:
            logger.error("Error while running sbatch command")
            raise RuntimeError("Error while submitting job")

        output = handler.output.decode("utf-8").strip(" \n")
        RE_SUBMITTED_JOB = re.compile(r"""^(\d+)(?:;.*)?$""", re.MULTILINE)
        m = RE_SUBMITTED_JOB.match(output)
        if m is None:
            raise RuntimeError(f"Could not get the submitted job ID from {output}")

        return BatchSlurmProcess(self.launcher, m.group(1))


class SlurmOptions:
    # Options
    nodes: Optional[int] = 1
    time: Optional[str] = None

    account: Optional[str] = None
    """The account for launching the job"""

    qos: Optional[str] = None
    """The requested Quality of Service"""

    mem: Optional[str] = None
    """Requested memory on the node (in megabytes by default)"""

    # GPU-related
    gpus: Optional[int] = None
    gpus_per_node: Optional[int] = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key in SlurmOptions.__dict__:
                self.__dict__[key] = value
            else:
                raise ValueError("%s is not a valid option for Slurm")

    def args(self) -> List[str]:
        """Returns the corresponding options"""
        options = []
        for key in get_type_hints(SlurmOptions).keys():
            value = getattr(self, key, None)
            if value is not None:
                options.append(
                    f"""--{key.replace("_", "-")}={value}""",
                )

        return options

    def merge(self, other: "SlurmOptions"):
        merged = SlurmOptions()

        for key, value in self.__dict__.items():
            setattr(merged, key, value)

        for key, value in other.__dict__.items():
            if value is not None:
                setattr(merged, key, value)

        return merged


class SlurmLauncher(Launcher):
    """Slurm workload manager launcher

    https://slurm.schedmd.com/documentation.html
    """

    def __init__(
        self,
        *,
        connector: Connector = None,
        options: SlurmOptions = None,
        interval: float = 60,
        main=None,
        launcherenv: Dict[str, str] = None,
        binpath="/usr/bin",
    ):
        """
        Arguments:
            main: Main slurm launcher to avoid launching too many polling jobs
            interval: seconds between polling job statuses
        """
        super().__init__(connector or LocalConnector.instance())
        self.binpath = Path(binpath)
        self.interval = interval
        self.launcherenv = launcherenv
        self.options = options or SlurmOptions()

    @cached_property
    def key(self):
        """Returns a dictionary characterizing this launcher when calling sacct/etc"""
        return (
            ("binpath", str(self.binpath)),
            ("interval", self.interval),
        )

    def config(self, **kwargs):
        """Returns a new Slurm launcher with the given configuration"""
        return SlurmLauncher(
            connector=self.connector,
            binpath=self.binpath,
            launcherenv=self.launcherenv,
            options=self.options.merge(SlurmOptions(**kwargs)),
            interval=self.interval,
        )

    def scriptbuilder(self):
        """Returns the script builder

        We assume *nix, but should be changed to PythonScriptBuilder when working
        """
        builder = ShScriptBuilder()
        builder.processtype = "slurm"
        return builder

    def processbuilder(self) -> SlurmProcessBuilder:
        """Returns the process builder for this launcher

        By default, returns the associated connector builder"""
        return SlurmProcessBuilder(self)
