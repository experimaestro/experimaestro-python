import logging
import threading
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    get_type_hints,
)
from experimaestro.connectors.local import LocalConnector
import re
from shlex import quote as shquote
from contextlib import contextmanager
from dataclasses import dataclass
from experimaestro.launcherfinder.registry import (
    LauncherRegistry,
)
from experimaestro.utils import ThreadingCondition
from experimaestro.tests.connectors.utils import OutputCaptureHandler
from experimaestro.utils.asyncio import asyncThreadcheck
from experimaestro.compat import cached_property
from experimaestro.launchers import Launcher
from experimaestro.scriptbuilder import PythonScriptBuilder
from experimaestro.connectors import (
    Connector,
    ProcessBuilder,
    Process,
    ProcessState,
    Redirect,
    RedirectType,
)

logger = logging.getLogger("xpm.slurm")


class SlurmJobState:
    start: str
    end: str
    status: str

    STATE_MAP = {
        "CONFIGURING": ProcessState.SCHEDULED,
        "REQUEUE_FED": ProcessState.SCHEDULED,
        "PENDING": ProcessState.SCHEDULED,
        "REQUEUE_HOLD": ProcessState.SCHEDULED,
        "COMPLETING": ProcessState.RUNNING,
        "RUNNING": ProcessState.RUNNING,
        "COMPLETED": ProcessState.DONE,
        "FAILED": ProcessState.ERROR,
        "DEADLINE": ProcessState.ERROR,
        "NODE_FAIL": ProcessState.ERROR,
        "REVOKED": ProcessState.ERROR,
        "TIMEOUT": ProcessState.ERROR,
        "CANCELLED": ProcessState.ERROR,
        "BOOT_FAIL": ProcessState.ERROR,
        "OUT_OF_MEMORY": ProcessState.ERROR,
    }

    def __init__(self, status, start, end):
        self.slurm_state = status if status[-1] == "+" else status
        if self.slurm_state.startswith("CANCELLED"):
            self.state = ProcessState.ERROR
        else:
            self.state = SlurmJobState.STATE_MAP.get(self.slurm_state, None)
            if self.state is None:
                logging.warning(
                    "Unknown state: %s (supposing this is an error)", self.slurm_state
                )
                self.state = ProcessState.ERROR

        self.start = start
        self.end = end

    def finished(self):
        """Returns true if the job has finished"""
        return self.state.finished

    def __repr__(self):
        return f"{self.slurm_state} ({self.start}-{self.end})"


class SlurmProcessWatcher(threading.Thread):
    """Process that calls sacct at regular interval to check job status"""

    WATCHERS: Dict[Tuple[Tuple[str, Any]], "SlurmProcessWatcher"] = {}

    def __init__(self, launcher: "SlurmLauncher"):
        super().__init__()
        self.launcher = launcher
        self.count = 1
        self.jobs: Dict[str, SlurmJobState] = {}

        self.cv = ThreadingCondition()
        self.fetched_event = threading.Event()
        self.updating_jobs = threading.Lock()
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

    def getjob(self, jobid, timeout=None):
        """Allows to share the calls to sacct"""

        # Ensures that we have fetched at least once
        self.fetched_event.wait()

        # Waits that jobs are refreshed (with a timeout)
        with self.cv:
            self.cv.wait(timeout=timeout)

        # Ensures jobs are not updated right now
        with self.updating_jobs:
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
            process = builder.start()

            with self.updating_jobs:
                self.jobs = {}
                output = handler.output.decode("utf-8")
                for line in output.split("\n"):
                    line = line.strip()
                    if line:
                        try:
                            jobid, state, start, end, *_ = line.split("|")
                            self.jobs[jobid] = SlurmJobState(state, start, end)
                            logger.debug("Parsed line: %s", line)
                        except ValueError:
                            logger.error("Could not parse line %s", line)
            process.kill()

            with self.cv:
                logger.debug("Jobs %s", self.jobs)
                self.fetched_event.set()
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
                    return 0 if state.slurm_state == "COMPLETED" else 1

    async def aio_state(self):
        def check():
            with SlurmProcessWatcher.get(self.launcher) as watcher:
                jobinfo = watcher.getjob(self.jobid)
                return jobinfo.state if jobinfo else ProcessState.SCHEDULED

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
        process = BatchSlurmProcess(launcher, spec["pid"])

        # Checks that the process is running
        with SlurmProcessWatcher.get(launcher) as watcher:
            logger.info("Checking SLURM job %s", process.jobid)
            jobinfo = watcher.getjob(process.jobid, timeout=0.1)
            if jobinfo and jobinfo.state.running:
                logger.debug(
                    "SLURM job is running (%s), returning process", process.jobid
                )
                return process
        return None


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

    def start(self, task_mode: bool = False) -> BatchSlurmProcess:
        """Start the process"""
        builder = self.launcher.connector.processbuilder()
        builder.environ = self.launcher.launcherenv
        builder.detach = False

        if not self.detach:
            # Simplest case: we wait for the output
            builder.workingDirectory = self.workingDirectory
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

        if not task_mode:
            # Use command line parameters when not running a task
            builder.command.extend(self.launcher.options.args())

            if self.workingDirectory:
                workdir = self.launcher.connector.resolve(self.workingDirectory)
                builder.command.append(f"--chdir={workdir}")
            addstream(builder.command, "-e", self.stderr)
            addstream(builder.command, "-o", self.stdout)
            addstream(builder.command, "-i", self.stdin)

        builder.command.extend(self.command)
        logger.info(
            "slurm sbatch command: %s", " ".join(f'"{s}"' for s in builder.command)
        )
        handler = OutputCaptureHandler()
        builder.stdout = Redirect.pipe(handler)
        builder.stderr = Redirect.inherit()
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


@dataclass
class SlurmOptions:
    # Options
    nodes: Optional[int] = 1
    """Number of requested nodes"""

    time: Optional[str] = None
    """Requested time"""

    account: Optional[str] = None
    """The account for launching the job"""

    qos: Optional[str] = None
    """The requested Quality of Service"""

    partition: Optional[str] = None
    """The requested partition"""

    constraint: Optional[str] = None
    """Logic expression on node features (as defined by the administator)"""

    mem: Optional[str] = None
    """Requested memory on the node (in megabytes by default)"""

    exclude: Optional[str] = None
    """List of hosts to exclude"""

    mem_per_gpu: Optional[str] = None
    """Requested memory per allocated GPU (size with units: K, M, G, or T)"""

    cpus_per_task: Optional[str] = None
    """Number of cpus requested per task"""

    nodelist: Optional[str] = None
    """Request a specific list of hosts"""

    # GPU-related
    gpus: Optional[int] = None
    """Number of GPUs"""

    gpus_per_node: Optional[int] = None
    """Number of GPUs per node"""

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

    @staticmethod
    def format_time(duration_s: int):
        """Format time for the SLURM option

        :param duration_s: Time duration in seconds1
        :return: The configuration string
        """
        seconds = duration_s % 60
        minutes = (duration_s // 60) % 60
        hours = duration_s // 3600
        return f"{hours}:{minutes}:{seconds}"


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

    def __str__(self):
        return f"SlurmLauncher({self.options}, path={self.binpath})"

    @staticmethod
    def init_registry(registry: LauncherRegistry):
        from .configuration import SlurmConfiguration

        registry.register_launcher("slurm", SlurmConfiguration)

    @staticmethod
    def get_cli():
        from .cli import cli

        return cli

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
        return SlurmScriptBuilder(self)

    def processbuilder(self) -> SlurmProcessBuilder:
        """Returns the process builder for this launcher

        By default, returns the associated connector builder"""
        return SlurmProcessBuilder(self)


class SlurmScriptBuilder(PythonScriptBuilder):
    def __init__(self, launcher: SlurmLauncher, pythonpath=None):
        super().__init__(pythonpath)
        self.launcher = launcher
        self.processtype = "slurm"

    def write(self, job):
        py_path = super().write(job)
        main_path = py_path.parent

        def relpath(path: Path):
            return shquote(self.launcher.connector.resolve(path, main_path))

        # Writes the sbatch shell script containing all the options
        sh_path = job.jobpath / ("%s.sh" % job.name)
        with sh_path.open("wt") as out:
            out.write("""#!/bin/sh\n\n""")

            workdir = self.launcher.connector.resolve(main_path)
            out.write(f"#SBATCH --chdir={shquote(workdir)}\n")
            out.write(f"""#SBATCH --error={relpath(job.stderr)}\n""")
            out.write(f"""#SBATCH --output={relpath(job.stdout)}\n""")

            for arg in self.launcher.options.args():
                out.write(f"""#SBATCH {arg}\n""")

            # We finish by the call to srun
            out.write(f"""\nsrun ./{relpath(py_path)}\n\n""")

        self.launcher.connector.setExecutable(sh_path, True)
        return sh_path
