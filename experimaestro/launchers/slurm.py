import threading
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    get_type_hints,
)
from experimaestro.connectors.local import LocalConnector
import re
import logging
import humanfriendly
from contextlib import contextmanager
from dataclasses import dataclass, field
from experimaestro.launcherfinder import YAMLDataClass, HostRequirement
from experimaestro.launcherfinder.base import LauncherConfiguration
from experimaestro.launcherfinder.registry import (
    GPU,
    GPUList,
    Initialize,
    LauncherRegistry,
    YAMLList,
)
from experimaestro.launcherfinder.specs import (
    CPUSpecification,
    CudaSpecification,
    HostSpecification,
)
from experimaestro.utils import ThreadingCondition
from experimaestro.tests.connectors.utils import OutputCaptureHandler
from experimaestro.utils.asyncio import asyncThreadcheck
from experimaestro.compat import cached_property
from . import Launcher
from experimaestro.scriptbuilder import PythonScriptBuilder
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
        return self.status in [
            "COMPLETED",
            "FAILED",
            "DEADLINE",
            "NODE_FAIL",
            "REVOKED",
            "TIMEOUT",
            "BOOT_FAIL",
        ] or self.status.startswith("CANCELLED")

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

    nodelist: Optional[str] = None
    """Request a specific list of hosts"""

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

    @staticmethod
    def init_registry(registry: LauncherRegistry):
        registry.register_launcher("slurm", SlurmConfiguration)

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
        builder = PythonScriptBuilder()
        builder.processtype = "slurm"
        return builder

    def processbuilder(self) -> SlurmProcessBuilder:
        """Returns the process builder for this launcher

        By default, returns the associated connector builder"""
        return SlurmProcessBuilder(self)


# ---- SLURM launcher finder


class SlurmNodesSpecification(HostSpecification):
    tags: List[str]
    hosts: List[str]
    partition: str


@dataclass
class SlurmNodes(YAMLDataClass):
    tags: List[str] = field(default_factory=lambda: [])
    hosts: List[str] = field(default_factory=lambda: [])
    count: int = 0


@dataclass
class SlurmPartition(YAMLDataClass):
    nodes: List[SlurmNodes]


class TagsConjunctiveNormalForm:
    clauses: List[Set[str]]

    def __init__(self):
        self.clauses = []

    def add(self, tags: List[str]):
        """Adds conjunction of tags"""
        raise NotImplementedError()

    def to_constraint(self):
        """Returns a constraint for sbatch/srun"""
        it = ("|".join(clause) for clause in self.clauses)
        return f"""({")&(".join(it)})"""


@dataclass
class SlurmConfiguration(YAMLDataClass, LauncherConfiguration):
    connector: str
    partitions: Dict[str, SlurmPartition]
    use_features: bool = True
    use_hosts: bool = True

    features_regex: Annotated[
        List[re.Pattern],
        Initialize(lambda regexps: [re.compile(regex) for regex in regexps]),
    ] = field(default_factory=lambda: [])
    """
    Regex to get the information from tags
        - CUDA: cuda:count, cuda:memory
    """

    @cached_property
    def computed_nodes(self) -> List[SlurmNodesSpecification]:
        hosts = []

        for partition_name, partition in self.partitions.items():
            for node in partition.nodes:
                cpu = CPUSpecification(0, 0)
                cuda = [CudaSpecification(0)]

                for tag in node.tags:
                    count = 1
                    # logger.debug("Looking at %s", self.features_regex)
                    for regex in self.features_regex:
                        # logger.debug("%s/%s => %s", regex, tag, regex.match(tag))
                        if m := regex.match(tag):
                            d = m.groupdict()
                            if _count := d.get("cuda_count", None):
                                count = int(_count)
                            if memory := d.get("cuda_memory", None):
                                cuda[0].memory = humanfriendly.parse_size(memory)

                    if count > 1:
                        cuda.extend([cuda[0] for _ in range(count - 1)])

                host = SlurmNodesSpecification(cpu, cuda)
                host.tags = node.tags
                host.partition = partition_name
                host.hosts = node.hosts
                hosts.append(host)
        return hosts

    def get(
        self, registry: "LauncherRegistry", requirement: HostRequirement
    ) -> Optional["Launcher"]:

        # Compute tags or hosts
        cnf = TagsConjunctiveNormalForm()
        hosts: List[str] = []
        partitions: Set[str] = set()

        for node in self.computed_nodes:
            logging.debug("Look if %s is OK with %s", requirement, node)
            if requirement.match(node):
                partitions.add(node.partition)
                cnf.add(node.tags)
                if node.hosts:
                    hosts.extend(node.hosts)

        if cnf.clauses:
            # Launching using tags
            launcher = SlurmLauncher(connector=registry.getConnector(self.connector))
            launcher.options.partition = ",".join(partitions)
            launcher.options.constraint = cnf.to_constraint()
            return launcher

        if hosts:
            # Launching using nodes
            launcher = SlurmLauncher(connector=registry.getConnector(self.connector))
            launcher.options.nodelist = ",".join(hosts)
            launcher.options.partition = ",".join(partitions)
            return launcher

        return None
