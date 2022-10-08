import codecs
from collections import defaultdict
import io
import sys
import click
import logging
import threading
from pathlib import Path
from experimaestro import Annotated
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    TextIO,
    Tuple,
    get_type_hints,
)
from experimaestro.connectors.local import LocalConnector
import re
import humanfriendly
from contextlib import contextmanager
from dataclasses import dataclass, field
from experimaestro.launcherfinder import YAMLDataClass, HostRequirement
from experimaestro.launcherfinder.base import LauncherConfiguration
from experimaestro.launcherfinder.registry import (
    Initialize,
    LauncherRegistry,
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


@dataclass
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

    @staticmethod
    def get_cli():
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
        builder = PythonScriptBuilder()
        builder.processtype = "slurm"
        return builder

    def processbuilder(self) -> SlurmProcessBuilder:
        """Returns the process builder for this launcher

        By default, returns the associated connector builder"""
        return SlurmProcessBuilder(self)


# --- Command line


@click.group()
def cli():
    pass


@cli.command()
def convert():
    """Convert the ouptut of 'scontrol show node' into a YAML form compatible with launchers.yaml"""
    import yaml
    from experimaestro.launcherfinder import LauncherRegistry

    configuration = SlurmConfiguration(id="", partitions={})
    fill_configuration(sys.stdin, configuration)
    yaml.dump(configuration, sys.stdout, Dumper=LauncherRegistry.instance().Dumper)


def fill_configuration(input: TextIO, configuration: "SlurmConfiguration"):
    re_nodename = re.compile(r"""^NodeName=(\w+)""")
    re_features = re.compile(r"""^\s*AvailableFeatures=([,\w]+)""")
    re_partitions = re.compile(r"""^\s*Partitions=([,\w]+)""")

    nodename = ""
    features = []
    partition_names = []
    partitions = configuration.partitions
    partitions2features2nodes = defaultdict(lambda: {})

    def process():
        for partition_name in partition_names:
            partition = partitions.get(partition_name, None)
            if partition is None:
                partition = SlurmPartition(nodes=[])
                partitions[partition_name] = partition

            fl = "&".join(sorted(features))
            nodes = partitions2features2nodes[partition_name].get(fl)
            if nodes is None:
                nodes = SlurmNodes(hosts=[nodename], features=features)
                partitions2features2nodes[partition_name][fl] = nodes
                partition.nodes.append(nodes)
            else:
                nodes.hosts.append(nodename)

    for line in input.readlines():
        if match := re_nodename.search(line):
            if nodename:
                process()
            nodename = match.group(1)
        elif match := re_features.search(line):
            features = match.group(1).split(",")
        elif match := re_partitions.search(line):
            partition_names = match.group(1).split(",")

    if nodename:
        process()


# ---- SLURM launcher finder


class SlurmNodesSpecification(HostSpecification):
    features: List[str]
    hosts: List[str]
    partition: str


@dataclass
class SlurmNodes(YAMLDataClass):
    features: List[str] = field(default_factory=lambda: [])
    hosts: List[str] = field(default_factory=lambda: [])
    count: int = 0


@dataclass
class SlurmPartition(YAMLDataClass):
    nodes: List[SlurmNodes]


class FeatureConjunction(List[str]):
    def __init__(self, features: List[str]):
        super().__init__(sorted(features))

    def __hash__(self) -> int:
        return sum([hash(tag) for tag in self])


class FeatureBooleanFormula:
    clauses: Set[List[str]]

    def __init__(self):
        self.clauses = set()

    def add(self, features: List[str]):
        """Adds conjunction of tags"""
        self.clauses.add(FeatureConjunction(features))

    def to_constraint(self):
        """Returns a constraint for sbatch/srun"""
        it = ("&".join(clause) for clause in self.clauses)
        return f"""({")|(".join(it)})"""


@dataclass
class SlurmConfiguration(YAMLDataClass, LauncherConfiguration):
    id: str
    """Slurm ID"""

    partitions: Dict[str, SlurmPartition]
    connector: str = "local"
    use_features: bool = True
    use_hosts: bool = True

    query_slurm: bool = False
    """True to query SLURM directly"""

    tags: List[str] = field(default_factory=lambda: [])
    weight: int = 0

    features_regex: Annotated[
        List[re.Pattern],
        Initialize(lambda regexps: [re.compile(regex) for regex in regexps]),
    ] = field(default_factory=lambda: [])
    """
    Regex to get the information from tags
        - CUDA: cuda:count, cuda:memory
    """

    def compute(self, registry: "LauncherRegistry"):
        if self.query_slurm:
            self.query_slurm = False

            connector = registry.getConnector(self.connector)
            pb = connector.processbuilder()
            pb.command = ["scontrol", "--hide", "show", "nodes"]

            def handle_output(input: io.BytesIO):
                StreamReader = codecs.getreader("utf-8")
                fill_configuration(StreamReader(input), self)

            pb.stdout = Redirect.pipe(handle_output)
            pb.start()

    @cached_property
    def computed_nodes(self) -> List[SlurmNodesSpecification]:
        hosts = []

        for partition_name, partition in self.partitions.items():
            for node in partition.nodes:
                cpu = CPUSpecification(1, 1)
                cuda = [CudaSpecification(0)]

                for feature in node.features:
                    count = 1
                    # logger.debug("Looking at %s", self.features_regex)
                    for regex in self.features_regex:
                        # logger.debug("%s/%s => %s", regex, tag, regex.match(tag))
                        if m := regex.match(feature):
                            d = m.groupdict()
                            if _count := d.get("cuda_count", None):
                                count = int(_count)
                            if memory := d.get("cuda_memory", None):
                                cuda[0].memory = humanfriendly.parse_size(memory)

                    if count > 1:
                        cuda.extend([cuda[0] for _ in range(count - 1)])

                host = SlurmNodesSpecification(cpu, cuda)
                host.features = node.features
                host.partition = partition_name
                host.hosts = node.hosts
                hosts.append(host)
        return hosts

    def get(
        self, registry: "LauncherRegistry", requirement: HostRequirement
    ) -> Optional["Launcher"]:

        # Compute the configuration if needed
        self.compute(registry)

        # Compute tags or hosts
        fbf = FeatureBooleanFormula()
        hosts: set[str] = set()
        partitions: Set[str] = set()

        # TODO:
        # - take into account maximum node duration
        current_match = None
        for node in self.computed_nodes:
            if match := requirement.match(node):
                logger.debug("Match %s for %s", match, node)
                if current_match and (
                    match.score <= current_match.score
                    and match.requirement is not current_match.requirement
                ):
                    continue

                if not current_match or (
                    match.requirement is not current_match.requirement
                ):
                    # Clear if the requirement changed
                    logger.debug("Clearing %s / %s", current_match, match)
                    partitions.clear()
                    fbf = FeatureBooleanFormula()
                    hosts.clear()
                    current_match = match

                logger.debug(
                    "Adding %s, %s, %s", node.partition, node.features, node.hosts
                )
                partitions.add(node.partition)
                fbf.add(node.features)
                if node.hosts:
                    hosts.update(node.hosts)

        # Returns the appropriate launcher (if any)
        use_features = fbf.clauses and self.use_features
        if use_features or hosts:
            assert current_match is not None

            # Launching using tags
            launcher = SlurmLauncher(connector=registry.getConnector(self.connector))

            launcher.options.partition = ",".join(partitions)
            launcher.options.gpus_per_node = (
                len(current_match.requirement.cuda_gpus)
                if current_match.requirement.cuda_gpus
                else 0
            )

            if use_features:
                launcher.options.constraint = fbf.to_constraint()
            else:
                logger.warning("Selecting first host")
                launcher.options.nodelist = next(iter(hosts))

            if current_match.requirement.duration > 0:
                total_seconds = current_match.requirement.duration
                seconds = total_seconds % 60
                minutes = (total_seconds // 60) % 60
                hours = total_seconds // 3600
                launcher.options.time = f"{hours}:{minutes}:{seconds}"

            logger.debug("Slurm options: %s", " ".join(launcher.options.args()))
            return launcher

        return None
