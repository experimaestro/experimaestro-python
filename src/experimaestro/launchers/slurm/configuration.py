import codecs
from collections import defaultdict
from copy import deepcopy
import io
from attr import Factory
from attrs import define
import sys
import logging
from experimaestro import Annotated
from typing import (
    Dict,
    List,
    Optional,
    Set,
    TextIO,
)
import re
import humanfriendly
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
from experimaestro.compat import cached_property
from . import Launcher
from experimaestro.connectors import (
    Redirect,
)

logger = logging.getLogger("xpm.slurm")


def fill_nodes_configuration(input: TextIO, configuration: "SlurmConfiguration"):
    """Parses the output of scontrol show nodes"""
    re_nodename = re.compile(r"""^NodeName=([_\-\w]+)""")
    re_features = re.compile(r"""^\s*AvailableFeatures=([,_\-\w]+)""")
    re_partitions = re.compile(r"""^\s*Partitions=([,\-_\w]+)""")

    nodename = ""
    features = []
    partition_names = []
    partitions = configuration.partitions
    partitions2features2nodes = defaultdict(lambda: {})

    def process():
        for partition_name in partition_names:
            partition = partitions.setdefault(partition_name, SlurmPartition(nodes=[]))

            fl = "&".join(sorted(features))
            nodes = partitions2features2nodes[partition_name].get(fl)
            if nodes is None:
                nodes = SlurmNodes(hosts=[nodename], features=features)
                partitions2features2nodes[partition_name][fl] = nodes
                partition.nodes.append(nodes)
            else:
                if nodename not in nodes.hosts:
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


def fill_partitions_configuration(input: TextIO, configuration: "SlurmConfiguration"):
    """Parses the output of scontrol show --oneliner partition"""
    re_partitionname = re.compile(r"""^PartitionName=(\w+)""")
    re_mem_per_cpu = re.compile(r"""(?:=|\s)DefMemPerCPU=(\d+)(?:\D|$)""")
    re_cpu_per_gpu = re.compile(r"""(?:=|\s)DefCpuPerGPU=(\d+)(?:\D|$)""")

    for line in input.readlines():
        if match := re_partitionname.search(line):
            name = match.group(1)
            cfg = configuration.partitions.setdefault(name, SlurmPartition(nodes=[]))

            if m := re_mem_per_cpu.search(line):
                cfg.mem_per_cpu = int(m.group(1)) * 1024

            if m := re_cpu_per_gpu.search(line):
                cfg.cpu_per_gpu = int(m.group(1))


# ---- SLURM launcher finder


def parse_size(s: Optional[str]):
    return humanfriendly.parse_size(s) if s else None


@define
class SlurmHostSpecification(HostSpecification):
    features: List[str] = Factory(list)
    hosts: List[str] = Factory(list)
    partition: str = Factory(str)

    qos_id: Optional[str] = Factory(lambda: None)
    """Quality of Service"""

    account_id: Optional[str] = Factory(lambda: None)
    """Account for this host"""


@dataclass
class SlurmNodes(YAMLDataClass):
    features: List[str] = field(default_factory=lambda: [])
    """Nodes features"""

    hosts: List[str] = field(default_factory=lambda: [])

    count: int = 0
    """Number of hosts (if list of hosts is empty)"""


@dataclass
class GPUConfig(YAMLDataClass):
    """Represents a GPU"""

    model: Optional[str] = None
    count: Optional[int] = None
    memory: Annotated[Optional[int], Initialize(parse_size)] = None
    min_memory: Annotated[Optional[int], Initialize(parse_size)] = None


@dataclass
class SlurmPartitionConfiguration(YAMLDataClass):
    cpu_per_gpu: int = 0
    """Number of CPU per GPU"""

    mem_per_cpu: Annotated[int, Initialize(humanfriendly.parse_size)] = 0
    """Memory per CPU"""

    max_duration: Annotated[Optional[int], Initialize(humanfriendly.parse_timespan)] = 0
    """Maximum duration of a job"""

    gpus: Optional[GPUConfig] = None
    """Default GPU settings"""


@dataclass
class SlurmPartition(YAMLDataClass):
    """A Slurm partition"""

    accounts: List[str] = field(default_factory=lambda: [])
    """List of accounts for this partition with the associated priority modifier"""

    qos: List[str] = field(default_factory=lambda: [])
    """List of QoS for this partition with the associated priority modifier"""

    nodes: List[SlurmNodes] = field(default_factory=lambda: [])
    """List of nodes"""

    configuration: Optional[SlurmPartitionConfiguration] = None
    """Partition configuration"""

    priority: int = 0
    """Priority for choosing this partition (higher preferred)"""

    disabled: bool = False
    """Can be used to disable a partition"""


class FeatureConjunction(List[str]):
    def __init__(self, features: List[str]):
        super().__init__(sorted(features))

    def __hash__(self) -> int:
        return sum([hash(tag) for tag in self])


@dataclass
class SlurmConfigGPUOptions(YAMLDataClass):
    min_mem_ratio: float = 0.0
    """Minimum amount of memory that we need to ask"""


@dataclass
class SlurmConfigOptions(YAMLDataClass):
    gpu: SlurmConfigGPUOptions = field(default_factory=lambda: SlurmConfigGPUOptions())


@dataclass
class SlurmFeature(YAMLDataClass):
    """Associate a configuration with a Slurm feature"""

    gpu: Optional[GPUConfig] = None


@dataclass
class SlurmQOS(YAMLDataClass):
    max_duration: Annotated[int, Initialize(humanfriendly.parse_timespan)] = 0
    """Maximum duration of a job"""

    min_gpu: int = 0
    """Minimum number of GPUs"""

    priority: int = 0
    """Priority modifier for this QoS"""


class NodesSpecComputer:
    def __init__(self, config: "SlurmConfiguration", partition: SlurmPartition):
        self.config = config
        self.partition = partition
        self.cpu = CPUSpecification(sys.maxsize, sys.maxsize)
        self.cuda = CudaSpecification(0)
        self.cuda_count = [1]
        self.max_duration = (
            self.partition.configuration.max_duration
            if self.partition.configuration
            else 0
        )
        self.priority = partition.priority
        self.qos_id = None
        self.min_gpu = 0

    def update_gpu(self, gpu: GPUConfig):
        if gpu:
            if gpu.count:
                self.cuda_count = gpu.count

            if gpu.memory:
                self.cuda.memory = gpu.memory
                self.cuda.min_memory = int(
                    self.cuda.memory * self.config.options.gpu.min_mem_ratio
                )

            if gpu.min_memory:
                self.cuda.min_memory = gpu.min_memory

    def update_with_qos(self, qos_id: str):
        self.qos_id = qos_id
        if qos := self.config.qos.get(qos_id, None):
            self.priority += qos.priority
            self.min_gpu = qos.min_gpu
            if qos.max_duration > 0:
                self.max_duration = qos.max_duration

    def get_host(self) -> SlurmHostSpecification:
        cuda = []
        if self.cuda.memory > 0 and self.cuda_count > 0:
            cuda.extend([self.cuda for _ in range(self.cuda_count)])

        host = SlurmHostSpecification(cpu=self.cpu, cuda=cuda)
        host.priority = self.priority
        host.max_duration = self.max_duration
        host.qos_id = self.qos_id
        host.min_gpu = self.min_gpu
        return host


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


class MatchingSpec:
    def __init__(self):
        self.fbf = FeatureBooleanFormula()
        self.hosts: set[str] = set()
        self.partitions: Set[str] = set()
        self.qos: Optional[str] = None
        self.account: Optional[str] = None

    def update(self, host_spec: SlurmHostSpecification):
        if host_spec.qos_id != self.qos and self.qos is not None:
            # Cannot update with other QoS
            return
        self.qos = host_spec.qos_id

        if host_spec.account_id != self.account and self.account is not None:
            # Cannot update with other account
            return
        self.account = host_spec.account_id

        self.partitions.add(host_spec.partition)
        self.fbf.add(host_spec.features)
        if host_spec.hosts:
            self.hosts.update(host_spec.hosts)


@dataclass
class SlurmConfiguration(YAMLDataClass, LauncherConfiguration):
    id: str
    """Slurm ID"""

    partitions: Dict[str, SlurmPartition]
    """List of partitions"""

    connector: str = "local"
    """Name of the connector"""

    use_features: bool = True
    """Whether features should be used"""

    use_hosts: bool = True
    """Whether hosts should be used in the query"""

    options: SlurmConfigOptions = field(default_factory=lambda: SlurmConfigOptions())
    """SLURM options"""

    query_slurm: bool = False
    """True to query SLURM directly (using scontrol)"""

    tags: List[str] = field(default_factory=lambda: [])

    weight: int = 0

    qos: Dict[str, SlurmQOS] = field(default_factory=lambda: {})

    features_regex: Annotated[
        List[re.Pattern],
        Initialize(lambda regexps: [re.compile(regex) for regex in regexps]),
    ] = field(default_factory=lambda: [])
    """
    Regex to get the information from features
        - CUDA: cuda:count, cuda:memory
    """

    features: Dict[str, SlurmFeature] = field(default_factory=lambda: {})
    """List of features with associated configurations"""

    def compute(self, registry: "LauncherRegistry"):
        if self.query_slurm:
            self.query_slurm = False

            # Read node information
            connector = registry.getConnector(self.connector)
            pb = connector.processbuilder()
            pb.command = ["scontrol", "--hide", "show", "nodes"]

            def handle_output(input: io.BytesIO):
                StreamReader = codecs.getreader("utf-8")
                fill_nodes_configuration(StreamReader(input), self)

            pb.stdout = Redirect.pipe(handle_output)
            pb.start()

            # Read partition information
            pb = connector.processbuilder()
            pb.command = ["scontrol", "--hide", "show", "--oneliner", "partition"]

            def handle_output(input: io.BytesIO):
                StreamReader = codecs.getreader("utf-8")
                fill_partitions_configuration(StreamReader(input), self)

            pb.stdout = Redirect.pipe(handle_output)
            pb.start()

    @cached_property
    def computed_nodes(self) -> List[SlurmHostSpecification]:
        """Computes the list of potential compute nodes (grouped by similar nodes)"""
        hosts = []

        for partition_name, partition in self.partitions.items():
            if partition.disabled:
                continue

            for node in partition.nodes:
                nodes_spec = NodesSpecComputer(self, partition)

                # Set partition GPU
                if partition.configuration:
                    nodes_spec.update_gpu(partition.configuration.gpus)

                for feature in node.features:
                    # Use feature data directly
                    if data := self.features.get(feature, None):
                        nodes_spec.update_gpu(data.gpu)

                    # logger.debug("Looking at %s", self.features_regex)
                    for regex in self.features_regex:
                        # logger.debug("%s/%s => %s", regex, tag, regex.match(tag))
                        if m := regex.match(feature):
                            d = m.groupdict()
                            if _count := d.get("cuda_count", None):
                                nodes_spec.cuda_count = int(_count)
                            if memory := d.get("cuda_memory", None):
                                nodes_spec.cuda.memory = humanfriendly.parse_size(
                                    memory
                                )
                                nodes_spec.cuda.min_memory = int(
                                    nodes_spec.cuda.memory
                                    * self.options.gpu.min_mem_ratio
                                )

                qos_list = partition.qos or [None]
                accounts = partition.accounts or [None]
                for qos in qos_list:
                    qos_nodes_spec = deepcopy(nodes_spec)
                    qos_nodes_spec.update_with_qos(qos)

                    host = qos_nodes_spec.get_host()
                    host.features = node.features
                    host.partition = partition_name
                    host.hosts = node.hosts

                    for account in accounts:
                        account_host = deepcopy(host)
                        account_host.account_id = account
                        hosts.append(account_host)
                    logging.debug("Computed slurm host: %s", host)

        hosts.sort(key=lambda host: -host.priority)
        return hosts

    def get(
        self, registry: "LauncherRegistry", requirement: HostRequirement
    ) -> Optional["Launcher"]:
        # Compute the configuration if needed
        self.compute(registry)

        # Compute tags or hosts

        # Current set of constraints
        current_match = None
        matching_spec = MatchingSpec()

        for node in self.computed_nodes:
            if match := requirement.match(node):
                logger.debug("Match %s for %s", match, node)

                # If score is below the current one, goes to the next one
                if current_match and (
                    match.score <= current_match.score
                    and match.requirement is not current_match.requirement
                ):
                    continue

                # If the requirement has changed, clear everything
                if not current_match or (
                    match.requirement is not current_match.requirement
                ):
                    # Clear if the requirement changed
                    logger.debug("Clearing %s / %s", current_match, match)
                    matching_spec = MatchingSpec()
                    current_match = match

                logger.debug(
                    "Adding %s, %s, %s", node.partition, node.features, node.hosts
                )
                matching_spec.update(node)

        # Returns the appropriate launcher (if any)
        use_features = matching_spec.fbf.clauses and self.use_features
        if use_features or matching_spec.hosts:
            assert current_match is not None

            # Launching using tags
            from .base import SlurmLauncher

            launcher = SlurmLauncher(connector=registry.getConnector(self.connector))

            launcher.options.partition = ",".join(matching_spec.partitions)
            launcher.options.gpus_per_node = (
                len(current_match.requirement.cuda_gpus)
                if current_match.requirement.cuda_gpus
                else 0
            )

            launcher.options.qos = matching_spec.qos
            launcher.options.account = matching_spec.account

            if current_match.requirement.cpu.memory > 0:
                launcher.options.mem = (
                    f"{current_match.requirement.cpu.memory // 1024}M"
                )

            if current_match.requirement.cpu.cores > 0:
                launcher.options.cpus_per_task = current_match.requirement.cpu.cores

            if use_features:
                launcher.options.constraint = matching_spec.fbf.to_constraint()
            else:
                logger.warning("Selecting first host")
                launcher.options.nodelist = next(iter(matching_spec.hosts))

            if current_match.requirement.duration > 0:
                total_seconds = current_match.requirement.duration
                seconds = total_seconds % 60
                minutes = (total_seconds // 60) % 60
                hours = total_seconds // 3600
                launcher.options.time = f"{hours}:{minutes}:{seconds}"

            logger.debug("Slurm options: %s", " ".join(launcher.options.args()))
            return launcher

        return None
