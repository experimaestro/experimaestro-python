import codecs
from collections import defaultdict
from copy import deepcopy
import io
import math
from attr import Factory
from attrs import define
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


@dataclass
class GPUConfig(YAMLDataClass):
    """Represents a GPU"""

    model: Optional[str] = None
    count: int = 0
    memory: Annotated[int, Initialize(parse_size)] = 0

    min_memory: Annotated[int, Initialize(parse_size)] = 0
    """Minimum memory to be allocated on this node"""

    min_mem_ratio: Optional[float] = 0.0
    """Minimum memory ratio"""

    def update(self, other: "GPUConfig"):
        if other.model:
            self.model = other.model
        if other.count:
            self.count = other.count
        if other.memory:
            self.memory = other.memory
        if other.min_memory:
            self.min_memory = other.min_memory

    def to_spec(self):
        cuda = []
        min_memory = max(int(self.memory * self.min_mem_ratio), self.min_memory)
        cuda.extend(
            [
                CudaSpecification(self.memory, self.model, min_memory)
                for _ in range(self.count)
            ]
        )
        return cuda


@dataclass
class CPUConfig(YAMLDataClass):
    cpu_per_gpu: int = 0
    """Number of CPU per GPU"""

    mem_per_cpu: Annotated[int, Initialize(humanfriendly.parse_size)] = 0
    """Memory per CPU"""

    cores: int = 0

    memory: Annotated[int, Initialize(parse_size)] = 0

    def update(self, other: "CPUConfig"):
        if other.cpu_per_gpu:
            self.cpu_per_gpu = other.cpu_per_gpu
        if other.mem_per_cpu:
            self.mem_per_cpu = other.mem_per_cpu
        if other.memory:
            self.memory = other.memory
        if other.cores:
            self.cores = other.cores

    def to_spec(self):
        return CPUSpecification(
            memory=self.memory,
            cores=self.cores,
            mem_per_cpu=self.mem_per_cpu,
            cpu_per_gpu=self.cpu_per_gpu,
        )


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
class SlurmNodeConfiguration(YAMLDataClass):
    max_duration: Annotated[int, Initialize(humanfriendly.parse_timespan)] = 0
    """Maximum duration of a job"""

    gpu: GPUConfig = field(default_factory=GPUConfig)
    """GPU Configuration"""

    cpu: CPUConfig = field(default_factory=CPUConfig)
    """CPU Configuration"""

    def update(self, other: "SlurmNodeConfiguration"):
        if other.max_duration:
            self.max_duration = other.max_duration

        if other.gpu:
            self.gpu.update(other.gpu)

        if other.cpu:
            self.cpu.update(other.cpu)

    def to_host_spec(self):
        spec = SlurmHostSpecification(
            cpu=(self.cpu or CPUConfig()).to_spec(),
            cuda=(self.gpu or GPUConfig()).to_spec(),
        )
        spec.max_duration = self.max_duration or 0
        return spec


@dataclass
class SlurmNodes(YAMLDataClass):
    features: List[str] = field(default_factory=list)
    """Nodes features"""

    hosts: List[str] = field(default_factory=list)
    """List of hostnames"""

    configuration: Optional[SlurmNodeConfiguration] = None
    """(optional) nodes configuration"""

    count: int = 0
    """Number of hosts (if list of hosts is empty)"""


@dataclass
class SlurmPartition(YAMLDataClass):
    """A Slurm partition"""

    accounts: List[str] = field(default_factory=list)
    """List of accounts for this partition with the associated priority modifier"""

    qos: List[str] = field(default_factory=list)
    """List of QoS for this partition with the associated priority modifier"""

    nodes: List[SlurmNodes] = field(default_factory=list)
    """List of nodes"""

    configuration: Optional[SlurmNodeConfiguration] = None
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
class SlurmFeature(YAMLDataClass):
    """Associate a configuration with a Slurm feature"""

    configuration: Optional[SlurmNodeConfiguration] = None


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
        self.main_config = config
        self.config = deepcopy(config.configuration)

        self.config.max_duration = (
            self.partition.configuration.max_duration
            if self.partition.configuration
            else None
        )
        self.priority = partition.priority
        self.qos_id = None
        self.min_gpu = 0

    def update(self, config: SlurmNodeConfiguration):
        self.config = deepcopy(self.config)
        self.config.update(config)

    def update_with_qos(self, qos_id: str):
        self.qos_id = qos_id
        if qos := self.main_config.qos.get(qos_id, None):
            self.priority += qos.priority
            self.min_gpu = qos.min_gpu
            if qos.max_duration > 0:
                self.config.max_duration = qos.max_duration

    def get_host(self) -> SlurmHostSpecification:
        host = self.config.to_host_spec()
        host.priority = self.priority
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
        s = f"""({")|(".join(it)})"""
        return None if s == "()" else s


class MatchingSpec:
    def __init__(self):
        self.fbf = FeatureBooleanFormula()
        self.hosts: set[str] = set()
        self.partitions: Set[str] = set()
        self.qos: Optional[str] = None
        self.account: Optional[str] = None
        self.mem_per_cpu: int = 0

    def update(self, host_spec: SlurmHostSpecification):
        if host_spec.qos_id != self.qos and self.qos is not None:
            # Cannot update with other QoS
            return
        self.qos = host_spec.qos_id

        if host_spec.account_id != self.account and self.account is not None:
            # Cannot update with other account
            return

        if (
            host_spec.cpu.mem_per_cpu > 0
            and self.mem_per_cpu > 0
            and host_spec.cpu.mem_per_cpu != self.mem_per_cpu
        ):
            # Cannot update with different mem per cpu
            return

        if host_spec.cpu.mem_per_cpu:
            self.mem_per_cpu = host_spec.cpu.mem_per_cpu

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

    path: str = "/usr/bin"
    """Path for SLURM commands"""

    use_features: bool = True
    """Whether features should be used"""

    use_hosts: bool = True
    """Whether hosts should be used in the query"""

    use_memory_contraint: bool = True
    """Whether memory constraint can be specified"""

    query_slurm: bool = False
    """True to query SLURM directly (using scontrol)"""

    tags: List[str] = field(default_factory=list)

    weight: int = 0

    qos: Dict[str, SlurmQOS] = field(default_factory=lambda: {})

    features_regex: Annotated[
        List[re.Pattern],
        Initialize(lambda regexps: [re.compile(regex) for regex in regexps]),
    ] = field(default_factory=list)
    """
    Regex to get the information from features
        - CUDA: cuda:count, cuda:memory
    """

    features: Dict[str, SlurmFeature] = field(default_factory=lambda: {})
    """List of features with associated configurations"""

    configuration: Optional[SlurmNodeConfiguration] = None
    """Partition configuration"""

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
                nodes_spec.update(self.configuration)

                # Set partition GPU
                if partition.configuration:
                    nodes_spec.update(partition.configuration)

                if node.configuration:
                    nodes_spec.update(node.configuration)

                for feature in node.features:
                    # Use feature data directly
                    if data := self.features.get(feature, None):
                        nodes_spec.update(data.configuration)

                    # logger.debug("Looking at %s", self.features_regex)
                    for regex in self.features_regex:
                        # logger.debug("%s/%s => %s", regex, tag, regex.match(tag))
                        if m := regex.match(feature):
                            d = m.groupdict()
                            if _count := d.get("cuda_count", None):
                                nodes_spec.config.gpu.count = int(_count)
                            if memory := d.get("cuda_memory", None):
                                nodes_spec.config.gpu.memory = humanfriendly.parse_size(
                                    memory
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

            launcher = SlurmLauncher(
                connector=registry.getConnector(self.connector), binpath=self.path
            )

            launcher.options.partition = ",".join(matching_spec.partitions)
            launcher.options.gpus_per_node = (
                len(current_match.requirement.cuda_gpus)
                if current_match.requirement.cuda_gpus
                else None
            )

            launcher.options.qos = matching_spec.qos
            launcher.options.account = matching_spec.account

            if current_match.requirement.cpu.cores > 0:
                launcher.options.cpus_per_task = current_match.requirement.cpu.cores

            if current_match.requirement.cpu.memory > 0:
                if self.use_memory_contraint:
                    launcher.options.mem = (
                        f"{current_match.requirement.cpu.memory // (1024*1024)}M"
                    )
                else:
                    assert (
                        matching_spec.mem_per_cpu > 0
                    ), "Memory per CPU should be specified"
                    cpus_per_task = math.ceil(
                        current_match.requirement.cpu.memory / matching_spec.mem_per_cpu
                    )
                    launcher.options.cpus_per_task = max(
                        launcher.options.cpus_per_task, cpus_per_task
                    )

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
