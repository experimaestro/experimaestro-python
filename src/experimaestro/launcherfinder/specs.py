import logging
from attr import Factory
from attrs import define
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import List, Optional, Union
from humanfriendly import parse_size, format_size, parse_timespan

# --- Host specification part


@dataclass
class CudaSpecification:
    memory: int
    """Memory (in bytes)"""

    model: str = ""
    """CUDA card model name"""

    min_memory: int = 0
    """Minimum request memory (in bytes)"""

    def __lt__(self, other: "CudaSpecification"):
        return self.memory < other.memory

    def match(self, spec: "CudaSpecification"):
        """Returns True if the specification matches this host"""
        return (self.memory >= spec.memory) and (self.min_memory <= spec.memory)

    def __repr__(self):
        return (
            f"CUDA({self.model} "
            f"{format_size(self.memory)}/{format_size(self.min_memory)})"
        )


@dataclass
class CPUSpecification:
    memory: int = 0
    """Memory in bytes"""

    cores: int = 0
    """Number of cores"""

    mem_per_cpu: int = 0
    """Memory per CPU (0 if not defined)"""

    cpu_per_gpu: int = 0
    """Number of CPU per GPU (0 if not defined)"""

    def __lt__(self, other: "CPUSpecification"):
        return self.memory < other.memory and self.cores < other.cores

    def total_memory(self, gpus: int = 0):
        return max(
            self.memory,
            self.mem_per_cpu * self.cores,
            self.cpu_per_gpu * self.mem_per_cpu * gpus,
        )


@define(kw_only=True)
class HostSpecification:
    """Specifies how the host is set"""

    cuda: List[CudaSpecification] = Factory(list)
    """CUDA GPUs"""

    cpu: CPUSpecification = Factory(CPUSpecification)
    """CPU specification for this host"""

    priority: int = 0
    """Priority for this host (higher better)"""

    max_duration: int = 0
    """Max job duration (in seconds)"""

    min_gpu: int = 0
    """Minimum number of allocated GPUs"""

    def __post_init__(self):
        self.cuda = sorted(self.cuda)


# --- Query part


@dataclass
class MatchRequirement:
    score: float
    requirement: "HostSimpleRequirement"


class HostRequirement:
    """A requirement must be a disjunction of host requirements"""

    requirements: List["HostSimpleRequirement"]

    def __init__(self) -> None:
        self.requirements = []

    def __or__(self, other: "HostRequirement"):
        return RequirementUnion(self, other)

    def match(self, host: HostSpecification) -> Optional[MatchRequirement]:
        raise NotImplementedError()


class RequirementUnion(HostRequirement):
    """Ordered list of simple host requirements -- the first one is the priority"""

    requirements: List["HostSimpleRequirement"]

    def __init__(self, *requirements: "HostSimpleRequirement"):
        self.requirements = list(requirements)

    def match(self, host: HostSpecification) -> Optional[MatchRequirement]:
        """Returns the matched requirement (if any)"""

        argmax: Optional[MatchRequirement] = None

        for req in self.requirements:
            max_score = float("-inf") if argmax is None else argmax.score

            if match := req.match(host):
                if match.score > max_score:
                    argmax = MatchRequirement(match.score, req)

        return argmax


class HostSimpleRequirement(HostRequirement):
    """Simple host requirement"""

    cuda_gpus: List["CudaSpecification"]
    cpu: "CPUSpecification"

    duration: int
    """Requested duration (in seconds)"""

    def __repr__(self):
        return f"Req(cpu={self.cpu}, cuda={self.cuda_gpus}, duration={self.duration})"

    def __init__(self, *reqs: "HostSimpleRequirement"):
        self.cuda_gpus = []
        self.cpu = CPUSpecification(0, 0)
        self.duration = 0
        for req in reqs:
            self._add(req)

    def __and__(self, other: "HostSimpleRequirement"):
        newself = copy(self)
        newself._add(other)
        return newself

    def _add(self, req: "HostSimpleRequirement"):
        self.cpu.memory = max(req.cpu.memory, self.cpu.memory)
        self.cpu.cores = max(req.cpu.cores, self.cpu.cores)
        self.duration = max(req.duration, self.duration)
        self.cuda_gpus.extend(req.cuda_gpus)
        self.cuda_gpus.sort()

    def match(self, host: HostSpecification) -> Optional[MatchRequirement]:
        if self.cuda_gpus:
            if len(host.cuda) < len(self.cuda_gpus):
                logging.debug(
                    "Not enough CUDA gpus (%d < %d)",
                    len(host.cuda),
                    len(self.cuda_gpus),
                )
                return None

            for host_gpu, req_gpu in zip(host.cuda, self.cuda_gpus):
                if not host_gpu.match(req_gpu):
                    return None

        if len(self.cuda_gpus) < host.min_gpu:
            logging.debug(
                "Not enough requested CUDA gpus (min=%d > %d)",
                host.min_gpu,
                len(self.cuda_gpus),
            )
            return None

        if host.cpu < self.cpu:
            return None

        if host.max_duration > 0 and self.duration > host.max_duration:
            return None

        return MatchRequirement(host.priority, self)

    def __mul__(self, count: int) -> "HostSimpleRequirement":
        if count == 1:
            return self

        _self = deepcopy(self)
        for _ in range(count - 1):
            _self.cuda_gpus.extend(self.cuda_gpus)
        _self.cuda_gpus.sort()

        return _self


def cpu(*, mem: Optional[str] = None, cores: int = 1):
    """CPU requirement"""
    r = HostSimpleRequirement()
    r.cpu = CPUSpecification(parse_size(mem) if mem else 0, cores)
    return r


def cuda_gpu(*, mem: Optional[str] = None):
    """CUDA GPU requirement"""
    _mem = parse_size(mem) if mem else 0
    r = HostSimpleRequirement()
    r.cuda_gpus.append(CudaSpecification(_mem))
    return r


def duration(timespec: Union[str, int]):
    """Request a given time duration


    :param timespec: A string like 5h (5 hours), 10m (10 minutes) or 42s (42
        seconds). parsable by
        [humanfriendly](https://humanfriendly.readthedocs.io/en/latest/api.html)
        or a number of seconds
    """
    r = HostSimpleRequirement()
    if isinstance(timespec, (int, float)):
        r.duration = int(timespec)
    else:
        r.duration = int(parse_timespan(timespec))

    return r
