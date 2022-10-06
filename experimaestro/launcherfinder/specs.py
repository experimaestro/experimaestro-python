from copy import copy
from dataclasses import dataclass
from difflib import Match
from enum import Enum
from typing import List, Optional, Set, Tuple, Union
from humanfriendly import parse_size, format_size, parse_timespan

# --- Host specification part


@dataclass
class CudaSpecification:
    memory: int
    model: str = ""

    def __lt__(self, other: "CudaSpecification"):
        return self.memory < other.memory

    def __repr__(self):
        return f"CUDA({self.model} {format_size(self.memory)})"


@dataclass
class CPUSpecification:
    memory: int
    """Memory in bytes"""

    cores: int
    """Number of cores"""

    def __lt__(self, other: "CPUSpecification"):
        return self.memory < other.memory and self.cores < other.cores


class HostSpecification:
    _cuda: List[CudaSpecification]
    _cpu: CPUSpecification

    def __init__(self, cpu: CPUSpecification, cuda: List[CudaSpecification]) -> None:
        self.cpu = cpu
        self.cuda = sorted(cuda)

    def __repr__(self) -> str:
        return f"Host({self.cpu}, {self.cuda})"


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
    requirements: List["HostSimpleRequirement"]

    def __init__(self, *requirements: "HostSimpleRequirement"):
        self.requirements = list(requirements)

    def match(self, host: HostSpecification) -> Optional[MatchRequirement]:
        """Returns the matched requirement (if any)"""
        argmax: Optional[MatchRequirement] = None
        for ix, r in enumerate(self.requirements):
            max_score = 0.0 if argmax is None else argmax.score
            if match := r.match(host):
                if match.score > max_score:
                    argmax = MatchRequirement(match.score, r)

        return argmax


class HostSimpleRequirement(HostRequirement):
    """Simple host requirement"""

    cuda_gpus: List["CudaSpecification"]
    cpu: "CPUSpecification"

    duration: int
    """Requested duration (in seconds)"""

    def __repr__(self):
        return f"Req(cpu={self.cpu}, cuda={self.cuda_gpus})"

    def __init__(self, *reqs: "HostSimpleRequirement"):
        self.cuda_gpus = []
        self.cpu = CPUSpecification(0, 1)
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
                # print("GPU", host.cuda, self.cuda_gpus)
                return None

            for host_gpu, req_gpu in zip(host.cuda, self.cuda_gpus):
                if host_gpu < req_gpu:
                    # print("GPU", host_gpu, req_gpu)
                    return None

        if host.cpu < self.cpu:
            print("Mem", host.cpu, self.cpu)
            return None

        return MatchRequirement(1, self)

    def __mul__(self, count: int) -> "HostSimpleRequirement":
        if count == 1:
            return self

        _self = copy(self)
        for _ in range(count - 1):
            _self.cuda_gpus.extend(self.cuda_gpus)
        self.cuda_gpus.sort()

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

    Args:
        timespec: A string like 5h (5 hours), 10m (10 minutes) or 42s (42 seconds). parsable by [humanfriendly](https://humanfriendly.readthedocs.io/en/latest/api.html) or a number of seconds
    """
    r = HostSimpleRequirement()
    if isinstance(timespec, (int, float)):
        r.duration = int(timespec)
    else:
        r.duration = int(parse_timespan(timespec))

    return r
