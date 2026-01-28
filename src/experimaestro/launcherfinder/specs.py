from abc import ABC, abstractmethod
from enum import Enum
import logging
import math
from attr import Factory
from attrs import define
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import List, Optional, Union
from humanfriendly import parse_size, format_size, parse_timespan

# --- Host specification part


class AcceleratorType(Enum):
    """Types of accelerators supported."""

    CUDA = "cuda"  # NVIDIA CUDA GPUs (dedicated memory)
    MPS = "mps"  # Apple Metal Performance Shaders (unified memory)
    ROCM = "rocm"  # AMD ROCm GPUs (future)


@dataclass
class AcceleratorSpecification:
    """Generic accelerator (GPU-like device) specification.

    This can match any accelerator type (CUDA, MPS, ROCm, etc.) based on
    memory requirements alone. Use this when you don't care about the
    specific accelerator type.

    For type-specific requirements, use CudaSpecification or MPSSpecification.
    """

    memory: int = 0
    """Memory in bytes"""

    model: str = ""
    """Model name"""

    min_memory: int = 0
    """Minimum request memory (in bytes)"""

    @property
    def accelerator_type(self) -> AcceleratorType | None:
        """Type of accelerator (None for generic)"""
        return None

    @property
    def unified_memory(self) -> bool:
        """If True, memory is shared with CPU (e.g., Apple Silicon)"""
        return False

    def match(self, spec: "AcceleratorSpecification") -> bool:
        """Returns True if this host accelerator can satisfy the spec requirement.

        Matching rules:
        - If spec is generic (AcceleratorSpecification), any accelerator matches
        - If spec is specific (CudaSpecification, MPSSpecification), types must match
        """
        # Check if the requested spec requires a specific type
        spec_type = spec.accelerator_type
        if spec_type is not None:
            # Specific type requested - must match exactly
            if self.accelerator_type != spec_type:
                return False

        # Check memory bounds
        if self.memory < spec.memory:
            return False
        if self.min_memory > spec.memory:
            return False

        return True

    def __repr__(self):
        mem_str = format_size(self.memory, binary=True)
        min_str = format_size(self.min_memory, binary=True)
        return f"Accelerator({self.model} max={mem_str}/min={min_str})"


@dataclass
class CudaSpecification(AcceleratorSpecification):
    """NVIDIA CUDA GPU specification (dedicated GPU memory).

    Only matches CUDA GPUs - will not match MPS or other accelerator types.
    """

    memory: int = 0
    """Memory (in bytes)"""

    model: str = ""
    """CUDA card model name"""

    min_memory: int = 0
    """Minimum request memory (in bytes)"""

    @property
    def accelerator_type(self) -> AcceleratorType:
        return AcceleratorType.CUDA

    def __repr__(self):
        return f"CUDA({self.model} max={format_size(self.memory, binary=True)}/min={format_size(self.min_memory, binary=True)})"


@dataclass
class MPSSpecification(AcceleratorSpecification):
    """Apple Metal Performance Shaders (MPS) specification.

    MPS uses unified memory - GPU memory is shared with CPU RAM.
    When a task requests GPU memory on MPS, it consumes system RAM.

    Only matches MPS - will not match CUDA or other accelerator types.
    """

    memory: int = 0
    """Memory in bytes (shared with CPU)"""

    model: str = ""
    """Apple Silicon model (e.g., 'M1', 'M2 Pro')"""

    min_memory: int = 0
    """Minimum request memory (in bytes)"""

    @property
    def accelerator_type(self) -> AcceleratorType:
        return AcceleratorType.MPS

    @property
    def unified_memory(self) -> bool:
        return True

    def __repr__(self):
        return f"MPS({self.model} mem={format_size(self.memory, binary=True)} unified)"


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

    def __repr__(self):
        return f"CPU(mem={format_size(self.memory, binary=True)}, cores={self.cores})"

    def match(self, other: "CPUSpecification"):
        return (self.memory >= other.memory) and (self.cores >= other.cores)

    def total_memory(self, gpus: int = 0):
        return max(
            self.memory,
            self.mem_per_cpu * self.cores,
            self.cpu_per_gpu * self.mem_per_cpu * gpus,
        )


@define(kw_only=True)
class HostSpecification:
    """Specifies how the host is set.

    Supports both CUDA GPUs and other accelerators (MPS, ROCm, etc.).
    Use `accelerators` for the generic list, or `cuda` for backwards compatibility.

    Examples:
        # New style - generic accelerators
        host = HostSpecification(accelerators=[CudaSpecification(memory=24*1024**3)])

        # Backwards compatible - cuda shorthand
        host = HostSpecification(cuda=[CudaSpecification(memory=24*1024**3)])
    """

    accelerators: List[AcceleratorSpecification] = Factory(list)
    """All accelerators (GPUs) available on this host"""

    cuda: List[CudaSpecification] = Factory(list)
    """CUDA GPUs (backwards compatibility, merged into accelerators)"""

    cpu: CPUSpecification = Factory(CPUSpecification)
    """CPU specification for this host"""

    priority: int = 0
    """Priority for this host (higher better)"""

    max_duration: int = 0
    """Max job duration (in seconds)"""

    min_gpu: int = 0
    """Minimum number of allocated GPUs"""

    def __attrs_post_init__(self):
        # Merge cuda into accelerators for backwards compatibility
        if self.cuda:
            self.accelerators = list(self.accelerators) + list(self.cuda)
            # Clear cuda to avoid double-counting
            object.__setattr__(self, "cuda", [])
        # Sort by memory descending
        self.accelerators = sorted(self.accelerators, key=lambda a: -a.memory)


# --- Query part


@dataclass
class MatchRequirement:
    score: float
    requirement: "HostSimpleRequirement"


class HostRequirement(ABC):
    """A requirement must be a disjunction of host requirements"""

    requirements: List["HostSimpleRequirement"]
    """List of requirements (by order of priority)"""

    def __init__(self) -> None:
        self.requirements = []

    def __or__(self, other: "HostRequirement"):
        return RequirementUnion(self, other)

    def match(self, host: HostSpecification) -> Optional[MatchRequirement]:
        raise NotImplementedError()

    @abstractmethod
    def multiply_duration(self, coefficient: float) -> "HostRequirement":
        """Returns a new HostRequirement with a duration multiplied by the
        provided coefficient"""
        ...


class RequirementUnion(HostRequirement):
    """Ordered list of simple host requirements -- the first one is the priority"""

    requirements: List["HostSimpleRequirement"]

    def __init__(self, *requirements: "HostSimpleRequirement"):
        self.requirements = list(requirements)

    def add(self, requirement: "HostRequirement"):
        match requirement:
            case HostSimpleRequirement():
                self.requirements.extend(*requirement.requirements)
            case RequirementUnion():
                self.requirements.append(requirement)
            case _:
                raise RuntimeError("Cannot handle type %s", type(requirement))
        return self

    def match(self, host: HostSpecification) -> Optional[MatchRequirement]:
        """Returns the matched requirement (if any)"""

        argmax: Optional[MatchRequirement] = None

        for req in self.requirements:
            max_score = float("-inf") if argmax is None else argmax.score

            if match := req.match(host):
                if match.score > max_score:
                    argmax = MatchRequirement(match.score, req)

        return argmax

    def multiply_duration(self, coefficient: float) -> "RequirementUnion":
        return RequirementUnion(
            *[r.multiply_duration(coefficient) for r in self.requirements]
        )

    def __repr__(self):
        return " | ".join(repr(r) for r in self.requirements)


class HostSimpleRequirement(HostRequirement):
    """Simple host requirement"""

    accelerators: List["AcceleratorSpecification"]
    """Specification for accelerators (GPUs)"""

    cpu: "CPUSpecification"
    """Specification for CPU"""

    duration: int
    """Requested duration (in seconds)"""

    def __repr__(self):
        return f"Req(cpu={self.cpu}, accelerators={self.accelerators}, duration={self.duration})"

    def multiply_duration(self, coefficient: float) -> "HostSimpleRequirement":
        r = HostSimpleRequirement(self)
        r.duration = math.ceil(self.duration * coefficient)
        return r

    def __init__(self, *reqs: "HostSimpleRequirement"):
        self.accelerators = []
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
        self.accelerators.extend(req.accelerators)
        self.accelerators.sort(key=lambda a: -a.memory)

    @property
    def cuda_gpus(self) -> List["CudaSpecification"]:
        """CUDA GPUs (backwards compatibility alias).

        Returns only CUDA accelerators from the accelerators list.
        """
        return [a for a in self.accelerators if isinstance(a, CudaSpecification)]

    def match(self, host: HostSpecification) -> Optional[MatchRequirement]:
        if self.accelerators:
            if len(host.accelerators) < len(self.accelerators):
                logging.debug(
                    "Not enough accelerators (%d < %d)",
                    len(host.accelerators),
                    len(self.accelerators),
                )
                return None

            # Match accelerators - each requested accelerator must find a match
            # Sort both by memory descending for greedy matching
            host_accels = sorted(host.accelerators, key=lambda a: -a.memory)
            req_accels = sorted(self.accelerators, key=lambda a: -a.memory)

            for host_accel, req_accel in zip(host_accels, req_accels):
                if not host_accel.match(req_accel):
                    logging.debug(
                        "Accelerator mismatch: host %s cannot satisfy %s",
                        host_accel,
                        req_accel,
                    )
                    return None

        if len(self.accelerators) < host.min_gpu:
            logging.debug(
                "Not enough requested accelerators (min=%d > %d)",
                host.min_gpu,
                len(self.accelerators),
            )
            return None

        if not host.cpu.match(self.cpu):
            return None

        # For unified memory systems (MPS), check that combined CPU + GPU memory
        # doesn't exceed total system memory
        unified_gpu_memory = sum(
            a.memory for a in self.accelerators if a.unified_memory
        )
        if unified_gpu_memory > 0:
            total_requested = self.cpu.memory + unified_gpu_memory
            if total_requested > host.cpu.memory:
                logging.debug(
                    "Unified memory exceeded: requested %d (CPU) + %d (GPU) > %d available",
                    self.cpu.memory,
                    unified_gpu_memory,
                    host.cpu.memory,
                )
                return None

        if host.max_duration > 0 and self.duration > host.max_duration:
            return None

        return MatchRequirement(host.priority, self)

    def __mul__(self, count: int) -> "HostSimpleRequirement":
        if count == 1:
            return self

        _self = deepcopy(self)
        for _ in range(count - 1):
            _self.accelerators.extend(self.accelerators)
        _self.accelerators.sort(key=lambda a: -a.memory)

        return _self


def cpu(*, mem: Optional[str] = None, cores: int = 1):
    """CPU requirement"""
    r = HostSimpleRequirement()
    r.cpu = CPUSpecification(parse_size(mem) if mem else 0, cores)
    return r


def cuda_gpu(*, mem: Optional[str] = None):
    """CUDA GPU requirement (NVIDIA only).

    Use this when you specifically need NVIDIA CUDA support.
    Will not match MPS or other accelerator types.
    """
    _mem = parse_size(mem) if mem else 0
    r = HostSimpleRequirement()
    r.accelerators.append(CudaSpecification(_mem))
    return r


def mps_gpu(*, mem: Optional[str] = None):
    """Apple MPS GPU requirement (Apple Silicon only).

    MPS uses unified memory - the GPU shares RAM with the CPU.
    Use this when you specifically need Apple Metal support.
    Will not match CUDA or other accelerator types.
    """
    _mem = parse_size(mem) if mem else 0
    r = HostSimpleRequirement()
    r.accelerators.append(MPSSpecification(_mem))
    return r


def gpu(*, mem: Optional[str] = None):
    """Generic GPU/accelerator requirement.

    Matches any accelerator type (CUDA, MPS, ROCm, etc.) that satisfies
    the memory requirement. Use this for cross-platform compatibility.
    """
    _mem = parse_size(mem) if mem else 0
    r = HostSimpleRequirement()
    r.accelerators.append(AcceleratorSpecification(_mem))
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
