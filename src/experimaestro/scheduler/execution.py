"""Where a job runs: hosts and allocated resources

The record is collected by the job process when it starts
(:meth:`ExecutionInfo.current`), from its host name and, when the job runs in a
SLURM allocation, the SLURM environment. It is stored in the job status
(``execution`` in ``status.json``) and sent with the "running" event of the
job, so that every monitor can show it without querying the launcher.
"""

import os
import re
import socket
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

#: A part of a SLURM node list: a prefix followed by bracketed ranges or text
_NODELIST_PART = re.compile(r"\[([^\]]*)\]|([^\[\],]+)")


def _split_nodelist(nodelist: str) -> list[str]:
    """Split a node list at the commas that are not within brackets"""
    parts, depth, start = [], 0, 0
    for ix, char in enumerate(nodelist):
        match char:
            case "[":
                depth += 1
            case "]":
                depth -= 1
            case "," if depth == 0:
                parts.append(nodelist[start:ix])
                start = ix + 1
    parts.append(nodelist[start:])
    return [part.strip() for part in parts if part.strip()]


def _expand_ranges(ranges: str) -> list[str]:
    """Expand the content of brackets (``01-03,07``), keeping the padding"""
    values: list[str] = []
    for item in ranges.split(","):
        low, sep, high = item.strip().partition("-")
        if sep and low.isdigit() and high.isdigit():
            width = len(low)
            values.extend(str(n).zfill(width) for n in range(int(low), int(high) + 1))
        elif item.strip():
            values.append(item.strip())
    return values


def expand_nodelist(nodelist: str) -> list[str]:
    """Host names of a SLURM node list, e.g. ``node[01-03,07],gpu5``"""
    hosts: list[str] = []
    for part in _split_nodelist(nodelist):
        names = [""]
        for ranges, text in _NODELIST_PART.findall(part):
            choices = _expand_ranges(ranges) if ranges else [text]
            names = [name + choice for name in names for choice in choices]
        hosts.extend(name for name in names if name)
    return hosts


def _split_ids(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


@dataclass
class ExecutionInfo:
    """Where a job runs (or ran, for its last run)"""

    #: Hosts of the job: the nodes of its allocation (the one running the job
    #: process first), or the host of the job process
    hosts: list[str] = field(default_factory=list)
    #: SLURM partition
    partition: str | None = None
    #: Number of CPUs allocated on the node running the job process
    cpus: int | None = None
    #: Identifiers of the GPUs allocated on the node running the job process
    gpus: list[str] = field(default_factory=list)

    @staticmethod
    def current(environ: Mapping[str, str] | None = None) -> "ExecutionInfo":
        """Where the current process runs"""
        if environ is None:
            environ = os.environ

        # Outside of an allocation, the node list is not set
        hosts = expand_nodelist(environ.get("SLURM_JOB_NODELIST", ""))
        node = environ.get("SLURMD_NODENAME")
        if node in hosts:
            hosts.remove(node)
            hosts.insert(0, node)
        elif not hosts:
            hosts = [socket.gethostname()]

        cpus = environ.get("SLURM_CPUS_ON_NODE", "")
        gpus = (
            _split_ids(environ.get("SLURM_STEP_GPUS"))
            or _split_ids(environ.get("SLURM_JOB_GPUS"))
            or _split_ids(environ.get("CUDA_VISIBLE_DEVICES"))
        )
        return ExecutionInfo(
            hosts=hosts,
            partition=environ.get("SLURM_JOB_PARTITION") or None,
            cpus=int(cpus) if cpus.isdigit() else None,
            gpus=gpus,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "hosts": list(self.hosts),
            "partition": self.partition,
            "cpus": self.cpus,
            "gpus": list(self.gpus),
        }

    @staticmethod
    def from_state_dict(data: Any) -> "ExecutionInfo | None":
        """Read a record (None if there is none or it cannot be read)"""
        if not isinstance(data, dict):
            return None
        cpus = data.get("cpus")
        return ExecutionInfo(
            hosts=[str(host) for host in data.get("hosts") or []],
            partition=data.get("partition") or None,
            cpus=cpus if isinstance(cpus, int) else None,
            gpus=[str(gpu) for gpu in data.get("gpus") or []],
        )

    def describe(self) -> str:
        """One-line description, e.g. ``node042 (partition gpu, 8 CPUs, GPU 0)``"""
        resources = []
        if self.partition:
            resources.append(f"partition {self.partition}")
        if self.cpus:
            resources.append(f"{self.cpus} CPU{'s' if self.cpus > 1 else ''}")
        if self.gpus:
            label = "GPUs" if len(self.gpus) > 1 else "GPU"
            resources.append(f"{label} {','.join(self.gpus)}")
        text = ", ".join(self.hosts)
        if resources:
            text = f"{text} ({', '.join(resources)})" if text else ", ".join(resources)
        return text
