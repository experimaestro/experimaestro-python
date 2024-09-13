from experimaestro.launcherfinder.specs import (
    CPUSpecification,
    CudaSpecification,
    HostRequirement,
    HostSpecification,
)
from experimaestro.launchers.slurm.base import SlurmLauncher, SlurmOptions

GIGA = 1024**3


def find_launcher(requirements: HostRequirement, tags: set[str] = set()):
    host = HostSpecification(
        cpu=CPUSpecification(cores=16, memory=32 * GIGA),
        max_duration=3600 * 24 * 10,
        cuda=[CudaSpecification(memory=32 * GIGA) for _ in range(4)],
    )
    if match := requirements.match(host):
        return SlurmLauncher(
            options=SlurmOptions(
                gpus_per_node=len(match.requirement.cuda_gpus),
                partition="hard,electronic",
                constraint="(A6000&GPU2&GPUM48G)|(A6000&GPU3&GPUM48G)|(RTX&GPU4&GPUM48G)",
            )
        )
