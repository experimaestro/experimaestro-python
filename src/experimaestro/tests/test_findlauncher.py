from experimaestro.launcherfinder.registry import LauncherRegistry
from experimaestro.launcherfinder.specs import (
    CPUSpecification,
    CudaSpecification,
    HostSpecification,
    cpu,
    cuda_gpu,
    HostSimpleRequirement,
)
from experimaestro.launcherfinder import parse
from humanfriendly import parse_size, parse_timespan

from experimaestro.launchers.slurm import SlurmLauncher
from experimaestro.utils.resources import ResourcePathWrapper


def test_findlauncher_specs():
    """Test the launcher finder for various launchers"""
    req1 = cuda_gpu(mem="14G") * 2 & cpu(mem="7G")
    assert req1.cpu.cores == 1
    assert req1.cpu.memory == parse_size("7G")
    assert len(req1.cuda_gpus) == 2
    assert req1.cuda_gpus[0].memory == parse_size("14G")
    assert req1.cuda_gpus[1].memory == parse_size("14G")

    req2 = cuda_gpu(mem="10G") & cpu(mem="7G")
    assert req2.cpu.cores == 1
    assert req2.cpu.memory == parse_size("7G")
    assert len(req2.cuda_gpus) == 1
    assert req2.cuda_gpus[0].memory == parse_size("10G")

    # Match host
    host = HostSpecification(
        cpu=CPUSpecification(parse_size("12G"), 1),
        cuda=[CudaSpecification(parse_size("48G"))],
    )
    req = req1 | req2
    m = req.match(host)
    assert m is not None
    assert m.requirement is req2


def test_findlauncher_specs_gpu_mem():
    host = HostSpecification(
        cpu=CPUSpecification(parse_size("12G"), 1),
        cuda=[CudaSpecification(parse_size("48G"), min_memory=parse_size("24G"))],
    )

    # Not enough requested
    req = cuda_gpu(mem="20G")
    assert req.match(host) is None

    # Too much requested
    req = cuda_gpu(mem="50G")
    assert req.match(host) is None

    # Just enough
    req = cuda_gpu(mem="30G")
    assert req.match(host) is not None


def test_findlauncher_parse():
    (r,) = parse("""duration=4 d & cuda(mem=4G) * 2 & cpu(mem=400M, cores=4)""")
    assert isinstance(r, HostSimpleRequirement)

    assert len(r.cuda_gpus) == 2
    assert r.cuda_gpus[0].memory == parse_size("4G")
    assert r.duration == parse_timespan("4 d")
    assert r.cpu.memory == parse_size("400M")
    assert r.cpu.cores == 4


def split_set(s: str, sep=","):
    return set(s.split(sep))


def slurm_constraint_split(constraint: str):
    return [split_set(c[1:-1], "&") for c in constraint.split("|")].sort()


def test_findlauncher_slurm():
    path = ResourcePathWrapper.create(f"{__package__ }.launchers", "config_slurm")

    assert (path / "launchers.py").is_file()

    registry = LauncherRegistry(path)
    launcher = registry.find("""duration=4 days & cuda(mem=24G) * 2""")
    assert isinstance(launcher, SlurmLauncher)

    options = launcher.options

    assert options.gpus_per_node == 2
    assert split_set(options.partition) == set(["hard", "electronic"])
    assert slurm_constraint_split(options.constraint) == slurm_constraint_split(
        "(A6000&GPU2&GPUM48G)|(A6000&GPU3&GPUM48G)|(RTX&GPU4&GPUM48G)"
    )
