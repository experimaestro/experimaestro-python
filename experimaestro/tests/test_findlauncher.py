from experimaestro.launcherfinder.specs import (
    CPUSpecification,
    CudaSpecification,
    HostSpecification,
    cpu,
    cuda_gpu,
)
from humanfriendly import parse_size


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
        CPUSpecification(parse_size("12G"), 1), [CudaSpecification(parse_size("48G"))]
    )
    req = req1 | req2
    m = req.match(host)
    assert m is not None
    assert m.requirement is req2
