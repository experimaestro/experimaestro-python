import pytest

from experimaestro.launcherfinder.registry import LauncherRegistry
from experimaestro.launcherfinder.specs import (
    AcceleratorSpecification,
    AcceleratorType,
    CPUSpecification,
    CudaSpecification,
    MPSSpecification,
    HostSpecification,
    RequirementUnion,
    cpu,
    cuda_gpu,
    mps_gpu,
    gpu,
)
from experimaestro.launcherfinder import parse
from humanfriendly import parse_size, parse_timespan

from experimaestro.launchers.slurm import SlurmLauncher
from experimaestro.utils.resources import ResourcePathWrapper

# Mark all tests in this module as launcher tests
pytestmark = [pytest.mark.launchers]


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

    # Multiply
    req2 = req.multiply_duration(2)
    for i in range(2):
        assert req2.requirements[i].duration == req.requirements[i].duration * 2


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
    r = parse("""duration=4 d & cuda(mem=4G) * 2 & cpu(mem=400M, cores=4)""")
    assert isinstance(r, RequirementUnion)

    r = r.requirements[0]

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
    path = ResourcePathWrapper.create(f"{__package__}.launchers", "config_slurm")

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


# =============================================================================
# Tests for AcceleratorSpecification (generic accelerators)
# =============================================================================


class TestAcceleratorTypes:
    """Test accelerator type definitions and properties"""

    def test_cuda_specification_type(self):
        """CudaSpecification should have CUDA accelerator type"""
        spec = CudaSpecification(memory=parse_size("8G"))
        assert spec.accelerator_type == AcceleratorType.CUDA
        assert spec.unified_memory is False

    def test_mps_specification_type(self):
        """MPSSpecification should have MPS accelerator type with unified memory"""
        spec = MPSSpecification(memory=parse_size("16G"))
        assert spec.accelerator_type == AcceleratorType.MPS
        assert spec.unified_memory is True

    def test_generic_accelerator_type(self):
        """Generic AcceleratorSpecification should have no specific type"""
        spec = AcceleratorSpecification(memory=parse_size("8G"))
        assert spec.accelerator_type is None
        assert spec.unified_memory is False


class TestAcceleratorMatching:
    """Test accelerator matching logic"""

    def test_cuda_matches_cuda(self):
        """CUDA host should match CUDA requirement"""
        host_spec = CudaSpecification(memory=parse_size("24G"))
        req_spec = CudaSpecification(memory=parse_size("8G"))
        assert host_spec.match(req_spec) is True

    def test_cuda_not_matches_mps(self):
        """CUDA host should NOT match MPS requirement"""
        host_spec = CudaSpecification(memory=parse_size("24G"))
        req_spec = MPSSpecification(memory=parse_size("8G"))
        assert host_spec.match(req_spec) is False

    def test_mps_matches_mps(self):
        """MPS host should match MPS requirement"""
        host_spec = MPSSpecification(memory=parse_size("32G"))
        req_spec = MPSSpecification(memory=parse_size("8G"))
        assert host_spec.match(req_spec) is True

    def test_mps_not_matches_cuda(self):
        """MPS host should NOT match CUDA requirement"""
        host_spec = MPSSpecification(memory=parse_size("32G"))
        req_spec = CudaSpecification(memory=parse_size("8G"))
        assert host_spec.match(req_spec) is False

    def test_cuda_matches_generic(self):
        """CUDA host should match generic accelerator requirement"""
        host_spec = CudaSpecification(memory=parse_size("24G"))
        req_spec = AcceleratorSpecification(memory=parse_size("8G"))
        assert host_spec.match(req_spec) is True

    def test_mps_matches_generic(self):
        """MPS host should match generic accelerator requirement"""
        host_spec = MPSSpecification(memory=parse_size("32G"))
        req_spec = AcceleratorSpecification(memory=parse_size("8G"))
        assert host_spec.match(req_spec) is True

    def test_memory_insufficient(self):
        """Host should not match if memory is insufficient"""
        host_spec = CudaSpecification(memory=parse_size("8G"))
        req_spec = CudaSpecification(memory=parse_size("16G"))
        assert host_spec.match(req_spec) is False

    def test_min_memory_constraint(self):
        """Host with min_memory should reject requests below minimum"""
        host_spec = CudaSpecification(
            memory=parse_size("48G"), min_memory=parse_size("24G")
        )
        # Request below minimum - should fail
        req_below = CudaSpecification(memory=parse_size("16G"))
        assert host_spec.match(req_below) is False

        # Request at minimum - should pass
        req_at_min = CudaSpecification(memory=parse_size("24G"))
        assert host_spec.match(req_at_min) is True

        # Request above minimum - should pass
        req_above = CudaSpecification(memory=parse_size("32G"))
        assert host_spec.match(req_above) is True


class TestHostSpecificationAccelerators:
    """Test HostSpecification with different accelerator types"""

    def test_host_with_cuda(self):
        """Host with CUDA GPUs should match CUDA requirements"""
        host = HostSpecification(
            cpu=CPUSpecification(parse_size("64G"), 8),
            accelerators=[CudaSpecification(parse_size("24G"))],
        )
        req = cuda_gpu(mem="16G")
        assert req.match(host) is not None

    def test_host_with_mps(self):
        """Host with MPS should match MPS requirements"""
        host = HostSpecification(
            cpu=CPUSpecification(parse_size("32G"), 8),
            accelerators=[MPSSpecification(parse_size("32G"))],
        )
        req = mps_gpu(mem="8G")
        assert req.match(host) is not None

    def test_host_cuda_not_match_mps_req(self):
        """Host with CUDA should NOT match MPS requirements"""
        host = HostSpecification(
            cpu=CPUSpecification(parse_size("64G"), 8),
            accelerators=[CudaSpecification(parse_size("24G"))],
        )
        req = mps_gpu(mem="8G")
        assert req.match(host) is None

    def test_host_mps_not_match_cuda_req(self):
        """Host with MPS should NOT match CUDA requirements"""
        host = HostSpecification(
            cpu=CPUSpecification(parse_size("32G"), 8),
            accelerators=[MPSSpecification(parse_size("32G"))],
        )
        req = cuda_gpu(mem="8G")
        assert req.match(host) is None

    def test_host_matches_generic_gpu(self):
        """Both CUDA and MPS hosts should match generic gpu() requirements"""
        cuda_host = HostSpecification(
            cpu=CPUSpecification(parse_size("64G"), 8),
            accelerators=[CudaSpecification(parse_size("24G"))],
        )
        mps_host = HostSpecification(
            cpu=CPUSpecification(parse_size("32G"), 8),
            accelerators=[MPSSpecification(parse_size("32G"))],
        )
        req = gpu(mem="8G")
        assert req.match(cuda_host) is not None
        assert req.match(mps_host) is not None

    def test_cuda_backwards_compatibility(self):
        """cuda= constructor argument should populate accelerators"""
        # Using cuda= in constructor (backwards compatible)
        host = HostSpecification(
            cpu=CPUSpecification(parse_size("64G"), 8),
            cuda=[CudaSpecification(parse_size("24G"))],
        )
        # cuda= is merged into accelerators
        assert len(host.accelerators) == 1
        assert isinstance(host.accelerators[0], CudaSpecification)

        # Using accelerators= directly (new style)
        host2 = HostSpecification(
            cpu=CPUSpecification(parse_size("64G"), 8),
            accelerators=[CudaSpecification(parse_size("24G"))],
        )
        assert len(host2.accelerators) == 1
        assert isinstance(host2.accelerators[0], CudaSpecification)


class TestParseNewSyntax:
    """Test parsing of new accelerator syntax (mps, gpu)"""

    def test_parse_mps(self):
        """Parse mps() requirement"""
        r = parse("mps(mem=8G)")
        assert isinstance(r, RequirementUnion)
        req = r.requirements[0]
        assert len(req.accelerators) == 1
        assert isinstance(req.accelerators[0], MPSSpecification)
        assert req.accelerators[0].memory == parse_size("8G")

    def test_parse_mps_multiplied(self):
        """Parse mps() with multiplier (multiple MPS accelerators)"""
        r = parse("mps(mem=4G) * 2")
        req = r.requirements[0]
        assert len(req.accelerators) == 2
        assert all(isinstance(a, MPSSpecification) for a in req.accelerators)

    def test_parse_gpu_generic(self):
        """Parse generic gpu() requirement"""
        r = parse("gpu(mem=16G)")
        assert isinstance(r, RequirementUnion)
        req = r.requirements[0]
        assert len(req.accelerators) == 1
        assert isinstance(req.accelerators[0], AcceleratorSpecification)
        # Should NOT be a subclass instance
        assert type(req.accelerators[0]) is AcceleratorSpecification
        assert req.accelerators[0].memory == parse_size("16G")

    def test_parse_gpu_multiplied(self):
        """Parse generic gpu() with multiplier"""
        r = parse("gpu(mem=8G) * 4")
        req = r.requirements[0]
        assert len(req.accelerators) == 4
        assert all(type(a) is AcceleratorSpecification for a in req.accelerators)

    def test_parse_combined_requirements(self):
        """Parse combined CPU + GPU requirements"""
        r = parse("duration=2h & cpu(mem=32G, cores=4) & mps(mem=16G)")
        req = r.requirements[0]
        assert req.duration == parse_timespan("2h")
        assert req.cpu.memory == parse_size("32G")
        assert req.cpu.cores == 4
        assert len(req.accelerators) == 1
        assert isinstance(req.accelerators[0], MPSSpecification)

    def test_parse_alternatives_cross_platform(self):
        """Parse cross-platform alternatives (CUDA | MPS)"""
        r = parse("cuda(mem=8G) | mps(mem=8G)")
        assert len(r.requirements) == 2

        # First alternative: CUDA
        cuda_req = r.requirements[0]
        assert len(cuda_req.accelerators) == 1
        assert isinstance(cuda_req.accelerators[0], CudaSpecification)

        # Second alternative: MPS
        mps_req = r.requirements[1]
        assert len(mps_req.accelerators) == 1
        assert isinstance(mps_req.accelerators[0], MPSSpecification)

    def test_parse_cuda_backwards_compatible(self):
        """Existing cuda() syntax should still work"""
        r = parse("cuda(mem=24G) * 2")
        req = r.requirements[0]
        assert len(req.accelerators) == 2
        assert all(isinstance(a, CudaSpecification) for a in req.accelerators)
        # Backwards compatibility: cuda_gpus property
        assert len(req.cuda_gpus) == 2


class TestCrossPlatformMatching:
    """Test cross-platform requirement matching"""

    def test_cross_platform_alternatives_cuda_host(self):
        """CUDA | MPS requirement should match CUDA host via first alternative"""
        host = HostSpecification(
            cpu=CPUSpecification(parse_size("64G"), 8),
            accelerators=[CudaSpecification(parse_size("24G"))],
        )
        req = parse("cuda(mem=8G) | mps(mem=8G)")
        match = req.match(host)
        assert match is not None
        # Should match the CUDA alternative
        assert isinstance(match.requirement.accelerators[0], CudaSpecification)

    def test_cross_platform_alternatives_mps_host(self):
        """CUDA | MPS requirement should match MPS host via second alternative"""
        host = HostSpecification(
            cpu=CPUSpecification(parse_size("32G"), 8),
            accelerators=[MPSSpecification(parse_size("32G"))],
        )
        req = parse("cuda(mem=8G) | mps(mem=8G)")
        match = req.match(host)
        assert match is not None
        # Should match the MPS alternative
        assert isinstance(match.requirement.accelerators[0], MPSSpecification)

    def test_generic_gpu_matches_any(self):
        """Generic gpu() should match any accelerator type"""
        cuda_host = HostSpecification(
            cpu=CPUSpecification(parse_size("64G"), 8),
            accelerators=[CudaSpecification(parse_size("24G"))],
        )
        mps_host = HostSpecification(
            cpu=CPUSpecification(parse_size("32G"), 8),
            accelerators=[MPSSpecification(parse_size("32G"))],
        )
        req = parse("gpu(mem=8G)")

        assert req.match(cuda_host) is not None
        assert req.match(mps_host) is not None


class TestUnifiedMemory:
    """Test unified memory handling (Apple Silicon MPS)"""

    def test_mps_unified_memory_flag(self):
        """MPS specification should indicate unified memory"""
        mps = MPSSpecification(memory=parse_size("32G"))
        assert mps.unified_memory is True

        cuda = CudaSpecification(memory=parse_size("24G"))
        assert cuda.unified_memory is False

        generic = AcceleratorSpecification(memory=parse_size("16G"))
        assert generic.unified_memory is False

    def test_unified_memory_host_detection(self):
        """Can detect if host uses unified memory"""
        mps_host = HostSpecification(
            cpu=CPUSpecification(parse_size("32G"), 8),
            accelerators=[MPSSpecification(parse_size("32G"))],
        )
        cuda_host = HostSpecification(
            cpu=CPUSpecification(parse_size("64G"), 8),
            accelerators=[CudaSpecification(parse_size("24G"))],
        )

        # Check if any accelerator has unified memory
        has_unified_mps = any(a.unified_memory for a in mps_host.accelerators)
        has_unified_cuda = any(a.unified_memory for a in cuda_host.accelerators)

        assert has_unified_mps is True
        assert has_unified_cuda is False

    def test_unified_memory_combined_limit(self):
        """MPS: CPU + GPU memory must not exceed total system memory"""
        # Apple Silicon with 32GB unified memory
        mps_host = HostSpecification(
            cpu=CPUSpecification(parse_size("32G"), 8),
            accelerators=[MPSSpecification(parse_size("32G"))],
        )

        # Request 16G CPU + 8G GPU = 24G total - should pass
        req_ok = mps_gpu(mem="8G") & cpu(mem="16G")
        assert req_ok.match(mps_host) is not None

        # Request 20G CPU + 16G GPU = 36G total - should fail (exceeds 32G)
        req_exceed = mps_gpu(mem="16G") & cpu(mem="20G")
        assert req_exceed.match(mps_host) is None

        # Request 16G CPU + 16G GPU = 32G exactly - should pass
        req_exact = mps_gpu(mem="16G") & cpu(mem="16G")
        assert req_exact.match(mps_host) is not None

    def test_unified_memory_not_applied_to_cuda(self):
        """CUDA: CPU + GPU memory are independent (no combined limit)"""
        # Host with 32G CPU RAM and 24G dedicated GPU memory
        cuda_host = HostSpecification(
            cpu=CPUSpecification(parse_size("32G"), 8),
            accelerators=[CudaSpecification(parse_size("24G"))],
        )

        # Request 30G CPU + 20G GPU - should pass (separate memory pools)
        req = cuda_gpu(mem="20G") & cpu(mem="30G")
        assert req.match(cuda_host) is not None
