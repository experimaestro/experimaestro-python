"""Tests for direct launcher generation (launchers.py and tokens.yaml).

These tests generate launcher code and execute it to verify the find_launcher()
function works correctly with various requirement configurations.
"""

from experimaestro.launchers.direct import (
    DirectLauncher,
    GPUInfo,
    LauncherConfig,
    SystemInfo,
    _generate_launchers_py,
    _generate_tokens_yaml,
)
from experimaestro.launcherfinder.specs import cpu, cuda_gpu, mps_gpu, gpu


def _exec_find_launcher(code: str):
    """Execute generated code and return the find_launcher function."""
    namespace = {}
    exec(code, namespace)
    return namespace["find_launcher"]


class TestGeneratedLauncherMPS:
    """Test generated launcher code for Apple Silicon MPS."""

    def _make_mps_launcher(self) -> str:
        """Generate MPS launcher code."""
        config = LauncherConfig(
            use_memory_tokens=False,  # Disable tokens for testing
            use_gpu_tokens=False,
            total_memory_tokens=128,
            total_gpu_tokens=0,
            memory_token_unit_mib=256,
            gpu_token_unit_mib=256,
            reserved_memory_gib=2.0,
            gpu_mode="memory",
            is_mps=True,
            mps_shared_memory=True,
        )
        sys_info = SystemInfo(
            total_memory_bytes=34 * (1024**3),  # 34 GiB total (32 available)
            cpu_count=10,
            gpus=[],
            is_apple_silicon=True,
        )
        return _generate_launchers_py(config, sys_info)

    def test_mps_matches_mps_requirement(self):
        """MPS host matches MPS GPU requirement."""
        find_launcher = _exec_find_launcher(self._make_mps_launcher())

        req = mps_gpu(mem="8G") & cpu(mem="4G")
        launcher = find_launcher(req)

        assert launcher is not None
        assert isinstance(launcher, DirectLauncher)

    def test_mps_matches_generic_gpu_requirement(self):
        """MPS host matches generic GPU requirement."""
        find_launcher = _exec_find_launcher(self._make_mps_launcher())

        req = gpu(mem="8G") & cpu(mem="4G")
        launcher = find_launcher(req)

        assert launcher is not None
        assert isinstance(launcher, DirectLauncher)

    def test_mps_rejects_cuda_requirement(self):
        """MPS host rejects CUDA-specific requirement."""
        find_launcher = _exec_find_launcher(self._make_mps_launcher())

        req = cuda_gpu(mem="8G") & cpu(mem="4G")
        launcher = find_launcher(req)

        assert launcher is None

    def test_mps_unified_memory_combined_limit(self):
        """MPS rejects when CPU + GPU memory exceeds unified memory."""
        find_launcher = _exec_find_launcher(self._make_mps_launcher())

        # Request 20G CPU + 16G GPU = 36G total, but only 32G available
        req = mps_gpu(mem="16G") & cpu(mem="20G")
        launcher = find_launcher(req)

        assert launcher is None

    def test_mps_unified_memory_within_limit(self):
        """MPS accepts when CPU + GPU memory within unified memory."""
        find_launcher = _exec_find_launcher(self._make_mps_launcher())

        # Request 16G CPU + 8G GPU = 24G total, 32G available
        req = mps_gpu(mem="8G") & cpu(mem="16G")
        launcher = find_launcher(req)

        assert launcher is not None

    def test_mps_cpu_only_requirement(self):
        """MPS host matches CPU-only requirement."""
        find_launcher = _exec_find_launcher(self._make_mps_launcher())

        req = cpu(mem="8G")
        launcher = find_launcher(req)

        assert launcher is not None

    def test_mps_rejects_excessive_cpu_memory(self):
        """MPS rejects CPU memory exceeding available."""
        find_launcher = _exec_find_launcher(self._make_mps_launcher())

        # Request 40G CPU but only 32G available
        req = cpu(mem="40G")
        launcher = find_launcher(req)

        assert launcher is None


class TestGeneratedLauncherCUDA:
    """Test generated launcher code for NVIDIA CUDA GPUs."""

    def _make_cuda_launcher(self, gpu_mode: str = "exclusive") -> str:
        """Generate CUDA launcher code."""
        config = LauncherConfig(
            use_memory_tokens=False,  # Disable tokens for testing
            use_gpu_tokens=False,
            total_memory_tokens=256,
            total_gpu_tokens=2 if gpu_mode == "exclusive" else 96,
            memory_token_unit_mib=256,
            gpu_token_unit_mib=256,
            reserved_memory_gib=4.0,
            gpu_mode=gpu_mode,
            is_mps=False,
            mps_shared_memory=False,
        )
        sys_info = SystemInfo(
            total_memory_bytes=68 * (1024**3),  # 68 GiB total (64 available)
            cpu_count=32,
            gpus=[
                GPUInfo(name="NVIDIA RTX 4090", memory_bytes=24 * (1024**3)),
                GPUInfo(name="NVIDIA RTX 4090", memory_bytes=24 * (1024**3)),
            ],
            is_apple_silicon=False,
        )
        return _generate_launchers_py(config, sys_info)

    def test_cuda_matches_cuda_requirement(self):
        """CUDA host matches CUDA GPU requirement."""
        find_launcher = _exec_find_launcher(self._make_cuda_launcher())

        req = cuda_gpu(mem="16G") & cpu(mem="8G")
        launcher = find_launcher(req)

        assert launcher is not None
        assert isinstance(launcher, DirectLauncher)

    def test_cuda_matches_generic_gpu_requirement(self):
        """CUDA host matches generic GPU requirement."""
        find_launcher = _exec_find_launcher(self._make_cuda_launcher())

        req = gpu(mem="16G") & cpu(mem="8G")
        launcher = find_launcher(req)

        assert launcher is not None

    def test_cuda_rejects_mps_requirement(self):
        """CUDA host rejects MPS-specific requirement."""
        find_launcher = _exec_find_launcher(self._make_cuda_launcher())

        req = mps_gpu(mem="16G") & cpu(mem="8G")
        launcher = find_launcher(req)

        assert launcher is None

    def test_cuda_matches_multiple_gpus(self):
        """CUDA host matches multi-GPU requirement."""
        find_launcher = _exec_find_launcher(self._make_cuda_launcher())

        req = cuda_gpu(mem="16G") * 2 & cpu(mem="8G")
        launcher = find_launcher(req)

        assert launcher is not None

    def test_cuda_rejects_too_many_gpus(self):
        """CUDA host rejects requirement for more GPUs than available."""
        find_launcher = _exec_find_launcher(self._make_cuda_launcher())

        # Request 3 GPUs but only 2 available
        req = cuda_gpu(mem="16G") * 3 & cpu(mem="8G")
        launcher = find_launcher(req)

        assert launcher is None

    def test_cuda_rejects_excessive_gpu_memory(self):
        """CUDA host rejects GPU memory exceeding available."""
        find_launcher = _exec_find_launcher(self._make_cuda_launcher())

        # Request 32G GPU but only 24G available per GPU
        req = cuda_gpu(mem="32G") & cpu(mem="8G")
        launcher = find_launcher(req)

        assert launcher is None

    def test_cuda_independent_memory_pools(self):
        """CUDA CPU and GPU memory are independent (not unified)."""
        find_launcher = _exec_find_launcher(self._make_cuda_launcher())

        # Request 50G CPU + 20G GPU - should work because separate pools
        # (64G CPU available, 24G GPU available)
        req = cuda_gpu(mem="20G") & cpu(mem="50G")
        launcher = find_launcher(req)

        assert launcher is not None

    def test_cuda_exclusive_mode_counts_gpus(self):
        """Exclusive GPU mode counts GPUs, not memory."""
        find_launcher = _exec_find_launcher(self._make_cuda_launcher("exclusive"))

        # 2 GPUs available, request 2
        req = cuda_gpu(mem="8G") * 2 & cpu(mem="4G")
        launcher = find_launcher(req)
        assert launcher is not None

        # Request 3 GPUs - should fail
        req = cuda_gpu(mem="8G") * 3 & cpu(mem="4G")
        launcher = find_launcher(req)
        assert launcher is None


class TestGeneratedLauncherNoGPU:
    """Test generated launcher code for CPU-only systems."""

    def _make_cpu_only_launcher(self) -> str:
        """Generate CPU-only launcher code."""
        config = LauncherConfig(
            use_memory_tokens=False,  # Disable tokens for testing
            use_gpu_tokens=False,
            total_memory_tokens=64,
            total_gpu_tokens=0,
            memory_token_unit_mib=256,
            gpu_token_unit_mib=256,
            reserved_memory_gib=2.0,
            gpu_mode="memory",
            is_mps=False,
            mps_shared_memory=False,
        )
        sys_info = SystemInfo(
            total_memory_bytes=18 * (1024**3),  # 18 GiB total (16 available)
            cpu_count=8,
            gpus=[],
            is_apple_silicon=False,
        )
        return _generate_launchers_py(config, sys_info)

    def test_cpu_only_matches_cpu_requirement(self):
        """CPU-only host matches CPU requirement."""
        find_launcher = _exec_find_launcher(self._make_cpu_only_launcher())

        req = cpu(mem="8G")
        launcher = find_launcher(req)

        assert launcher is not None
        assert isinstance(launcher, DirectLauncher)

    def test_cpu_only_rejects_gpu_requirement(self):
        """CPU-only host rejects any GPU requirement."""
        find_launcher = _exec_find_launcher(self._make_cpu_only_launcher())

        req = cuda_gpu(mem="8G") & cpu(mem="4G")
        launcher = find_launcher(req)

        assert launcher is None

    def test_cpu_only_rejects_generic_gpu(self):
        """CPU-only host rejects generic GPU requirement."""
        find_launcher = _exec_find_launcher(self._make_cpu_only_launcher())

        req = gpu(mem="8G") & cpu(mem="4G")
        launcher = find_launcher(req)

        assert launcher is None

    def test_cpu_only_rejects_excessive_memory(self):
        """CPU-only host rejects memory exceeding available."""
        find_launcher = _exec_find_launcher(self._make_cpu_only_launcher())

        req = cpu(mem="20G")  # Only 16G available
        launcher = find_launcher(req)

        assert launcher is None


class TestGeneratedLauncherAlternatives:
    """Test generated launcher with requirement alternatives."""

    def _make_mps_launcher(self) -> str:
        config = LauncherConfig(
            use_memory_tokens=False,  # Disable tokens for testing
            use_gpu_tokens=False,
            total_memory_tokens=128,
            total_gpu_tokens=0,
            memory_token_unit_mib=256,
            gpu_token_unit_mib=256,
            reserved_memory_gib=2.0,
            gpu_mode="memory",
            is_mps=True,
            mps_shared_memory=True,
        )
        sys_info = SystemInfo(
            total_memory_bytes=34 * (1024**3),
            cpu_count=10,
            gpus=[],
            is_apple_silicon=True,
        )
        return _generate_launchers_py(config, sys_info)

    def _make_cuda_launcher(self) -> str:
        config = LauncherConfig(
            use_memory_tokens=False,  # Disable tokens for testing
            use_gpu_tokens=False,
            total_memory_tokens=256,
            total_gpu_tokens=2,
            memory_token_unit_mib=256,
            gpu_token_unit_mib=256,
            reserved_memory_gib=4.0,
            gpu_mode="exclusive",
            is_mps=False,
            mps_shared_memory=False,
        )
        sys_info = SystemInfo(
            total_memory_bytes=68 * (1024**3),
            cpu_count=32,
            gpus=[GPUInfo(name="RTX 4090", memory_bytes=24 * (1024**3))],
            is_apple_silicon=False,
        )
        return _generate_launchers_py(config, sys_info)

    def test_cross_platform_requirement_matches_mps(self):
        """Cross-platform requirement (CUDA | MPS) matches MPS host."""
        find_launcher = _exec_find_launcher(self._make_mps_launcher())

        # Cross-platform: prefer CUDA but accept MPS
        req = (cuda_gpu(mem="8G") & cpu(mem="4G")) | (mps_gpu(mem="8G") & cpu(mem="4G"))
        launcher = find_launcher(req)

        assert launcher is not None

    def test_cross_platform_requirement_matches_cuda(self):
        """Cross-platform requirement (CUDA | MPS) matches CUDA host."""
        find_launcher = _exec_find_launcher(self._make_cuda_launcher())

        # Cross-platform: prefer CUDA but accept MPS
        req = (cuda_gpu(mem="8G") & cpu(mem="4G")) | (mps_gpu(mem="8G") & cpu(mem="4G"))
        launcher = find_launcher(req)

        assert launcher is not None


class TestGenerateTokensYaml:
    """Tests for tokens.yaml generation."""

    def test_memory_tokens_structure(self):
        """Generated YAML has correct memory token structure."""
        config = LauncherConfig(
            use_memory_tokens=True,
            use_gpu_tokens=False,
            total_memory_tokens=128,
            total_gpu_tokens=0,
            memory_token_unit_mib=256,
            gpu_token_unit_mib=256,
            reserved_memory_gib=2.0,
            gpu_mode="memory",
            is_mps=False,
            mps_shared_memory=False,
        )

        yaml = _generate_tokens_yaml(config)

        assert "countertoken:" in yaml
        assert "memory:" in yaml
        assert "tokens: 128" in yaml

    def test_gpu_tokens_exclusive_mode(self):
        """Generated YAML includes GPU tokens in exclusive mode."""
        config = LauncherConfig(
            use_memory_tokens=True,
            use_gpu_tokens=True,
            total_memory_tokens=256,
            total_gpu_tokens=4,
            memory_token_unit_mib=256,
            gpu_token_unit_mib=256,
            reserved_memory_gib=4.0,
            gpu_mode="exclusive",
            is_mps=False,
            mps_shared_memory=False,
        )

        yaml = _generate_tokens_yaml(config)

        assert "gpu:" in yaml
        assert "tokens: 4" in yaml
        assert "exclusive" in yaml.lower()

    def test_no_gpu_section_when_zero_tokens(self):
        """No GPU section when total_gpu_tokens is zero."""
        config = LauncherConfig(
            use_memory_tokens=True,
            use_gpu_tokens=True,
            total_memory_tokens=128,
            total_gpu_tokens=0,
            memory_token_unit_mib=256,
            gpu_token_unit_mib=256,
            reserved_memory_gib=2.0,
            gpu_mode="memory",
            is_mps=True,
            mps_shared_memory=True,
        )

        yaml = _generate_tokens_yaml(config)

        assert "gpu:" not in yaml
