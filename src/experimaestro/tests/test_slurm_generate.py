"""Tests for SLURM CLI parsing functions."""

from experimaestro.launchers.slurm.cli import (
    parse_gres,
    parse_time_to_seconds,
)


# =============================================================================
# Unit tests for parsing functions
# =============================================================================


class TestParseTimeToSeconds:
    def test_days_hours_minutes_seconds(self):
        assert parse_time_to_seconds("4-04:00:00") == 4 * 86400 + 4 * 3600

    def test_hours_minutes_seconds(self):
        assert parse_time_to_seconds("20:00:00") == 20 * 3600

    def test_minutes_seconds(self):
        assert parse_time_to_seconds("30:00") == 30 * 60

    def test_seconds_only(self):
        assert parse_time_to_seconds("120") == 120

    def test_unlimited(self):
        assert parse_time_to_seconds("UNLIMITED") is None

    def test_none(self):
        assert parse_time_to_seconds(None) is None

    def test_two_hours(self):
        assert parse_time_to_seconds("02:00:00") == 2 * 3600


class TestParseGres:
    def test_gpu_count_only(self):
        count, gpu_type = parse_gres("gpu:4(S:0-1)")
        assert count == 4
        assert gpu_type is None

    def test_gpu_with_type(self):
        count, gpu_type = parse_gres("gpu:tesla:4(S:0-1)")
        assert count == 4
        assert gpu_type == "tesla"

    def test_gpu_8(self):
        count, gpu_type = parse_gres("gpu:8(S:0-1)")
        assert count == 8
        assert gpu_type is None

    def test_null(self):
        count, gpu_type = parse_gres("(null)")
        assert count == 0
        assert gpu_type is None

    def test_empty(self):
        count, gpu_type = parse_gres("")
        assert count == 0
        assert gpu_type is None


class TestSlurmGenerateCLI:
    """Test non-interactive SLURM launcher generation from cached cluster data."""

    def test_slurm_generate_yes_from_cache(self, tmp_path):
        import json
        from click.testing import CliRunner
        from experimaestro.launchers.slurm.cli import cli as slurm_cli
        from experimaestro.launchers.slurm import SlurmLauncher
        from experimaestro.launcherfinder.parser import parse

        cache_data = {
            "sinfo": {
                "meta": {"slurm": {"cluster": "test_cluster"}},
                "nodes": [],
                "sinfo": [],
            },
            "partitions": {
                "partitions": [
                    {
                        "name": "gpu_a100",
                        "partition": {"state": ["UP"]},
                        "cpus": {"total": 64},
                        "nodes": {"configured": "node[01-02]", "total": 2},
                        "tres": {
                            "configured": "cpu=64,mem=256G,node=1,gres/gpu:a100=4"
                        },
                        "maximums": {"time": {"number": 120, "infinite": False}},
                    },
                    {
                        "name": "cpu_std",
                        "partition": {"state": ["UP"]},
                        "cpus": {"total": 32},
                        "nodes": {"configured": "node[03-04]", "total": 2},
                        "tres": {"configured": "cpu=32,mem=128G,node=1"},
                        "maximums": {"time": {"number": 60, "infinite": False}},
                    },
                ]
            },
            "qos": {"qos": []},
            "associations": {
                "associations": [
                    {"account": "myproj", "is_default": True, "qos": ["normal"]}
                ]
            },
            "config": {},
        }
        cache_file = tmp_path / "slurm_cache.json"
        cache_file.write_text(json.dumps(cache_data))
        config_file = tmp_path / "slurm.yaml"
        out_file = tmp_path / "launchers.py"

        runner = CliRunner()
        result = runner.invoke(
            slurm_cli,
            [
                "generate",
                "--cache-file",
                str(cache_file),
                "--use-cache",
                "-c",
                str(config_file),
                "-o",
                str(out_file),
                "-y",
            ],
        )

        assert result.exit_code == 0, result.output
        assert config_file.is_file()
        assert out_file.is_file()

        # Execute the generated launchers.py and verify find_launcher works
        ns = {}
        exec(out_file.read_text(), ns)
        find_launcher_fn = ns["find_launcher"]

        req = parse("duration=1h & gpu(mem=20G)*2 & cpu(cores=8)").requirements[0]
        launcher = find_launcher_fn(req)
        assert isinstance(launcher, SlurmLauncher)
        assert launcher.options.partition == "gpu_a100"
        assert launcher.options.gpus == 2
        assert launcher.options.account == "myproj"
        assert launcher.options.time == "1:0:0"
