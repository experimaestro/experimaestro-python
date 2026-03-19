"""Tests for auto-serialization of experiment configs at finalize time."""

import json

import pytest

from experimaestro import Config, Param, Task
from experimaestro.tests.utils import TemporaryExperiment, TemporaryDirectory


pytestmark = pytest.mark.tasks


class SharedConfig(Config):
    value: Param[str]


class TaskX(Task):
    shared: Param[SharedConfig]
    x: Param[int]

    def execute(self):
        pass


class TaskY(Task):
    shared: Param[SharedConfig]
    y: Param[int]

    def execute(self):
        pass


def test_configs_json_created():
    """Test that configs.json is created in the run dir after experiment finalize."""
    with TemporaryDirectory(prefix="xpm") as workdir:
        with TemporaryExperiment("test-configs", workdir=workdir) as xp:
            shared = SharedConfig.C(value="hello")
            TaskX.C(shared=shared, x=1).tag("model", "A").submit()
            TaskY.C(shared=shared, y=2).tag("model", "B").submit()

            run_dir = xp.workdir

        # After experiment exit, configs.json should exist
        configs_path = run_dir / "configs.json"
        assert configs_path.exists(), f"configs.json not created at {configs_path}"

        # Should be valid JSON with expected structure
        with configs_path.open() as f:
            data = json.load(f)

        assert "objects" in data
        assert "data" in data
        assert "tags" in data

        # Should have tags for both jobs
        assert len(data["tags"]) == 2
        tag_values = {v["model"] for v in data["tags"].values()}
        assert tag_values == {"A", "B"}


def test_load_configs_shared_references():
    """Test that loading configs preserves shared object references."""
    from experimaestro.scheduler.workspace_state_provider import (
        WorkspaceStateProvider,
    )

    with TemporaryDirectory(prefix="xpm") as workdir:
        with TemporaryExperiment("test-shared", workdir=workdir):
            shared = SharedConfig.C(value="shared-val")
            TaskX.C(shared=shared, x=10).tag("run", "first").submit()
            TaskY.C(shared=shared, y=20).tag("run", "second").submit()

        # Load configs via WorkspaceStateProvider
        provider = WorkspaceStateProvider(workdir, no_cleanup=True)
        try:
            configs = provider.load_configs("test-shared")

            # Should have 2 configs
            assert len(configs) == 2

            # Get the two configs
            config_list = list(configs.values())
            shared_refs = [getattr(c, "shared", None) for c in config_list]

            # Both should reference the SAME SharedConfig instance
            assert shared_refs[0] is not None
            assert shared_refs[1] is not None
            assert shared_refs[0] is shared_refs[1], (
                "Shared config references should be the same object"
            )

            # Check value is correct
            assert shared_refs[0].value == "shared-val"

            # Check tags were restored
            for job_id, config in configs.items():
                tags = config.tags()
                assert "run" in tags, f"Tag 'run' not found on config {job_id}"
                assert tags["run"] in ("first", "second")
        finally:
            provider.close()


def test_load_configs_tags_match_tags_map():
    """Test that tags from load_configs match get_tags_map for DataFrame building."""
    from experimaestro.scheduler.workspace_state_provider import (
        WorkspaceStateProvider,
    )

    with TemporaryDirectory(prefix="xpm") as workdir:
        with TemporaryExperiment("test-tags-map", workdir=workdir):
            shared = SharedConfig.C(value="v")
            TaskX.C(shared=shared, x=1).tag("lr", "0.01").tag("model", "big").submit()
            TaskX.C(shared=shared, x=2).tag("lr", "0.001").tag(
                "model", "small"
            ).submit()
            TaskY.C(shared=shared, y=3).tag("lr", "0.01").tag("model", "big").submit()

        provider = WorkspaceStateProvider(workdir, no_cleanup=True)
        try:
            configs = provider.load_configs("test-tags-map")
            tags_map = provider.get_tags_map("test-tags-map")

            assert len(configs) == 3

            # Every job_id in configs should have tags in tags_map
            for job_id in configs:
                assert job_id in tags_map, f"Job {job_id} missing from tags_map"

            # Tags from load_configs (restored on config) should match tags_map
            for job_id, config in configs.items():
                config_tags = config.tags()
                map_tags = tags_map[job_id]
                assert config_tags == map_tags, (
                    f"Tags mismatch for {job_id}: "
                    f"config.tags()={config_tags} vs tags_map={map_tags}"
                )

            # Verify specific tag values are present
            all_lrs = {tags_map[jid]["lr"] for jid in tags_map}
            assert all_lrs == {"0.01", "0.001"}

            all_models = {tags_map[jid]["model"] for jid in tags_map}
            assert all_models == {"big", "small"}
        finally:
            provider.close()


def test_load_configs_standalone():
    """Test loading configs with standalone load_configs (no provider needed)."""
    from experimaestro import load_configs

    with TemporaryDirectory(prefix="xpm") as workdir:
        with TemporaryExperiment("test-standalone", workdir=workdir) as xp:
            shared = SharedConfig.C(value="standalone")
            TaskX.C(shared=shared, x=5).tag("group", "alpha").submit()
            TaskY.C(shared=shared, y=6).tag("group", "beta").submit()

            run_dir = xp.workdir

        # Load directly from run dir path
        configs = load_configs(run_dir)

        assert len(configs) == 2

        # Shared references preserved
        config_list = list(configs.values())
        shared_refs = [getattr(c, "shared", None) for c in config_list]
        assert shared_refs[0] is shared_refs[1]
        assert shared_refs[0].value == "standalone"

        # Tags restored
        tag_values = {config.tags()["group"] for config in configs.values()}
        assert tag_values == {"alpha", "beta"}

        # Also works pointing directly at configs.json
        configs2 = load_configs(run_dir / "configs.json")
        assert len(configs2) == 2


def test_load_configs_not_found():
    """Test that FileNotFoundError is raised when configs.json doesn't exist."""
    from experimaestro.scheduler.workspace_state_provider import (
        WorkspaceStateProvider,
    )

    with TemporaryDirectory(prefix="xpm") as workdir:
        # Create minimal experiment structure without configs.json
        exp_dir = workdir / "experiments" / "nonexistent" / "20260101_000000"
        exp_dir.mkdir(parents=True)

        provider = WorkspaceStateProvider(workdir, no_cleanup=True)
        try:
            # Create a current symlink
            current = workdir / "experiments" / "nonexistent" / "current"
            current.symlink_to(exp_dir)

            with pytest.raises(FileNotFoundError):
                provider.load_configs("nonexistent")
        finally:
            provider.close()
