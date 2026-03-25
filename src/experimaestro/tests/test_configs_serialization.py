"""Tests for auto-serialization of experiment configs via objects.jsonl."""

import json

import pytest

from experimaestro import Config, Param, Task, load_xp_info
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


def test_objects_jsonl_created():
    """Test that objects.jsonl is created in the run dir after experiment."""
    with TemporaryDirectory(prefix="xpm") as workdir:
        with TemporaryExperiment("test-configs", workdir=workdir) as xp:
            shared = SharedConfig.C(value="hello")
            TaskX.C(shared=shared, x=1).tag("model", "A").submit()
            TaskY.C(shared=shared, y=2).tag("model", "B").submit()

            run_dir = xp.workdir

        # After experiment exit, objects.jsonl should exist
        objects_path = run_dir / "objects.jsonl"
        assert objects_path.exists(), f"objects.jsonl not created at {objects_path}"

        # Each line is a single serialized object
        lines = [line for line in objects_path.read_text().strip().split("\n") if line]
        assert len(lines) >= 2, "Expected at least 2 serialized objects"

        # Each line should be valid JSON with object structure
        for line in lines:
            obj = json.loads(line)
            assert "id" in obj  # Python id()
            assert "type" in obj  # class qualname


def test_load_xp_info_shared_references():
    """Test that loading configs preserves shared object references."""
    with TemporaryDirectory(prefix="xpm") as workdir:
        with TemporaryExperiment("test-shared", workdir=workdir) as xp:
            shared = SharedConfig.C(value="shared-val")
            TaskX.C(shared=shared, x=10).tag("run", "first").submit()
            TaskY.C(shared=shared, y=20).tag("run", "second").submit()

            run_dir = xp.workdir

        info = load_xp_info(run_dir)

        # Should have 2 job configs
        assert len(info.jobs) == 2

        # Get the two configs
        config_list = list(info.jobs.values())
        shared_refs = [getattr(c, "shared", None) for c in config_list]

        # Both should reference the SAME SharedConfig instance
        assert shared_refs[0] is not None
        assert shared_refs[1] is not None
        assert shared_refs[0] is shared_refs[1], (
            "Shared config references should be the same object"
        )

        # Check value is correct
        assert shared_refs[0].value == "shared-val"


def test_load_xp_info_via_provider():
    """Test loading via WorkspaceStateProvider."""
    from experimaestro.scheduler.workspace_state_provider import (
        WorkspaceStateProvider,
    )

    with TemporaryDirectory(prefix="xpm") as workdir:
        with TemporaryExperiment("test-provider", workdir=workdir):
            shared = SharedConfig.C(value="v")
            TaskX.C(shared=shared, x=1).tag("lr", "0.01").tag("model", "big").submit()
            TaskX.C(shared=shared, x=2).tag("lr", "0.001").tag(
                "model", "small"
            ).submit()
            TaskY.C(shared=shared, y=3).tag("lr", "0.01").tag("model", "big").submit()

        provider = WorkspaceStateProvider(workdir, no_cleanup=True)
        try:
            info = provider.load_xp_info("test-provider")
            tags_map = provider.get_tags_map("test-provider")

            assert len(info.jobs) == 3

            # Every job_id in jobs should have tags in tags_map
            for job_id in info.jobs:
                assert job_id in tags_map, f"Job {job_id} missing from tags_map"

            # Verify specific tag values are present
            all_lrs = {tags_map[jid]["lr"] for jid in tags_map}
            assert all_lrs == {"0.01", "0.001"}

            all_models = {tags_map[jid]["model"] for jid in tags_map}
            assert all_models == {"big", "small"}
        finally:
            provider.close()


def test_load_xp_info_standalone():
    """Test loading with standalone load_xp_info (no provider needed)."""
    with TemporaryDirectory(prefix="xpm") as workdir:
        with TemporaryExperiment("test-standalone", workdir=workdir) as xp:
            shared = SharedConfig.C(value="standalone")
            TaskX.C(shared=shared, x=5).tag("group", "alpha").submit()
            TaskY.C(shared=shared, y=6).tag("group", "beta").submit()

            run_dir = xp.workdir

        # Load directly from run dir path
        info = load_xp_info(run_dir)

        assert len(info.jobs) == 2

        # Shared references preserved
        config_list = list(info.jobs.values())
        shared_refs = [getattr(c, "shared", None) for c in config_list]
        assert shared_refs[0] is shared_refs[1]
        assert shared_refs[0].value == "standalone"


def test_load_xp_info_not_found():
    """Test that FileNotFoundError is raised when no serialization files exist."""
    from experimaestro.scheduler.workspace_state_provider import (
        WorkspaceStateProvider,
    )

    with TemporaryDirectory(prefix="xpm") as workdir:
        # Create minimal experiment structure without objects.jsonl
        exp_dir = workdir / "experiments" / "nonexistent" / "20260101_000000"
        exp_dir.mkdir(parents=True)

        provider = WorkspaceStateProvider(workdir, no_cleanup=True)
        try:
            # Create a current symlink
            current = workdir / "experiments" / "nonexistent" / "current"
            current.symlink_to(exp_dir)

            with pytest.raises(FileNotFoundError):
                provider.load_xp_info("nonexistent")
        finally:
            provider.close()


def test_load_xp_info_backward_compat_configs_json():
    """Test that load_xp_info falls back to configs.json for old experiments."""
    import tempfile
    from pathlib import Path
    from experimaestro.core.serialization import state_dict
    from experimaestro.core.context import SerializationContext

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Create old-format configs.json
        config = SharedConfig.C(value="old-format")
        config.seal()

        context = SerializationContext(save_directory=None)
        data = state_dict(context, {"job1": config})
        data["tags"] = {}

        with (run_dir / "configs.json").open("w") as f:
            json.dump(data, f)

        info = load_xp_info(run_dir)
        assert len(info.jobs) == 1
        assert "job1" in info.jobs
        assert len(info.actions) == 0
