"""Tests for job/workspace-relative path encoding in params.json.

These checks cover the path encoding introduced for issue #228: paths inside
the current job are stored job-relative, paths under the workspace but in
another job are stored workspace-relative, and external paths are kept
absolute. This avoids path rewriting when copying a job to a new location.
"""

import json
from pathlib import Path

import pytest

import experimaestro.taskglobals as taskglobals
from experimaestro import Config, Param, from_state_dict, state_dict
from experimaestro.core.context import SerializationContext
from experimaestro.core.objects import ConfigInformation

pytestmark = [
    pytest.mark.serialization,
    pytest.mark.dependency(depends=["mod_identifier"], scope="session"),
]


class ConfigWithPath(Config):
    p: Param[Path]


# --- _outputjsonvalue encoding ---


def _encode_path(value: Path, *, job_path=None, workspace_path=None):
    context = SerializationContext()
    context.job_path = job_path
    context.workspace_path = workspace_path
    return ConfigInformation._outputjsonvalue(value, context)


def test_encode_job_relative(tmp_path):
    job = tmp_path / "jobs" / "task" / "abc"
    ws = tmp_path
    encoded = _encode_path(job / "out" / "file.txt", job_path=job, workspace_path=ws)
    assert encoded == {"type": "path", "value": "out/file.txt", "base": "job"}


def test_encode_workspace_relative(tmp_path):
    job = tmp_path / "jobs" / "task" / "abc"
    ws = tmp_path
    other_job_file = ws / "jobs" / "other_task" / "def" / "model.pt"
    encoded = _encode_path(other_job_file, job_path=job, workspace_path=ws)
    assert encoded == {
        "type": "path",
        "value": "jobs/other_task/def/model.pt",
        "base": "workspace",
    }


def test_encode_absolute_when_outside(tmp_path):
    job = tmp_path / "jobs" / "task" / "abc"
    ws = tmp_path
    external = Path("/etc/passwd")
    encoded = _encode_path(external, job_path=job, workspace_path=ws)
    assert encoded == {"type": "path", "value": "/etc/passwd"}


def test_encode_absolute_when_no_anchors():
    """Plain SerializationContext (no workspace/job) keeps absolute paths."""
    encoded = _encode_path(Path("/some/abs/path"))
    assert encoded == {"type": "path", "value": "/some/abs/path"}


# --- Deserialization via taskglobals ---


@pytest.fixture
def restore_env():
    env = taskglobals.Env.instance()
    prev_ws, prev_task = env.wspath, env.taskpath
    yield env
    env.wspath, env.taskpath = prev_ws, prev_task


def test_decode_job_relative(tmp_path, restore_env):
    restore_env.taskpath = tmp_path / "jobs" / "task" / "abc"
    restore_env.wspath = tmp_path
    decoded = ConfigInformation._objectFromParameters(
        {"type": "path", "value": "out/file.txt", "base": "job"}, {}
    )
    assert decoded == tmp_path / "jobs" / "task" / "abc" / "out" / "file.txt"


def test_decode_workspace_relative(tmp_path, restore_env):
    restore_env.taskpath = tmp_path / "jobs" / "task" / "abc"
    restore_env.wspath = tmp_path
    decoded = ConfigInformation._objectFromParameters(
        {"type": "path", "value": "jobs/other/def/model.pt", "base": "workspace"}, {}
    )
    assert decoded == tmp_path / "jobs" / "other" / "def" / "model.pt"


def test_decode_absolute_backward_compat(restore_env):
    """Old-format entries without a "base" key keep working as absolute paths."""
    decoded = ConfigInformation._objectFromParameters(
        {"type": "path", "value": "/some/abs/file"}, {}
    )
    assert decoded == Path("/some/abs/file")


def test_decode_job_relative_without_taskpath_errors(restore_env):
    restore_env.taskpath = None
    with pytest.raises(RuntimeError, match="taskpath"):
        ConfigInformation._objectFromParameters(
            {"type": "path", "value": "out/file.txt", "base": "job"}, {}
        )


def test_decode_workspace_relative_without_wspath_errors(restore_env):
    restore_env.wspath = None
    with pytest.raises(RuntimeError, match="wspath"):
        ConfigInformation._objectFromParameters(
            {"type": "path", "value": "jobs/o/d/f", "base": "workspace"}, {}
        )


# --- End-to-end round-trip ---


def test_roundtrip_job_relative(tmp_path, restore_env):
    job = tmp_path / "jobs" / "task" / "abc"
    job.mkdir(parents=True)
    ws = tmp_path

    ctx = SerializationContext()
    ctx.job_path = job
    ctx.workspace_path = ws

    config = ConfigWithPath.C(p=job / "out" / "result.txt")
    config.__xpm__.seal(ctx)
    data = state_dict(ctx, config)

    # Field should be encoded job-relative
    fields = data["objects"][0]["fields"]
    assert fields["p"]["base"] == "job"
    assert fields["p"]["value"] == "out/result.txt"

    # Round-trip with taskglobals set should yield the absolute path back
    restore_env.taskpath = job
    restore_env.wspath = ws
    reloaded = from_state_dict(data)
    assert reloaded.p == job / "out" / "result.txt"


def test_roundtrip_workspace_relative(tmp_path, restore_env):
    job = tmp_path / "jobs" / "task" / "abc"
    job.mkdir(parents=True)
    ws = tmp_path

    ctx = SerializationContext()
    ctx.job_path = job
    ctx.workspace_path = ws

    dep_file = ws / "jobs" / "other" / "def" / "model.pt"
    config = ConfigWithPath.C(p=dep_file)
    config.__xpm__.seal(ctx)
    data = state_dict(ctx, config)

    fields = data["objects"][0]["fields"]
    assert fields["p"]["base"] == "workspace"
    assert fields["p"]["value"] == "jobs/other/def/model.pt"

    restore_env.taskpath = job
    restore_env.wspath = ws
    reloaded = from_state_dict(data)
    assert reloaded.p == dep_file


def test_check_params_version_accepts_supported():
    """Current and older versions load without raising."""
    for version in (2, ConfigInformation.PARAMS_JSON_VERSION):
        ConfigInformation.check_params_version({"version": version}, source="test")


def test_check_params_version_rejects_newer():
    """A params.json from a newer experimaestro must fail with a clear error."""
    too_new = ConfigInformation.PARAMS_JSON_VERSION + 1
    with pytest.raises(RuntimeError, match=f"version {too_new}"):
        ConfigInformation.check_params_version({"version": too_new}, source="test")


def test_check_params_version_missing_field_accepts():
    """An ancient params.json without a version field is treated as old (v0)."""
    ConfigInformation.check_params_version({}, source="test")


def test_outputjson_writes_current_version(tmp_path):
    """outputjson must stamp params.json with PARAMS_JSON_VERSION."""
    import io
    import dataclasses
    from types import SimpleNamespace

    @dataclasses.dataclass
    class _Carbon:
        pass

    job = tmp_path / "jobs" / "task" / "abc"
    job.mkdir(parents=True)
    ctx = SerializationContext()
    ctx.job_path = job
    ctx.workspace_path = tmp_path
    ctx.workspace = SimpleNamespace(
        path=tmp_path, settings=SimpleNamespace(carbon=_Carbon())
    )

    config = ConfigWithPath.C(p=job / "out.txt")
    config.__xpm__.seal(ctx)
    buf = io.StringIO()
    config.__xpm__.outputjson(buf, ctx)
    payload = json.loads(buf.getvalue())
    assert payload["version"] == ConfigInformation.PARAMS_JSON_VERSION


def test_roundtrip_after_workspace_move(tmp_path, restore_env):
    """Copying a job to a new workspace must not require path rewriting."""
    src_ws = tmp_path / "src"
    src_job = src_ws / "jobs" / "task" / "abc"
    src_job.mkdir(parents=True)

    ctx = SerializationContext()
    ctx.job_path = src_job
    ctx.workspace_path = src_ws

    config = ConfigWithPath.C(p=src_job / "out" / "result.txt")
    config.__xpm__.seal(ctx)
    data = state_dict(ctx, config)

    # Simulate copying the job to a fresh workspace at a new location
    dst_ws = tmp_path / "dst"
    dst_job = dst_ws / "jobs" / "task" / "abc"
    dst_job.mkdir(parents=True)

    # Write/read the JSON to mimic params.json on disk
    raw = json.loads(json.dumps(data))

    restore_env.taskpath = dst_job
    restore_env.wspath = dst_ws
    reloaded = from_state_dict(raw)
    assert reloaded.p == dst_job / "out" / "result.txt"
