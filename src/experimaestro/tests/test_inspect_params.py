"""Automated tests for the standalone Params Inspector tool."""

from dataclasses import dataclass
from pathlib import Path
import json
from unittest.mock import patch

import pytest

from experimaestro import (
    Config,
    LightweightTask,
    Meta,
    Param,
    PathGenerator,
    Task,
    field,
)
from experimaestro.scheduler.workspace import RunMode
from experimaestro.settings import Settings, SSHSettings, WorkspaceSettings
from experimaestro.tests.utils import TemporaryExperiment
from experimaestro.tools.inspect_params import (
    NodeKind,
    NodeDetailPane,
    ParamsGraph,
    ParamsInspectorApp,
    ParamsTree,
    ResolvedPath,
    clean_job_target,
    fetch_remote_workspace_file,
    find_local_workspace_file,
    find_params_file,
    find_workspace_file,
    resolve_job_file_pattern,
    resolve_workspace,
)

# --- A sample workspace: a learner and its evaluations


class Dataset(Config):
    name: Param[str]


class CNN(Config):
    hidden_dim: Param[int]
    kernel_size: Param[int]


class Learn(Task):
    data: Param[Dataset]
    model: Param[CNN]
    batch_size: Param[int]
    lr: Param[float]
    run_path: Meta[Path] = field(default_factory=PathGenerator("runs"))

    def execute(self):
        pass


class ParameterLoader(LightweightTask):
    model: Param[CNN]
    path: Param[Path]

    def execute(self):
        pass


class Evaluate(Task):
    model: Param[CNN]
    data: Param[Dataset]

    def execute(self):
        pass


@dataclass
class SampleWorkspace:
    path: Path
    learn_id: str
    evaluate_id: str

    @property
    def learn_params(self) -> Path:
        return self.path / "jobs" / f"{__name__}.learn" / self.learn_id / "params.json"

    @property
    def evaluate_params(self) -> Path:
        return (
            self.path
            / "jobs"
            / f"{__name__}.evaluate"
            / self.evaluate_id
            / "params.json"
        )


@pytest.fixture(scope="module")
def sample_workspace(tmp_path_factory) -> SampleWorkspace:
    """Workspace whose params.json files are generated (jobs are not run)"""
    workdir = tmp_path_factory.mktemp("workspace").resolve()
    with TemporaryExperiment("inspect", workdir=workdir):
        cnn = CNN.C(hidden_dim=2, kernel_size=3)
        learn = Learn.C(
            data=Dataset.C(name="train"), model=cnn, batch_size=64, lr=0.001
        ).submit(run_mode=RunMode.GENERATE_ONLY)
        evaluate = Evaluate.C(model=cnn, data=Dataset.C(name="test")).submit(
            init_tasks=[ParameterLoader.C(model=cnn, path=learn.run_path)],
            run_mode=RunMode.GENERATE_ONLY,
        )
        # More than 16 jobs: at least two share their first hexadecimal digit
        for ix in range(16):
            Evaluate.C(model=cnn, data=Dataset.C(name=f"test-{ix}")).submit(
                run_mode=RunMode.GENERATE_ONLY
            )

    ws = SampleWorkspace(
        workdir,
        learn.__xpm__.identifier.all.hex(),
        evaluate.__xpm__.identifier.all.hex(),
    )
    (ws.learn_params.parent / "runs").mkdir()
    return ws


@pytest.fixture(autouse=True)
def setup_test_workspaces(sample_workspace):
    """Settings with the sample workspace only"""
    settings = Settings()
    settings.workspaces.append(
        WorkspaceSettings(id="sample", path=sample_workspace.path)
    )
    with patch("experimaestro.settings.get_settings", return_value=settings):
        yield


def test_learn_params_graph_parsing(sample_workspace):
    """Verify parsing and reference graph construction on the learn job."""
    params_path = sample_workspace.learn_params
    assert params_path.exists(), f"Test fixture not found: {params_path}"
    graph = ParamsGraph.from_file(params_path)

    assert graph.version == 3
    assert len(graph.objects) == 3

    # Root task should be Learn
    assert graph.root_task_id is not None
    root = graph.objects[graph.root_task_id]
    assert root.type_name == "Learn"
    assert root.is_root_task is True
    assert root.is_task is True
    assert root.identifier == sample_workspace.learn_id

    # Outgoing references
    out_map = dict(root.outgoing_refs)
    assert "data" in out_map
    assert "model" in out_map

    # Check model (CNN)
    cnn_id = out_map["model"]
    cnn_obj = graph.objects[cnn_id]
    assert cnn_obj.type_name == "CNN"
    assert cnn_obj.fields["hidden_dim"] == 2
    assert cnn_obj.fields["kernel_size"] == 3
    assert (root.id, "model") in cnn_obj.incoming_refs

    # Check path resolution
    run_path_res = graph.resolve_path(root.fields["run_path"])
    assert isinstance(run_path_res, ResolvedPath)
    assert run_path_res.base == "job"
    assert run_path_res.raw_value == "runs"
    assert run_path_res.resolved == params_path.parent / "runs"
    assert run_path_res.exists is True
    assert run_path_res.is_dir is True

    # Python code synthesis
    py_code = graph.to_python_snippet(root.id)
    assert "Learn.C(" in py_code
    assert "CNN.C(" in py_code
    assert "batch_size=64" in py_code
    assert "lr=0.001" in py_code


def test_evaluate_params_with_init_tasks_and_shared_nodes(sample_workspace):
    """Verify parsing on evaluate job with init-tasks and shared CNN config."""
    params_path = sample_workspace.evaluate_params
    assert params_path.exists(), f"Test fixture not found: {params_path}"
    graph = ParamsGraph.from_file(params_path)

    assert graph.root_task_id is not None
    root = graph.objects[graph.root_task_id]
    assert root.type_name == "Evaluate"
    assert len(root.init_tasks) == 1

    init_task_id = root.init_tasks[0]
    init_task_node = graph.objects[init_task_id]
    assert init_task_node.type_name == "ParameterLoader"

    # CNN model should be shared between Evaluate and ParameterLoader
    cnn_id = None
    for k, v in root.fields.items():
        if k == "model":
            cnn_id = int(v["value"])
    assert cnn_id is not None
    cnn_node = graph.objects[cnn_id]
    assert cnn_node.type_name == "CNN"
    assert len(cnn_node.incoming_refs) >= 2


def test_cycle_detection():
    """Verify that circular references are safely handled without infinite recursion."""
    cycle_data = {
        "workspace": "/tmp/test",
        "version": 3,
        "experimaestro": "test",
        "objects": [
            {
                "id": 101,
                "module": "test.pkg",
                "type": "ConfigA",
                "identifier": "aaaa" * 16,
                "fields": {
                    "next": {"type": "python", "value": 102},
                    "val": 42,
                },
            },
            {
                "id": 102,
                "module": "test.pkg",
                "type": "ConfigB",
                "identifier": "bbbb" * 16,
                "fields": {
                    "loop_back": {"type": "python", "value": 101},
                    "name": "hello",
                },
            },
        ],
    }

    graph = ParamsGraph(cycle_data, job_path=Path("/tmp/test/job"))
    assert len(graph.objects) == 2
    assert graph.root_task_id == 102

    # Code generator should not crash on cycle
    code = graph.to_python_snippet(101)
    assert "# cycle to #101" in code

    # Tree should render cycle node cleanly
    tree = ParamsTree(graph)
    tree.show_root = False
    tree.guide_depth = 3
    tree.rebuild()

    # Look for cycle node
    cycle_nodes = []

    def find_cycle(node):
        if node.data and node.data.kind == NodeKind.CYCLE:
            cycle_nodes.append(node)
        for c in node.children:
            find_cycle(c)

    find_cycle(tree.root)
    assert len(cycle_nodes) > 0, "Expected a cycle node to be rendered"
    assert cycle_nodes[0].data.cycle_chain == [102, 101, 102]


@pytest.mark.anyio
async def test_app_pilot_smoke_test(sample_workspace):
    """Headless pilot test of ParamsInspectorApp."""
    app = ParamsInspectorApp(sample_workspace.learn_params)
    async with app.run_test() as pilot:
        await pilot.pause()

        # Check title and subtitle
        assert "Experimaestro Parameters Inspector" in app.title

        tree = app.query_one(ParamsTree)
        detail = app.query_one(NodeDetailPane)

        # Verify initial tree state (categorized grouping)
        assert tree.group_fields is True
        assert detail.show_raw_json is False

        # Toggle field grouping (Alphabetical vs Categorized)
        await pilot.press("g")
        await pilot.pause()
        assert tree.group_fields is False

        await pilot.press("g")
        await pilot.pause()
        assert tree.group_fields is True

        # Toggle paths (Relative vs Absolute)
        await pilot.press("p")
        await pilot.pause()
        assert tree.show_absolute_paths is True

        # Toggle Raw JSON view in detail pane
        await pilot.press("J")
        await pilot.pause()
        assert detail.show_raw_json is True

        await pilot.press("J")
        await pilot.pause()
        assert detail.show_raw_json is False

        # Toggle search bar
        await pilot.press("slash")
        await pilot.pause()
        search_box = app.query_one("#search-bar-container")
        assert "hidden" not in search_box.classes

        # Close search bar with escape
        await pilot.press("escape")
        await pilot.pause()
        assert "hidden" in search_box.classes


@pytest.mark.anyio
async def test_app_pilot_jump_unfolds_config(sample_workspace):
    """Verify that jumping to primary definition with 'j' unfolds folded ancestors and config."""
    app = ParamsInspectorApp(sample_workspace.evaluate_params)
    async with app.run_test() as pilot:
        await pilot.pause()
        tree = app.query_one(ParamsTree)
        app.action_expand_all()
        await pilot.pause()

        # Find shared node pointing to CNN
        shared_node = None

        def find_shared(n):
            nonlocal shared_node
            if n.data and n.data.target_object_id:
                shared_node = n
                return
            for c in n.children:
                find_shared(c)

        find_shared(tree.root)
        assert shared_node is not None

        tid = shared_node.data.target_object_id
        primary = tree.primary_nodes[tid]

        # Explicitly collapse primary and its parent category
        primary.collapse()
        primary.parent.collapse()
        await pilot.pause()
        assert primary.is_expanded is False
        assert primary.parent.is_expanded is False

        # Move cursor to shared node
        tree.move_cursor(shared_node)
        await pilot.pause()
        assert tree.cursor_node == shared_node

        # Press 'j' to jump
        await pilot.press("j")
        await pilot.pause()

        # Verify cursor jumped to primary, and both parent and primary are unfolded
        assert tree.cursor_node == primary
        assert primary.parent.is_expanded is True
        assert primary.is_expanded is True

        # Re-collapse and test that Enter on a shared node also jumps and unfolds
        primary.collapse()
        primary.parent.collapse()
        await pilot.pause()
        assert primary.is_expanded is False

        tree.move_cursor(shared_node)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert tree.cursor_node == primary
        assert primary.parent.is_expanded is True
        assert primary.is_expanded is True


def test_find_params_file_absolute_path(sample_workspace):
    """Verify that absolute paths resolve directly to file or directory."""
    params = sample_workspace.learn_params

    # Absolute path to params.json
    assert find_params_file(str(params)) == params

    # Absolute path with explicit local workspace specified must NOT load from workspace
    assert find_params_file(str(params), workspace="sample") == params

    # Absolute path to parent job directory
    assert find_params_file(str(params.parent)) == params

    # Non-existent absolute path
    with pytest.raises(
        FileNotFoundError, match="params.json not found at absolute path"
    ):
        find_params_file("/tmp/nonexistent_dir_12345/params.json")


def test_find_params_file_job_id_default_workspace(
    sample_workspace, caplog, monkeypatch
):
    """Verify that job ID prefix resolves via default workspace discovery and logs it."""
    import logging

    monkeypatch.setenv("EXPERIMAESTRO_WORKSPACE", str(sample_workspace.path))
    with (
        patch("experimaestro.settings.get_settings", return_value=Settings()),
        caplog.at_level(logging.INFO),
    ):
        assert resolve_workspace() == sample_workspace.path
        res = find_params_file(sample_workspace.learn_id[:7])
        assert res == sample_workspace.learn_params
        assert "Using default workspace" in caplog.text


def test_find_params_file_explicit_workspace(sample_workspace, caplog):
    """Verify resolving job via explicit workspace name/path and logging it."""
    import logging

    prefix = sample_workspace.evaluate_id[:7]

    # By workspace name
    res1 = find_params_file(prefix, workspace="sample")
    assert res1 == sample_workspace.evaluate_params

    # By workspace path
    with caplog.at_level(logging.INFO):
        res2 = find_params_file(prefix, workspace=str(sample_workspace.path))
        assert res2 == sample_workspace.evaluate_params
        assert f"Using workspace {sample_workspace.path}" in caplog.text


def test_find_params_file_multiple_matches_warning(sample_workspace, capsys):
    """Verify warning when multiple jobs match a short prefix."""
    first_digits = [
        p.parent.name[0] for p in sample_workspace.path.glob("jobs/*/*/params.json")
    ]
    prefix = next(c for c in first_digits if first_digits.count(c) > 1)

    res = find_params_file(prefix, workspace="sample")
    captured = capsys.readouterr()
    assert "Warning: Multiple jobs" in captured.err
    assert "Using the first match:" in captured.err
    assert res.exists()
    assert res.name == "params.json"


def test_find_params_file_not_found():
    """Verify FileNotFoundError when job ID does not exist."""
    with pytest.raises(
        FileNotFoundError, match="No job matching 'nonexistent_job_prefix_xyz' found"
    ):
        find_params_file("nonexistent_job_prefix_xyz", workspace="sample")


def make_mock_remote_workspace() -> WorkspaceSettings:
    return WorkspaceSettings(
        id="mock-remote-ws",
        path=Path("/remote/workspace/path"),
        ssh=SSHSettings(host="remote-host", options=["-o", "StrictHostKeyChecking=no"]),
    )


def test_clean_job_target():
    """Verify target normalization with various prefixes and paths."""
    assert clean_job_target("ee76fb8f") == "ee76fb8f"
    assert clean_job_target("my_task/ee76fb8f") == "my_task/ee76fb8f"
    assert clean_job_target("jobs/my_task/ee76fb8f") == "my_task/ee76fb8f"
    assert clean_job_target("/jobs/my_task/ee76fb8f") == "my_task/ee76fb8f"
    assert (
        clean_job_target(
            "/remote/ws/jobs/my_task/ee76fb8f/params.json",
            ws_base="/remote/ws",
            filename="params.json",
        )
        == "my_task/ee76fb8f"
    )
    assert (
        clean_job_target(
            "my_task/ee76fb8f/experimaestro.json", filename="experimaestro.json"
        )
        == "my_task/ee76fb8f"
    )
    with pytest.raises(ValueError):
        clean_job_target("")
    with pytest.raises(ValueError):
        clean_job_target(".")


def test_resolve_job_file_pattern():
    """Verify relative glob pattern construction."""
    assert (
        resolve_job_file_pattern("ee76fb8f", "params.json")
        == "jobs/*/ee76fb8f*/params.json"
    )
    assert (
        resolve_job_file_pattern("my_task/ee76fb8f", "params.json")
        == "jobs/my_task/ee76fb8f*/params.json"
    )
    assert (
        resolve_job_file_pattern("my_task:ee76fb8f", "experimaestro.json")
        == "jobs/my_task/ee76fb8f*/experimaestro.json"
    )


def test_find_local_workspace_file_success(tmp_path):
    """Verify finding files within a local workspace directory."""
    job_dir = tmp_path / "jobs" / "my_task" / "ee76fb8f12345"
    job_dir.mkdir(parents=True)
    params_file = job_dir / "params.json"
    params_file.write_text("{}", encoding="utf-8")
    xp_file = job_dir / "experimaestro.json"
    xp_file.write_text("[]", encoding="utf-8")

    # Find params.json by prefix
    found = find_local_workspace_file(tmp_path, "ee76fb8f")
    assert found == params_file.resolve()

    # Find custom filename by task/prefix
    found_xp = find_local_workspace_file(
        tmp_path, "my_task/ee76", filename="experimaestro.json"
    )
    assert found_xp == xp_file.resolve()


def test_find_local_workspace_file_not_found(tmp_path):
    """Verify FileNotFoundError when local workspace has no matching job file."""
    with pytest.raises(FileNotFoundError, match="No job matching 'nonexistent' found"):
        find_local_workspace_file(tmp_path, "nonexistent")


def test_fetch_remote_workspace_file_success(tmp_path, monkeypatch):
    """Verify remote file discovery via SSH and synchronization via rsync."""
    ws = make_mock_remote_workspace()
    target_job = "ee76fb8f"
    remote_matched_path = f"{ws._raw_path}/jobs/my_task/{target_job}12345/params.json"
    cache_dir = tmp_path / "cache"

    def mock_subprocess_run(cmd, capture_output, text):
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        if cmd[0] == "ssh":
            assert ws.ssh.host in cmd
            res = Result()
            res.stdout = f"{remote_matched_path}\n"
            return res
        elif cmd[0] == "rsync":
            dest_file = Path(cmd[-1])
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            dest_file.write_text(
                '{"workspace": "/remote/workspace/path"}', encoding="utf-8"
            )
            return Result()
        return Result()

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)

    result_path = fetch_remote_workspace_file(ws, target_job, cache_dir=cache_dir)
    assert (
        result_path
        == cache_dir / "jobs" / "my_task" / f"{target_job}12345" / "params.json"
    )
    assert result_path.is_file()


def test_fetch_remote_workspace_file_caches_and_syncs_once(tmp_path, monkeypatch):
    """Verify that remote file is only synced once and subsequent queries use local cache."""
    ws = make_mock_remote_workspace()
    target_job = "ee76fb8f"
    remote_matched_path = f"{ws._raw_path}/jobs/my_task/{target_job}12345/params.json"
    cache_dir = tmp_path / "cache"

    call_count = {"ssh": 0, "rsync": 0}

    def mock_subprocess_run(cmd, capture_output, text):
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        if cmd[0] == "ssh":
            call_count["ssh"] += 1
            res = Result()
            res.stdout = f"{remote_matched_path}\n"
            return res
        elif cmd[0] == "rsync":
            call_count["rsync"] += 1
            dest_file = Path(cmd[-1])
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            dest_file.write_text(
                '{"workspace": "/remote/workspace/path"}', encoding="utf-8"
            )
            return Result()
        return Result()

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)

    # First call: triggers SSH + rsync
    res1 = fetch_remote_workspace_file(ws, target_job, cache_dir=cache_dir)
    assert res1.is_file()
    assert call_count == {"ssh": 1, "rsync": 1}

    # Second call for the same target: served from local cache with 0 SSH/rsync calls!
    res2 = fetch_remote_workspace_file(ws, target_job, cache_dir=cache_dir)
    assert res2 == res1
    assert call_count == {"ssh": 1, "rsync": 1}

    # Verify default get_session_cache_dir() behavior when cache_dir is None
    session_cache = tmp_path / "session_cache"
    monkeypatch.setattr(
        "experimaestro.utils.remote.get_session_cache_dir", lambda: session_cache
    )
    target_job2 = "abc12345"
    remote_matched_path2 = f"{ws._raw_path}/jobs/my_task/{target_job2}/params.json"

    def mock_subprocess_run_session(cmd, capture_output, text):
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        if cmd[0] == "ssh":
            call_count["ssh"] += 1
            res = Result()
            res.stdout = f"{remote_matched_path2}\n"
            return res
        elif cmd[0] == "rsync":
            call_count["rsync"] += 1
            dest_file = Path(cmd[-1])
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            dest_file.write_text('{"workspace": "/remote"}', encoding="utf-8")
            return Result()
        return Result()

    monkeypatch.setattr("subprocess.run", mock_subprocess_run_session)
    call_count["ssh"] = 0
    call_count["rsync"] = 0

    res3 = fetch_remote_workspace_file(ws, target_job2)
    assert res3.is_file()
    assert call_count == {"ssh": 1, "rsync": 1}

    res4 = fetch_remote_workspace_file(ws, target_job2)
    assert res4 == res3
    assert call_count == {"ssh": 1, "rsync": 1}


def test_fetch_remote_workspace_file_custom_filename(tmp_path, monkeypatch):
    """Verify fetching custom file (experimaestro.json) from remote workspace."""
    ws = make_mock_remote_workspace()
    remote_matched_path = (
        f"{ws._raw_path}/jobs/my_task/ee76fb8f12345/experimaestro.json"
    )
    cache_dir = tmp_path / "cache"

    def mock_subprocess_run(cmd, capture_output, text):
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        if cmd[0] == "ssh":
            res = Result()
            res.stdout = f"{remote_matched_path}\n"
            return res
        elif cmd[0] == "rsync":
            dest_file = Path(cmd[-1])
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            dest_file.write_text("[]", encoding="utf-8")
            return Result()
        return Result()

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)

    result_path = fetch_remote_workspace_file(
        ws, "ee76fb8f", filename="experimaestro.json", cache_dir=cache_dir
    )
    assert (
        result_path
        == cache_dir / "jobs" / "my_task" / "ee76fb8f12345" / "experimaestro.json"
    )
    assert result_path.is_file()


def test_fetch_remote_workspace_file_multiple_matches(tmp_path, monkeypatch, capsys):
    """Verify warning printed and first match taken when multiple remote jobs match."""
    ws = make_mock_remote_workspace()
    matches = [
        f"{ws._raw_path}/jobs/task_a/1111_job/params.json",
        f"{ws._raw_path}/jobs/task_b/1112_job/params.json",
    ]
    cache_dir = tmp_path / "cache"

    def mock_subprocess_run(cmd, capture_output, text):
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        if cmd[0] == "ssh":
            res = Result()
            res.stdout = "\n".join(matches) + "\n"
            return res
        elif cmd[0] == "rsync":
            dest_file = Path(cmd[-1])
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            dest_file.write_text("{}", encoding="utf-8")
            return Result()
        return Result()

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)

    result_path = fetch_remote_workspace_file(ws, "111", cache_dir=cache_dir)
    captured = capsys.readouterr()
    assert "Warning: Multiple jobs (2) matched '111'" in captured.err
    assert matches[0] in captured.err
    assert result_path == cache_dir / "jobs" / "task_a" / "1111_job" / "params.json"


def test_fetch_remote_workspace_file_not_found(monkeypatch):
    """Verify FileNotFoundError when no remote job matches."""
    ws = make_mock_remote_workspace()

    def mock_subprocess_run(cmd, capture_output, text):
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)

    with pytest.raises(
        FileNotFoundError,
        match="No job matching 'nonexistent' found in remote workspace 'mock-remote-ws'",
    ):
        fetch_remote_workspace_file(ws, "nonexistent")


def test_fetch_remote_workspace_file_ssh_error(monkeypatch):
    """Verify RuntimeError when SSH command fails."""
    ws = make_mock_remote_workspace()

    def mock_subprocess_run(cmd, capture_output, text):
        class Result:
            returncode = 255
            stdout = ""
            stderr = "Host unreachable"

        return Result()

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)

    with pytest.raises(
        RuntimeError,
        match="Failed to query remote workspace 'mock-remote-ws' via SSH: Host unreachable",
    ):
        fetch_remote_workspace_file(ws, "some_job")


def test_fetch_remote_workspace_file_empty_target():
    """Verify FileNotFoundError when target is empty or root."""
    ws = make_mock_remote_workspace()
    with pytest.raises(FileNotFoundError, match="Please specify a job ID or path"):
        fetch_remote_workspace_file(ws, "")

    with pytest.raises(FileNotFoundError, match="Please specify a job ID or path"):
        fetch_remote_workspace_file(ws, ".")


def test_find_params_file_remote_dispatch(tmp_path, monkeypatch):
    """Verify that find_params_file dispatches to find_workspace_file for remote workspace."""
    ws = make_mock_remote_workspace()
    expected_path = tmp_path / "params.json"

    monkeypatch.setattr(
        "experimaestro.tools.inspect_params.discovery.resolve_workspace", lambda w: ws
    )
    monkeypatch.setattr(
        "experimaestro.tools.inspect_params.discovery.find_workspace_file",
        lambda workspace, target, filename="params.json": expected_path,
    )

    result = find_params_file("my_remote_job", workspace="mock-remote-ws")
    assert result == expected_path


def test_find_params_file_remote_workspace_overrides_absolute_path(
    tmp_path, monkeypatch
):
    """Verify that an absolute path is dispatched to remote workspace rather than local resolution."""
    ws = make_mock_remote_workspace()
    expected_path = tmp_path / "params.json"
    remote_abs_path = (
        f"{ws._raw_path}/jobs/my_task/b0ef27e326897e6d874df9a028ba1cbf09177958"
    )

    called_with = []
    monkeypatch.setattr(
        "experimaestro.tools.inspect_params.discovery.resolve_workspace", lambda w: ws
    )
    monkeypatch.setattr(
        "experimaestro.tools.inspect_params.discovery.find_workspace_file",
        lambda workspace, target, filename="params.json": (
            called_with.append((workspace, target, filename)) or expected_path
        ),
    )

    result = find_params_file(remote_abs_path, workspace="mock-remote-ws")
    assert result == expected_path
    assert called_with == [(ws, remote_abs_path, "params.json")]


def test_fetch_remote_workspace_file_with_full_remote_absolute_path(
    tmp_path, monkeypatch
):
    """Verify fetch_remote_workspace_file correctly strips remote workspace prefix and /jobs/."""
    ws = make_mock_remote_workspace()
    target_job = "b0ef27e3"
    full_remote_path = (
        f"{ws._raw_path}/jobs/xpmir.evaluation.evaluate/{target_job}12345/params.json"
    )
    cache_dir = tmp_path / "cache"

    executed_cmds = []

    def mock_subprocess_run(cmd, capture_output, text):
        executed_cmds.append(cmd)

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        if cmd[0] == "ssh":
            res = Result()
            res.stdout = f"{full_remote_path}\n"
            return res
        elif cmd[0] == "rsync":
            dest_file = Path(cmd[-1])
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            dest_file.write_text("{}", encoding="utf-8")
            return Result()
        return Result()

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)

    result = fetch_remote_workspace_file(ws, full_remote_path, cache_dir=cache_dir)
    assert (
        result
        == cache_dir
        / "jobs"
        / "xpmir.evaluation.evaluate"
        / f"{target_job}12345"
        / "params.json"
    )

    # Check that SSH command queried the pattern properly without duplicated prefix
    assert len(executed_cmds) == 2
    assert executed_cmds[0][0] == "ssh"
    remote_py_cmd = executed_cmds[0][-1]
    assert (
        f"{ws._raw_path}/jobs/xpmir.evaluation.evaluate/{target_job}12345*/params.json"
        in remote_py_cmd
    )


def test_find_workspace_file_local(tmp_path):
    """Verify find_workspace_file dispatches to local workspace directory."""
    job_dir = tmp_path / "jobs" / "task_x" / "9999"
    job_dir.mkdir(parents=True)
    f = job_dir / "experimaestro.json"
    f.write_text("[]", encoding="utf-8")

    res = find_workspace_file(tmp_path, "9999", filename="experimaestro.json")
    assert res == f.resolve()


def test_params_graph_list_payload(tmp_path):
    """Verify that ParamsGraph supports direct list payloads (experimaestro.json format)."""
    list_data = [
        {
            "id": 1001,
            "module": "mock_mod",
            "type": "MockTokenizer",
            "identifier": "tok123",
            "fields": {"model_name": "bert"},
        },
        {
            "id": 1002,
            "module": "mock_mod",
            "type": "MockModel",
            "identifier": "mod123",
            "task": 1002,
            "fields": {
                "tokenizer": {"type": "python", "value": 1001},
                "weights": {"type": "path.serialized", "value": "model.safetensors"},
            },
        },
    ]

    p = tmp_path / "experimaestro.json"
    p.write_text(json.dumps(list_data), encoding="utf-8")
    (tmp_path / "model.safetensors").write_text("weights")

    graph = ParamsGraph.from_file(p)
    assert len(graph.objects) == 2
    assert graph.workspace_path is None
    assert graph.version == 0
    assert graph.experimaestro_version == "unknown"
    assert graph.root_task_id == 1002

    model_node = graph.objects[1002]
    assert model_node.is_root_task is True
    assert ("tokenizer", 1001) in model_node.outgoing_refs

    # Path resolution for path.serialized relative to job_dir
    weights_path = graph.resolve_path(model_node.fields["weights"])
    assert weights_path is not None
    assert weights_path.exists is True
    assert weights_path.resolved == tmp_path / "model.safetensors"

    # Python snippet formatting
    py_code = graph.to_python_snippet(1002)
    assert "MockModel.C(" in py_code
    assert 'Path("model.safetensors")' in py_code


def test_find_params_file_directory_with_experimaestro_json(tmp_path):
    """Verify finding experimaestro.json in directory when params.json is absent."""
    exp_file = tmp_path / "experimaestro.json"
    exp_file.write_text("[]", encoding="utf-8")

    res = find_params_file(str(tmp_path))
    assert res == exp_file


@pytest.mark.anyio
async def test_experimaestro_json_tui_app(tmp_path):
    """Verify that ParamsInspectorApp mounts and renders multi-task roots from list payloads."""
    list_data = [
        {
            "id": 2001,
            "module": "pkg",
            "type": "DatasetTask",
            "identifier": "ds2001",
            "task": 2001,
            "fields": {"size": 100},
        },
        {
            "id": 2002,
            "module": "pkg",
            "type": "EvalTask",
            "identifier": "eval2002",
            "task": 2002,
            "fields": {"metric": "ndcg"},
        },
    ]
    p = tmp_path / "experimaestro.json"
    p.write_text(json.dumps(list_data), encoding="utf-8")

    app = ParamsInspectorApp(p)
    async with app.run_test():
        tree = app.query_one(ParamsTree)
        assert len(tree.root.children) >= 2
        labels = [str(c.label) for c in tree.root.children]
        assert any("Root Task" in lbl for lbl in labels)
        assert any("Other Tasks" in lbl for lbl in labels)


@pytest.mark.anyio
async def test_root_model_primary_placement(tmp_path):
    """Verify that core fields ('value', 'model') render as primary at the root rather than in listeners."""
    # Construct a task where 'settings' alphabetically precedes 'value',
    # and 'settings' contains a nested listener referencing the same model.
    data = [
        {
            "id": 1,
            "module": "mock",
            "type": "TargetModel",
            "identifier": "model123",
            "fields": {"hidden_size": 768},
        },
        {
            "id": 2,
            "module": "mock",
            "type": "Listener",
            "identifier": "listener123",
            "fields": {
                "scorer": {"type": "python", "value": 1},
            },
        },
        {
            "id": 3,
            "module": "mock",
            "type": "RootTask",
            "identifier": "roottask123",
            "task": 3,
            "fields": {
                "settings": {"type": "python", "value": 2},
                "value": {"type": "python", "value": 1},
            },
        },
    ]
    p = tmp_path / "experimaestro.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    app = ParamsInspectorApp(p)
    async with app.run_test():
        tree = app.query_one(ParamsTree)
        # Root task is child 1 (after metadata)
        root_node = tree.root.children[1]

        # The primary node for model #1 must be a direct child of root task under 'value'
        primary_model = tree.primary_nodes.get(1)
        assert primary_model is not None
        assert primary_model.parent == root_node
        assert "value:" in str(primary_model.label)
        assert "Shared" not in str(primary_model.label)

        # The listener's reference under 'settings' must be marked as Shared
        settings_node = [c for c in root_node.children if "settings:" in str(c.label)][
            0
        ]
        listener_scorer = settings_node.children[0]
        assert "Shared" in str(listener_scorer.label)
        assert listener_scorer.data.target_object_id == 1


def test_cli_inspect_help():
    """Verify experimaestro inspect and experimaestro jobs inspect CLI commands."""
    from click.testing import CliRunner
    from experimaestro.cli import cli

    runner = CliRunner()

    # 1. Main experimaestro --help lists inspect command
    res_main = runner.invoke(cli, ["--help"])
    assert res_main.exit_code == 0
    assert "inspect" in res_main.output

    # 2. Top-level experimaestro inspect --help has no deprecation warning and documents root -w
    res_inspect = runner.invoke(cli, ["inspect", "--help"])
    assert res_inspect.exit_code == 0
    assert "Interactive TUI Parameters Inspector" in res_inspect.output
    assert "Deprecated" not in res_inspect.output
    assert "experimaestro -w my_workspace inspect" in res_inspect.output

    # 3. Subcommand experimaestro jobs inspect --help also works
    res_jobs = runner.invoke(cli, ["jobs", "inspect", "--help"])
    assert res_jobs.exit_code == 0
    assert "Interactive TUI Parameters Inspector" in res_jobs.output


def test_params_graph_objects_jsonl_and_value_resolution(tmp_path):
    """Verify ParamsGraph and find_params_file support objects.jsonl and resolve_serialized_value."""
    from experimaestro.tools.inspect_params import resolve_serialized_value

    objs = [
        {
            "id": 10,
            "module": "pkg.models",
            "type": "Encoder",
            "identifier": "enc_hash",
            "fields": {
                "dim": 512,
                "mode": {"type": "enum", "enum": "Mode", "value": "TRAIN"},
                "tags_set": {"type": "set", "items": ["a", "b"]},
            },
        },
        {
            "id": 20,
            "module": "pkg.tasks",
            "type": "LearnTask",
            "identifier": "learn_hash",
            "fields": {"model": {"type": "python", "value": 10}},
        },
        {
            "id": 30,
            "module": "pkg.models",
            "type": "CheckpointLoader",
            "identifier": "ckpt_hash",
            "task": 20,
            "fields": {"path": {"type": "path", "value": "/tmp/ckpt"}},
        },
    ]

    jsonl_path = tmp_path / "objects.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(o) for o in objs) + "\n", encoding="utf-8"
    )

    assert find_params_file(str(tmp_path)) == jsonl_path
    graph = ParamsGraph.from_file(tmp_path, load_docstrings=False)

    assert len(graph.objects) == 3
    assert 20 in graph.task_object_ids
    assert graph.objects[20].is_task is True
    assert graph.objects[30].task_id == 20

    resolved_fields = resolve_serialized_value(graph.objects[10].fields, graph.objects)
    assert resolved_fields["dim"] == 512
    assert resolved_fields["mode"] == "TRAIN"
    assert resolved_fields["tags_set"] == ["a", "b"]
    assert (
        resolve_serialized_value(graph.objects[20].fields["model"], graph.objects)
        == graph.objects[10]
    )
