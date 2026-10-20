"""Tests for the standalone experimaestro DAG visualization tool."""

import json
import logging

from experimaestro import Config, Param, Task
from experimaestro.tools.xpm_dag import (
    build_dag_json,
    draw_dag,
    extract_config_tags,
    generate_dag_html,
    get_experiment_configs,
)


class DummyModel(Config):
    hidden_dim: Param[int]
    kernel_size: Param[int]
    n_layers: Param[int]


class DummyTask(Task):
    pass


def test_extract_config_tags():
    """Verify extract_config_tags extracts hyperparameter tags from Config objects."""
    model = DummyModel.C(hidden_dim=32, kernel_size=3, n_layers=2)

    # 1. With grid_keys: filters to specified grid dimensions
    tags_filtered = extract_config_tags(model, {"hidden_dim", "kernel_size"})
    assert tags_filtered == {"hidden_dim": 32, "kernel_size": 3}

    # 2. Without grid_keys: returns all primitive scalar attributes
    tags_all = extract_config_tags(model)
    assert tags_all == {"hidden_dim": 32, "kernel_size": 3, "n_layers": 2}

    # 3. None or non-config object
    assert extract_config_tags(None) == {}
    assert extract_config_tags("string") == {}


def test_get_experiment_configs_detection_logging(tmp_path, caplog):
    """Verify that get_experiment_configs logs detected grid search configurations at INFO level."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    jobs_file = run_dir / "jobs.jsonl"
    records = [
        {"job_id": "j1", "tags": {"lr": 0.01, "hidden_dim": 32}},
        {"job_id": "j2", "tags": {"lr": 0.02, "hidden_dim": 64}},
        {"job_id": "j3", "tags": None},
    ]
    with jobs_file.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    with caplog.at_level(logging.INFO):
        configs = get_experiment_configs(run_dir)
        assert len(configs) == 2
        assert "Detected 2 grid search configuration(s)" in caplog.text
        assert "hidden_dim, lr" in caplog.text


def test_build_dag_json_path_display(tmp_path):
    """Verify build_dag_json populates path, display_path, and config tags."""
    model = DummyModel.C(hidden_dim=64, kernel_size=5, n_layers=1)
    task = DummyTask()

    model_id = "model_" + "a" * 60
    task_id = "task_" + "b" * 60

    full_jobs = {model_id: model, task_id: task}
    dependencies = {task_id: [model_id]}
    job_tags_dict = {task_id: {"lr": 0.01, "hidden_dim": 64}}

    workspace = tmp_path / "my_workspace"
    run_dir = workspace / "experiments" / "test_exp" / "20261016_120000"
    run_dir.mkdir(parents=True)

    jobs_file = run_dir / "jobs.jsonl"
    with jobs_file.open("w") as f:
        f.write(
            json.dumps(
                {
                    "job_id": task_id,
                    "task_id": "dummy.package.task",
                    "tags": {"lr": 0.01, "hidden_dim": 64},
                }
            )
            + "\n"
        )

    dag_json = build_dag_json(
        full_jobs,
        dependencies,
        job_tags_dict,
        run_dir=run_dir,
        workspace=str(workspace),
    )

    nodes_by_id = {n["id"]: n for n in dag_json["nodes"]}
    task_node = nodes_by_id[task_id]
    model_node = nodes_by_id[model_id]

    expected_full_path = str(workspace / "jobs" / "dummy.package.task" / task_id)
    assert task_node["path"] == expected_full_path
    assert task_node["locator"] == expected_full_path
    assert task_node["display_path"] == f".../dummy.package.task/{task_id[:20]}..."

    assert model_node["tags"] == {"hidden_dim": 64}
    assert model_node["path"] == ""
    assert model_node["display_path"] == ""


def test_web_dag_renders_path_in_drawer():
    """Verify generated HTML drawer contains Path row and display_path."""
    dag_json = {
        "experiment_id": "exp_test",
        "run_id": "run_test",
        "workspace": "/tmp/workspace",
        "nodes": [
            {
                "id": "job_12345678901234567890",
                "name": "Learn",
                "task_id": "my_package.learn",
                "type": "task",
                "status": "done",
                "tags": {"lr": 0.001, "hidden_dim": 32},
                "path": "/tmp/workspace/jobs/my_package.learn/job_12345678901234567890",
                "display_path": ".../my_package.learn/job_1234567890123456...",
                "locator": "/tmp/workspace/jobs/my_package.learn/job_12345678901234567890",
                "depends_on": [],
                "depth": 0,
                "x": 0.0,
                "y": 0.0,
            }
        ],
        "edges": [],
        "configs": [{"lr": 0.001, "hidden_dim": 32}],
    }

    html = generate_dag_html(dag_json)
    assert '<tr><td class="key">Path</td>' in html
    assert "copyText" in html
    assert "node.display_path" in html


def test_draw_dag_cli_save_as(tmp_path, monkeypatch):
    """Verify draw_dag --save-as auto-detects extensions and saves without launching server."""
    full_jobs = {"j_root": DummyTask(), "j_train": DummyTask()}
    dependencies = {"j_train": ["j_root"]}
    job_tags_dict = {"j_train": {"lr": 0.001}}

    run_dir = tmp_path / "experiments" / "test_exp" / "20260101_120000"
    run_dir.mkdir(parents=True)

    monkeypatch.setattr(
        draw_dag,
        "resolve_run_dir",
        lambda ws, exp_id, run_id: ("20260101_120000", run_dir),
    )
    monkeypatch.setattr(
        draw_dag,
        "load_run_graph",
        lambda r_dir: (full_jobs, dependencies, job_tags_dict),
    )

    server_called = []
    monkeypatch.setattr(
        draw_dag,
        "serve_dag_web",
        lambda *args, **kwargs: server_called.append(True),
    )

    # 1. Test --save-as with .html: saves HTML and does NOT launch server even if --web was passed
    html_out = tmp_path / "subfolder" / "dag.html"
    monkeypatch.setattr(
        "sys.argv",
        ["draw_dag", "--web", "--save-as", str(html_out)],
    )
    draw_dag.main()
    assert html_out.exists()
    assert "<!DOCTYPE html>" in html_out.read_text()
    assert len(server_called) == 0, (
        "Server should not be launched when --save-as ends in .html"
    )

    # 2. Test --save-as with .json
    json_out = tmp_path / "subfolder" / "dag.json"
    monkeypatch.setattr(
        "sys.argv",
        ["draw_dag", "--save-as", str(json_out)],
    )
    draw_dag.main()
    assert json_out.exists()
    assert "j_train" in json_out.read_text()


def test_web_dag_template_loading():
    """Verify that the dedicated HTML template file loads correctly."""
    from experimaestro.tools.xpm_dag.xpm_web_dag import (
        HTML_TEMPLATE,
        _load_html_template,
    )

    loaded = _load_html_template()
    assert len(loaded) > 0
    assert loaded == HTML_TEMPLATE
    assert "<!DOCTYPE html>" in loaded
    assert "__TITLE__" in loaded
    assert "__RUN_ID__" in loaded
    assert "__DEFAULT_LAYOUT__" in loaded
    assert "__GRAPH_DATA__" in loaded


def test_cli_dag_help():
    """Verify experimaestro dag --help and experimaestro experiments dag --help."""
    from click.testing import CliRunner
    from experimaestro.cli import cli

    runner = CliRunner()
    res1 = runner.invoke(cli, ["dag", "--help"])
    assert res1.exit_code == 0
    assert "Visualize and export experiment dependency DAGs" in res1.output
    assert "--web" in res1.output
    assert "--save-as" in res1.output
    assert "--list-configs" in res1.output

    res2 = runner.invoke(cli, ["experiments", "dag", "--help"])
    assert res2.exit_code == 0
    assert "Visualize and export experiment dependency DAGs" in res2.output


def test_cli_dag_save_as_and_json(tmp_path, monkeypatch):
    """Verify experimaestro dag --save-as and --json via Click CLI."""
    from click.testing import CliRunner
    from experimaestro.cli import cli
    from experimaestro.tools.xpm_dag import draw_dag

    full_jobs = {"j_root": DummyTask(), "j_train": DummyTask()}
    dependencies = {"j_train": ["j_root"]}
    job_tags_dict = {"j_train": {"lr": 0.001}}

    run_dir = tmp_path / "experiments" / "test_exp" / "20260101_120000"
    run_dir.mkdir(parents=True)

    monkeypatch.setattr(
        draw_dag,
        "resolve_run_dir",
        lambda ws, exp_id, run_id: ("20260101_120000", run_dir),
    )
    monkeypatch.setattr(
        draw_dag,
        "load_run_graph",
        lambda r_dir: (full_jobs, dependencies, job_tags_dict),
    )

    runner = CliRunner()

    # 1. Test --save-as html
    html_out = tmp_path / "cli_out.html"
    res_html = runner.invoke(cli, ["dag", "test_exp", "--save-as", str(html_out)])
    assert res_html.exit_code == 0
    assert html_out.exists()
    assert "<!DOCTYPE html>" in html_out.read_text()

    # 2. Test --json stdout
    res_json = runner.invoke(cli, ["dag", "test_exp", "--json"])
    assert res_json.exit_code == 0
    parsed = json.loads(res_json.output)
    assert "nodes" in parsed
    assert any(n["id"] == "j_train" for n in parsed["nodes"])


def test_resolve_run_dir_auto_detect(tmp_path):
    """Verify resolve_run_dir auto-selects when a single experiment exists."""
    from experimaestro.tools.xpm_dag.xpm_dag import resolve_run_dir

    ws_dir = tmp_path / "workspace"
    exp_dir = ws_dir / "experiments" / "single_experiment" / "20260101_120000"
    exp_dir.mkdir(parents=True)

    # 1. Single experiment exists: auto-selects without passing experiment_id
    res = resolve_run_dir(workspace_arg=str(ws_dir), experiment_id=None)
    assert res is not None
    run_id, resolved_dir = res
    assert run_id == "20260101_120000"
    assert resolved_dir == exp_dir

    # 2. Multiple experiments exist: returns None when experiment_id is omitted
    exp2_dir = ws_dir / "experiments" / "another_experiment" / "20260101_130000"
    exp2_dir.mkdir(parents=True)
    res_multi = resolve_run_dir(workspace_arg=str(ws_dir), experiment_id=None)
    assert res_multi is None


def test_cli_dag_defaults_to_web(tmp_path, monkeypatch):
    """Verify experimaestro dag defaults to launching the web viewer."""
    from click.testing import CliRunner
    from experimaestro.cli import cli
    from experimaestro.tools.xpm_dag import draw_dag

    full_jobs = {"j_root": DummyTask(), "j_train": DummyTask()}
    dependencies = {"j_train": ["j_root"]}
    job_tags_dict = {"j_train": {"lr": 0.001}}

    run_dir = tmp_path / "experiments" / "test_exp" / "20260101_120000"
    run_dir.mkdir(parents=True)

    monkeypatch.setattr(
        draw_dag,
        "resolve_run_dir",
        lambda ws, exp_id, run_id: ("20260101_120000", run_dir),
    )
    monkeypatch.setattr(
        draw_dag,
        "load_run_graph",
        lambda r_dir: (full_jobs, dependencies, job_tags_dict),
    )

    server_called = []
    monkeypatch.setattr(
        draw_dag,
        "serve_dag_web",
        lambda *args, **kwargs: server_called.append(True),
    )

    runner = CliRunner()
    # Calling dag with NO flags: should default to web view
    res = runner.invoke(cli, ["dag", "test_exp"])
    assert res.exit_code == 0
    assert len(server_called) == 1, "Web server should be launched by default"


def test_cli_dag_optional_tree(tmp_path, monkeypatch):
    """Verify experimaestro dag --tree prints text tree and does NOT launch server."""
    from click.testing import CliRunner
    from experimaestro.cli import cli
    from experimaestro.tools.xpm_dag import draw_dag

    full_jobs = {"j_root": DummyTask(), "j_train": DummyTask()}
    dependencies = {"j_train": ["j_root"]}
    job_tags_dict = {"j_train": {"lr": 0.001}}

    run_dir = tmp_path / "experiments" / "test_exp" / "20260101_120000"
    run_dir.mkdir(parents=True)

    monkeypatch.setattr(
        draw_dag,
        "resolve_run_dir",
        lambda ws, exp_id, run_id: ("20260101_120000", run_dir),
    )
    monkeypatch.setattr(
        draw_dag,
        "load_run_graph",
        lambda r_dir: (full_jobs, dependencies, job_tags_dict),
    )

    server_called = []
    monkeypatch.setattr(
        draw_dag,
        "serve_dag_web",
        lambda *args, **kwargs: server_called.append(True),
    )

    runner = CliRunner()
    res = runner.invoke(cli, ["dag", "test_exp", "--tree"])
    assert res.exit_code == 0
    assert len(server_called) == 0, (
        "Web server should not be launched when --tree is given"
    )
    assert "DummyTask" in res.output
    assert "j_root" in res.output


def test_matplotlib_lazy_import(tmp_path, monkeypatch):
    """Verify that running web view or --tree does not import matplotlib."""
    import sys
    from click.testing import CliRunner
    from experimaestro.cli import cli
    from experimaestro.tools.xpm_dag import draw_dag

    full_jobs = {"j_root": DummyTask()}
    dependencies = {}
    job_tags_dict = {}

    run_dir = tmp_path / "experiments" / "test_exp" / "20260101_120000"
    run_dir.mkdir(parents=True)

    monkeypatch.setattr(
        draw_dag,
        "resolve_run_dir",
        lambda ws, exp_id, run_id: ("20260101_120000", run_dir),
    )
    monkeypatch.setattr(
        draw_dag,
        "load_run_graph",
        lambda r_dir: (full_jobs, dependencies, job_tags_dict),
    )

    # Ensure matplotlib is not in sys.modules
    sys.modules.pop("matplotlib", None)
    sys.modules.pop("matplotlib.pyplot", None)

    runner = CliRunner()
    res = runner.invoke(cli, ["dag", "test_exp", "--tree"])
    assert res.exit_code == 0
    assert "matplotlib" not in sys.modules
    assert "matplotlib.pyplot" not in sys.modules


def test_draw_dag_depths_alignment_and_deep_deps():
    """Verify that compute_dag_depths aligns same-stage tasks and pushes deeper dependencies."""
    import networkx as nx
    from experimaestro.tools.xpm_dag.draw_dag import compute_dag_depths

    def make_node(name: str):
        return type(name, (), {})()

    full_jobs = {
        "dataset": make_node("MNISTLabelledImages"),
        "extra_prep": make_node("PreprocessTask"),
        "deep_step": make_node("DeepHelperTask"),
        "learn_shallow": make_node("Learn"),
        "learn_deep": make_node("Learn"),
        "eval_shallow": make_node("Evaluate"),
        "eval_deep": make_node("Evaluate"),
    }

    dag = nx.DiGraph()
    for jid in full_jobs:
        dag.add_node(jid)

    # Shallow pipeline: dataset -> learn_shallow -> eval_shallow
    dag.add_edge("dataset", "learn_shallow")
    dag.add_edge("learn_shallow", "eval_shallow")

    # Deep pipeline: dataset -> extra_prep -> deep_step -> learn_deep -> eval_deep
    dag.add_edge("dataset", "extra_prep")
    dag.add_edge("extra_prep", "deep_step")
    dag.add_edge("deep_step", "learn_deep")
    dag.add_edge("learn_deep", "eval_deep")

    depths = compute_dag_depths(
        dag,
        full_jobs,
        align_types=True,
        aligned_class_names={"Learn", "Evaluate"},
    )

    assert depths["learn_shallow"] == depths["learn_deep"] == 3
    assert depths["eval_shallow"] >= 4
    assert depths["eval_deep"] >= 4

    for u, v in dag.edges():
        assert depths[v] > depths[u], f"Edge {u} -> {v} violated monotonicity"


def test_draw_dag_filter_helpers(tmp_path):
    """Verify tag parsing, config listing, and subgraph extraction."""
    from experimaestro.tools.xpm_dag.draw_dag import (
        filter_subgraph_by_config,
        parse_filter_arg,
    )

    # 1. Test parse_filter_arg
    parsed = parse_filter_arg("lr=0.001,kernel_size=3,name=cnn,flag=true")
    assert parsed == {"lr": 0.001, "kernel_size": 3, "name": "cnn", "flag": "true"}

    # 2. Test filter_subgraph_by_config
    class MockObj:
        pass

    full_jobs = {"root": MockObj(), "j1": MockObj(), "j2": MockObj(), "j3": MockObj()}
    dependencies = {
        "j1": ["root"],
        "j2": ["root", "j1"],
        "j3": ["root"],
    }
    job_tags_dict = {
        "j1": {"lr": 0.01, "layers": 2},
        "j2": {"lr": 0.01, "layers": 2},
        "j3": {"lr": 0.001, "layers": 3},
    }

    filtered_jobs, filtered_deps = filter_subgraph_by_config(
        full_jobs,
        dependencies,
        job_tags_dict,
        target_tags={"lr": 0.01, "layers": 2},
    )

    assert set(filtered_jobs.keys()) == {"root", "j1", "j2"}
    assert "j3" not in filtered_jobs
    assert filtered_deps["j1"] == ["root"]
    assert filtered_deps["j2"] == ["root", "j1"]


def test_resolve_run_dir_error_reporting(tmp_path, caplog):
    """Verify that resolve_run_dir provides helpful error messages listing available experiments and runs."""
    from experimaestro.tools.xpm_dag.xpm_dag import resolve_run_dir

    ws_dir = tmp_path / "test_ws"
    exp_dir = ws_dir / "experiments"
    (exp_dir / "exp_alpha" / "20260101_120000").mkdir(parents=True)
    (exp_dir / "exp_beta" / "20260102_150000").mkdir(parents=True)

    with caplog.at_level(logging.ERROR):
        # 1. Non-existent experiment
        res = resolve_run_dir(str(ws_dir), "exp_gamma", None)
        assert res is None
        assert "Experiment 'exp_gamma' not found" in caplog.text
        assert "'exp_alpha'" in caplog.text and "'exp_beta'" in caplog.text

        caplog.clear()

        # 2. Non-existent run
        res = resolve_run_dir(str(ws_dir), "exp_alpha", "wrong_run_id")
        assert res is None
        assert "Run 'wrong_run_id' not found" in caplog.text
        assert "'20260101_120000'" in caplog.text


def test_resolve_run_dir_remote_ssh(monkeypatch, tmp_path):
    """Verify that resolve_run_dir delegates to remote SSH fetcher when workspace is remote."""
    from pathlib import Path
    from experimaestro.settings import SSHSettings, WorkspaceSettings
    from experimaestro.tools.xpm_dag.xpm_dag import resolve_run_dir

    ssh = SSHSettings(host="cluster.example.org")
    ws = WorkspaceSettings(
        id="cluster_ws",
        path=Path("/remote/workspace"),
        ssh=ssh,
    )

    fake_run_dir = tmp_path / "cached_run"
    fake_run_dir.mkdir()

    called = []

    def mock_fetch(remote_ws, exp_id, r_id):
        called.append((remote_ws.id, exp_id, r_id))
        return "20260101_120000", fake_run_dir

    monkeypatch.setattr(
        "experimaestro.tools.xpm_dag.xpm_dag.find_workspace",
        lambda **kwargs: ws,
    )
    monkeypatch.setattr(
        "experimaestro.utils.remote.find_remote_workspace_experiment_run",
        mock_fetch,
    )

    resolved_id, run_d = resolve_run_dir(
        "cluster_ws", "my_remote_exp", "20260101_120000"
    )
    assert resolved_id == "20260101_120000"
    assert run_d == fake_run_dir
    assert called == [("cluster_ws", "my_remote_exp", "20260101_120000")]


def test_load_run_graph_without_importing_remote_modules(tmp_path):
    """Verify load_run_graph builds the DAG from JSONL without importing remote modules or files."""
    from experimaestro import Action, Config, Task
    from experimaestro.tools.xpm_dag.xpm_dag import load_run_graph

    run_dir = tmp_path / "remote_run"
    run_dir.mkdir()

    cfg_id = "a" * 64
    task_id = "b" * 64
    out_cfg_id = "c" * 64
    eval_task_id = "d" * 64
    action_id = "e" * 64

    objects_data = [
        {
            "id": 1,
            "module": "non_existent_remote_pkg.models",
            "type": "RemoteEncoderConfig",
            "typename": "non_existent_remote_pkg.models.RemoteEncoderConfig",
            "identifier": cfg_id,
            "file": "/lustre/fsn1/projects/rech/cdt/uqo57sk/code/cross-encoders/src/tests.py",
            "fields": {
                "hidden_dim": 768,
                "lr": 0.0001,
                "mode": {
                    "type": "enum",
                    "module": "non_existent_remote_pkg.enums",
                    "enum": "Mode",
                    "value": "FAST",
                },
                "data_path": {
                    "type": "path",
                    "base": "workspace",
                    "value": "data/corpus",
                },
                "generated_listener_path": {
                    "type": "path",
                    "value": f"/lustre/ws/jobs/non_existent_remote_pkg.tasks.learn/{task_id}/out/listener",
                },
            },
        },
        {
            "id": 2,
            "module": "non_existent_remote_pkg.tasks",
            "type": "Learn",
            "typename": "non_existent_remote_pkg.tasks.Learn",
            "identifier": task_id,
            "file": "/lustre/fsn1/projects/rech/cdt/uqo57sk/code/cross-encoders/src/training_utils.py",
            "fields": {
                "encoder": {"type": "python", "value": 1},
            },
        },
        {
            "id": 3,
            "module": "non_existent_remote_pkg.models",
            "type": "ParameterLoader",
            "typename": "non_existent_remote_pkg.models.ParameterLoader",
            "identifier": out_cfg_id,
            "task": 2,
            "fields": {},
        },
        {
            "id": 4,
            "module": "tests",
            "type": "Evaluate",
            "typename": "tests.Evaluate",
            "identifier": eval_task_id,
            "file": "/lustre/fsn1/projects/rech/cdt/uqo57sk/code/cross-encoders/src/tests.py",
            "init-tasks": [3],
            "fields": {
                "model": {"type": "python", "value": 1},
            },
        },
        {
            "id": 5,
            "module": "non_existent_remote_pkg.actions",
            "type": "ExportHuggingFace",
            "typename": "non_existent_remote_pkg.actions.ExportHuggingFace",
            "identifier": action_id,
            "fields": {
                "task": {"type": "python", "value": 2},
            },
        },
    ]

    with (run_dir / "objects.jsonl").open("w") as f:
        for obj in objects_data:
            f.write(json.dumps(obj) + "\n")

    jobs_data = [
        {
            "job_id": task_id,
            "task_id": "non_existent_remote_pkg.tasks.learn",
            "tags": {"lr": 0.0001, "hidden_dim": 768},
        },
        {
            "job_id": eval_task_id,
            "task_id": "tests.evaluate",
            "tags": {"lr": 0.0001, "hidden_dim": 768},
        },
    ]
    with (run_dir / "jobs.jsonl").open("w") as f:
        for j in jobs_data:
            f.write(json.dumps(j) + "\n")

    status_data = {
        "job_states": {task_id: "done", eval_task_id: "running"},
        "actions": {action_id: {"description": "Export model"}},
    }
    (run_dir / "status.json").write_text(json.dumps(status_data))

    res = load_run_graph(run_dir)
    assert res is not None
    full_jobs, dependencies, job_tags_dict = res

    assert set(full_jobs.keys()) == {
        cfg_id,
        task_id,
        out_cfg_id,
        eval_task_id,
        action_id,
    }
    assert isinstance(full_jobs[cfg_id], Config)
    assert not isinstance(full_jobs[cfg_id], (Task, Action))
    assert isinstance(full_jobs[task_id], Task)
    assert isinstance(full_jobs[eval_task_id], Task)
    assert isinstance(full_jobs[action_id], Action)

    assert type(full_jobs[cfg_id]).__name__ == "RemoteEncoderConfig"
    assert type(full_jobs[task_id]).__name__ == "Learn"
    assert type(full_jobs[action_id]).__name__ == "ExportHuggingFace"

    assert cfg_id in dependencies[task_id]
    assert task_id not in dependencies.get(cfg_id, [])
    assert task_id in dependencies[out_cfg_id]
    assert out_cfg_id in dependencies[eval_task_id]
    assert cfg_id in dependencies[eval_task_id]
    assert task_id in dependencies[action_id]

    assert job_tags_dict[cfg_id] == {"hidden_dim": 768, "lr": 0.0001}
    assert job_tags_dict[task_id] == {"hidden_dim": 768, "lr": 0.0001}

    dag_json = build_dag_json(full_jobs, dependencies, job_tags_dict, run_dir=run_dir)
    assert len(dag_json["nodes"]) == 5
