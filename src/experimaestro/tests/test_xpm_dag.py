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
    assert len(server_called) == 0, "Server should not be launched when --save-as ends in .html"

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

