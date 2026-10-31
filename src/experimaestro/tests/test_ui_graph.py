"""Experiment dependency graph (experimaestro.ui.graph)"""

from functools import partial
import json
from pathlib import Path
import time

import pytest

from experimaestro import (
    Config,
    Meta,
    Param,
    PathGenerator,
    ResumableTask,
    Task,
    field,
)
from experimaestro.scheduler.interfaces import ExperimentJobInformation
from experimaestro.tests.tasks.test_dynamic import Model, Validation, evaluate
from experimaestro.tests.utils import (
    TemporaryDirectory,
    TemporaryExperiment,
    wait_for,
)
from experimaestro.ui.graph import (
    NodeKind,
    build_experiment_graph,
    detect_grid_configs,
    load_experiment_graph,
    run_progress,
    standalone_html,
    transitive_reduction,
)

# --- Synthetic runs: a learner per learning rate, evaluated on two datasets


def h(name: str) -> str:
    """A job identifier (64 hexadecimal characters) for a readable name"""
    return name.encode().hex().ljust(64, "0")[:64]


def ref(oid: int) -> dict:
    return {"type": "python", "value": oid}


def job_dir(task_id: str, name: str) -> str:
    return f"/workspace/jobs/{task_id}/{h(name)}"


def obj(oid: int, type_: str, name: str, fields: dict, **extra) -> dict:
    return {
        "id": oid,
        "module": "pkg",
        "type": type_,
        "identifier": h(name),
        "fields": fields,
        **extra,
    }


def sweep_run():
    """Objects and jobs of a run with 2 learners and 4 evaluations"""
    objects = [obj(1, "Model", "model", {"size": 3})]
    jobs = {}
    oid = 2
    for lr in (0.1, 0.01):
        learn, trained = oid, oid + 1
        objects += [
            obj(
                learn,
                "Learn",
                f"learn-{lr}",
                {
                    "model": ref(1),
                    "lr": lr,
                    # Generated path within the job directory (self-reference)
                    "out": {
                        "type": "path",
                        "value": job_dir("pkg.learn", f"learn-{lr}") + "/model.pt",
                    },
                },
            ),
            # Output of the learner
            obj(trained, "Trained", f"trained-{lr}", {"model": ref(1)}, task=learn),
        ]
        jobs[h(f"learn-{lr}")] = ExperimentJobInformation(
            h(f"learn-{lr}"), "pkg.learn", {"lr": lr}
        )
        oid += 2
        for dataset in ("a", "b"):
            name = f"eval-{lr}-{dataset}"
            objects.append(
                obj(
                    oid, "Evaluate", name, {"trained": ref(trained), "dataset": dataset}
                )
            )
            jobs[h(name)] = ExperimentJobInformation(
                h(name), "pkg.evaluate", {"lr": lr, "dataset": dataset}
            )
            oid += 1
    return objects, jobs


@pytest.fixture
def graph():
    objects, jobs = sweep_run()
    return build_experiment_graph("sweep", "run-1", objects, jobs, set())


def test_nodes(graph):
    kinds = {n.name: n.kind for n in graph.nodes.values()}
    assert kinds == {
        "Model": NodeKind.CONFIG,
        "Learn": NodeKind.TASK,
        "Trained": NodeKind.CONFIG,
        "Evaluate": NodeKind.TASK,
    }

    learn = graph.nodes[h("learn-0.1")]
    assert learn.depends_on == [h("model")]
    assert learn.task_id == "pkg.learn"
    assert learn.tags == {"lr": 0.1}
    # Output configurations depend on the task that produces them
    assert set(graph.nodes[h("trained-0.1")].depends_on) == {
        h("model"),
        h("learn-0.1"),
    }
    assert graph.nodes[h("eval-0.1-a")].depends_on == [h("trained-0.1")]


def test_job_dependencies(graph):
    # Through the trained model configuration
    deps = graph.job_dependencies()
    assert deps[h("eval-0.1-a")] == [h("learn-0.1")]
    assert deps[h("learn-0.1")] == []


def test_generated_path_cycle():
    """A configuration within a task, with a path in the task directory"""
    objects = [
        obj(
            1,
            "Listener",
            "listener",
            {"path": {"type": "path", "value": job_dir("pkg.learn", "learn") + "/x"}},
        ),
        obj(2, "Learn", "learn", {"listener": ref(1)}),
    ]
    jobs = {h("learn"): ExperimentJobInformation(h("learn"), "pkg.learn")}
    graph = build_experiment_graph("xp", "run", objects, jobs, set())
    assert graph.nodes[h("learn")].depends_on == [h("listener")]
    assert graph.nodes[h("listener")].depends_on == []


def test_grid_configs(graph):
    # The learning rate is shared by the learners and their evaluations
    assert graph.configs == [{"lr": 0.1}, {"lr": 0.01}]
    assert detect_grid_configs([("t", {"x": 1}), ("t", {"x": 2}), ("t", {"x": 1})]) == [
        {"x": 1},
        {"x": 2},
    ]


def members(view) -> set[frozenset[str]]:
    return {frozenset(n.members) for n in view.nodes}


def test_view_all(graph):
    view = graph.view(stack=False)
    assert len(view.nodes) == len(graph.nodes)
    # Redundant edges are removed (model -> trained goes through learn)
    assert (h("model"), h("trained-0.1")) not in view.edges
    assert (h("learn-0.1"), h("trained-0.1")) in view.edges


def test_view_stacked_jobs(graph):
    view = graph.view(tasks_only=True)
    assert members(view) == {
        frozenset({h("learn-0.1")}),
        frozenset({h("learn-0.01")}),
        frozenset({h("eval-0.1-a"), h("eval-0.1-b")}),
        frozenset({h("eval-0.01-a"), h("eval-0.01-b")}),
    }
    stack = next(n for n in view.nodes if len(n.members) == 2)
    assert stack.tags["dataset"] == "<2 values>"
    assert sorted(view.edges) == sorted(
        [
            (h("learn-0.1"), min(h("eval-0.1-a"), h("eval-0.1-b"))),
            (h("learn-0.01"), min(h("eval-0.01-a"), h("eval-0.01-b"))),
        ]
    )


def test_view_config(graph):
    view = graph.view(1, stack=False)
    assert members(view) == {
        frozenset({n})
        for n in [
            h("model"),
            h("learn-0.01"),
            h("trained-0.01"),
            h("eval-0.01-a"),
            h("eval-0.01-b"),
        ]
    }
    # Selecting by tag values, matching only a downstream stage
    view = graph.view({"dataset": "a"}, tasks_only=True, stack=False)
    assert members(view) == {
        frozenset({n})
        for n in [h("learn-0.1"), h("learn-0.01"), h("eval-0.1-a"), h("eval-0.01-a")]
    }


def test_views(graph):
    views = graph.views()
    assert set(views) == {
        f"{config}:{tasks_only}:{stack}"
        for config in ("all", 0, 1)
        for tasks_only in (0, 1)
        for stack in (0, 1)
    }


def test_transitive_reduction():
    deps = {"a": [], "b": ["a"], "c": ["a", "b"], "d": ["c", "a", "x"]}
    assert sorted(transitive_reduction(deps)) == [("a", "b"), ("b", "c"), ("c", "d")]
    # Cycles: all edges are kept
    assert sorted(transitive_reduction({"a": ["b"], "b": ["a"]})) == [
        ("a", "b"),
        ("b", "a"),
    ]


def test_standalone_html(graph):
    graph.nodes[h("learn-0.1")].tags["note"] = "</script><b>"
    page = standalone_html(
        graph,
        {h("learn-0.1"): "done", h("learn-0.2"): "running"},
        {h("learn-0.2"): 0.5},
    )
    data = page.split('<script id="xpm-graph" type="application/json">')[1]
    data = data.split("</script>")[0]
    state = json.loads(data)
    assert state["statuses"] == {h("learn-0.1"): "done", h("learn-0.2"): "running"}
    assert state["progress"] == {h("learn-0.2"): 0.5}
    assert set(state["views"]) == set(graph.views())
    assert "</script><b>" not in page


def test_run_progress():
    """The top level progress of the running jobs that report one"""
    from experimaestro.notifications import LevelInformation
    from experimaestro.scheduler.state_provider import MockJob

    def job(name: str, state: str, progress: list) -> MockJob:
        return MockJob(
            identifier=h(name),
            task_id="pkg.Task",
            path=None,
            state=state,
            starttime=None,
            endtime=None,
            progress=progress,
            updated_at="",
        )

    jobs = [
        job("running", "running", [LevelInformation(0, None, 0.25)]),
        job("serialized", "running", [{"level": 0, "progress": 0.5, "desc": None}]),
        job("silent", "running", []),
        job("done", "done", [LevelInformation(0, None, 1.0)]),
    ]

    class Provider:
        def get_jobs(self, experiment_id, run_id=None):
            return jobs

    assert run_progress(Provider(), "xp") == {h("running"): 0.25, h("serialized"): 0.5}


# --- Through a workspace


class Encoder(Config):
    dim: Param[int]


class Train(Task):
    encoder: Param[Encoder]
    checkpoint: Meta[Path] = field(default_factory=PathGenerator("checkpoint"))

    def task_outputs(self, dep):
        return dep(Trained.C(encoder=self.encoder, path=self.checkpoint))

    def execute(self):
        pass


class Trained(Config):
    encoder: Param[Encoder]
    path: Param[Path]


class Score(Task):
    trained: Param[Trained]

    def execute(self):
        pass


@pytest.mark.tasks
def test_load_from_workspace():
    from experimaestro.scheduler.workspace_state_provider import (
        WorkspaceStateProvider,
    )

    with TemporaryDirectory(prefix="xpm") as workdir:
        with TemporaryExperiment("graph", workdir=workdir):
            trained = Train.C(encoder=Encoder.C(dim=4)).tag("dim", 4).submit()
            Score.C(trained=trained).tag("dim", 4).submit()

        provider = WorkspaceStateProvider(workdir, no_cleanup=True)
        try:
            graph = load_experiment_graph(provider, "graph")
        finally:
            provider.close()

    by_name = {n.name: n for n in graph.nodes.values()}
    assert set(by_name) == {"Encoder", "Train", "Trained", "Score"}
    assert by_name["Score"].kind == NodeKind.TASK
    assert by_name["Trained"].kind == NodeKind.CONFIG
    assert graph.job_dependencies()[by_name["Score"].id] == [by_name["Train"].id]
    assert by_name["Encoder"].tags == {"dim": 4}


class WaitingLearn(ResumableTask):
    """Outputs a checkpoint when told to, then runs until released"""

    model: Param[Model]
    validation: Param[Validation]
    control: Meta[Path]

    def execute(self):
        (self.control / "started").touch()
        while not (self.control / "checkpoint").exists():
            time.sleep(0.05)
        self.validation.compute(15)
        while not (self.control / "release").exists():
            time.sleep(0.05)


def reachable(view, source: str) -> set[str]:
    """Nodes reachable from a node through the edges of a view"""
    found, queue = set(), [source]
    while queue:
        node = queue.pop()
        for s, t in view.edges:
            if s == node and t not in found:
                found.add(t)
                queue.append(t)
    return found


@pytest.mark.dynamic
def test_graph_of_running_experiment(tmp_path):
    """Jobs submitted on the outputs of a running task depend on it in the
    graph a monitor reads while the experiment runs"""
    from experimaestro.scheduler.workspace_state_provider import (
        WorkspaceStateProvider,
    )

    workdir, control = tmp_path / "workdir", tmp_path / "control"
    workdir.mkdir()
    control.mkdir()
    evaluations = []
    provider = WorkspaceStateProvider(workdir, no_cleanup=True)
    try:
        with TemporaryExperiment(
            "running-graph", workdir=workdir, timeout_multiplier=6
        ):
            model = Model.C()
            validation = Validation.C(model=model)
            learn = WaitingLearn.C(model=model, validation=validation, control=control)
            learn.watch_output(validation.checkpoint, partial(evaluate, evaluations))
            learn.submit()
            try:
                learn_id = learn.__xpm__.job.identifier
                wait_for((control / "started").exists, "learner started")
                # The monitor shows the graph before the checkpoint
                assert (
                    learn_id in load_experiment_graph(provider, "running-graph").nodes
                )

                (control / "checkpoint").touch()
                wait_for(lambda: evaluations, "evaluation submitted")
                eval_id = evaluations[0].__xpm__.job.identifier
                wait_for(
                    lambda: (
                        eval_id in provider.get_experiment_job_info("running-graph")
                    ),
                    "evaluation seen by the monitor",
                )
                graph = load_experiment_graph(provider, "running-graph")
            finally:
                (control / "release").touch()
    finally:
        provider.close()

    assert eval_id in graph.nodes
    # In the jobs-only views, the evaluation comes after the learner (through
    # the init task loading the checkpoint)
    for stack in (False, True):
        view = graph.view(tasks_only=True, stack=stack)
        assert eval_id in reachable(view, learn_id), f"{view.edges=} ({stack=})"
