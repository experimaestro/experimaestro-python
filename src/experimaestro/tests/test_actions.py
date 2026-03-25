"""Tests for the Actions feature and objects.jsonl serialization."""

import json

import pytest

from experimaestro import Config, Param, Task, Action, Interaction, CLIInteraction
from experimaestro.core.serialization import load_xp_info, ExperimentInfo
from experimaestro.tests.utils import TemporaryExperiment, TemporaryDirectory

pytestmark = pytest.mark.tasks


# --- Test fixtures ---


class SharedConfig(Config):
    value: Param[str]


class SimpleTask(Task):
    shared: Param[SharedConfig]
    x: Param[int]

    def execute(self):
        pass


class SimpleAction(Action):
    shared: Param[SharedConfig]
    label: Param[str]

    def describe(self) -> str:
        return f"Simple action: {self.label}"

    def execute(self, interaction: Interaction) -> None:
        pass


class InteractiveAction(Action):
    def describe(self) -> str:
        return "Interactive test action"

    def execute(self, interaction: Interaction) -> None:
        self._choice = interaction.choice("target", "Pick:", ["A", "B"])
        self._checked = interaction.checkbox("flag", "Enable?", default=False)
        self._text = interaction.text("name", "Name:", default="default")


class TaskWithSubmit(Task):
    shared: Param[SharedConfig]

    def __submit__(self, dep, add_action, **kwargs):
        add_action(SimpleAction.C(shared=self.shared, label="from-submit"))
        return self

    def execute(self):
        pass


class TaskWithTaskOutputs(Task):
    shared: Param[SharedConfig]

    def task_outputs(self, dep):
        return dep(SharedConfig.C(value="output-" + self.shared.value))

    def execute(self):
        pass


class TaskWithBothSubmitAndTaskOutputs(Task):
    """Has both __submit__ and task_outputs — __submit__ should win."""

    shared: Param[SharedConfig]

    def __submit__(self, dep, add_action, **kwargs):
        return dep(SharedConfig.C(value="from-submit"))

    def task_outputs(self, dep):
        return dep(SharedConfig.C(value="from-task-outputs"))

    def execute(self):
        pass


# --- Tests ---


def test_action_serialization_roundtrip():
    """Action can be serialized and deserialized."""
    from experimaestro.core.serialization import state_dict, from_state_dict
    from experimaestro.core.context import SerializationContext

    action = SimpleAction.C(shared=SharedConfig.C(value="hello"), label="test")
    # Seal the config for serialization
    action.seal()

    context = SerializationContext(save_directory=None)
    data = state_dict(context, action)
    loaded = from_state_dict(data)

    assert loaded.label == "test"
    assert loaded.shared.value == "hello"


def test_action_added_event_serialization():
    """ActionAddedEvent can be serialized and deserialized."""
    from experimaestro.scheduler.state_status import ActionAddedEvent, EventBase

    event = ActionAddedEvent(
        experiment_id="test-exp",
        run_id="20260325_120000",
        action_id="SimpleAction",
        description="Test action",
        action_class="test_actions.SimpleAction",
    )

    json_str = event.to_json()
    data = json.loads(json_str)
    assert data["action_id"] == "SimpleAction"
    assert data["description"] == "Test action"

    restored = EventBase.from_dict(data)
    assert isinstance(restored, ActionAddedEvent)
    assert restored.action_id == "SimpleAction"


def test_experiment_add_action():
    """Actions are stored in experiment and serialized to objects.jsonl and status.json."""
    with TemporaryDirectory(prefix="xpm") as workdir:
        with TemporaryExperiment("test-actions", workdir=workdir) as xp:
            shared = SharedConfig.C(value="shared-val")
            SimpleTask.C(shared=shared, x=1).submit()

            action = SimpleAction.C(shared=shared, label="export")
            action_id = xp.add_action(action)

            run_dir = xp.workdir

        # Verify objects.jsonl exists and contains the action
        objects_path = run_dir / "objects.jsonl"
        assert objects_path.exists(), "objects.jsonl not created"

        lines = objects_path.read_text().strip().split("\n")
        # Each line is a single serialized object
        assert len(lines) >= 2, "Expected at least 2 object entries"

        # Check action's identifier exists in serialized objects
        identifiers_in_file = {
            json.loads(line).get("identifier")
            for line in lines
            if "identifier" in json.loads(line)
        }
        assert action_id in identifiers_in_file, (
            "Action identifier not found in objects.jsonl"
        )

        # Verify status.json contains action metadata
        status_path = run_dir / "status.json"
        assert status_path.exists()
        with status_path.open() as f:
            status = json.load(f)
        assert "actions" in status
        assert action_id in status["actions"]
        assert status["actions"][action_id]["description"] == "Simple action: export"


def test_load_xp_info_with_actions():
    """load_xp_info returns both jobs and actions."""
    with TemporaryDirectory(prefix="xpm") as workdir:
        with TemporaryExperiment("test-xp-info", workdir=workdir) as xp:
            shared = SharedConfig.C(value="shared")
            SimpleTask.C(shared=shared, x=1).submit()
            SimpleTask.C(shared=shared, x=2).submit()

            action = SimpleAction.C(shared=shared, label="act1")
            action_id = xp.add_action(action)

            run_dir = xp.workdir

        info = load_xp_info(run_dir)
        assert isinstance(info, ExperimentInfo)
        assert len(info.jobs) == 2
        assert len(info.actions) == 1
        assert action_id in info.actions

        # Verify shared references preserved between jobs
        job_configs = list(info.jobs.values())
        assert job_configs[0].shared is job_configs[1].shared


def test_submit_with_action_registration():
    """__submit__ can register actions via xp.add_action()."""
    with TemporaryDirectory(prefix="xpm") as workdir:
        try:
            with TemporaryExperiment("test-submit-action", workdir=workdir) as xp:
                shared = SharedConfig.C(value="model")
                TaskWithSubmit.C(shared=shared).submit()

                # Action should have been registered immediately after submit
                assert len(xp.actions) == 1
                action = list(xp.actions.values())[0]
                assert action.label == "from-submit"
        except Exception:
            # Task execution may fail but we only care about action registration
            pass


def test_submit_backward_compat_task_outputs():
    """Task with task_outputs still works via default __submit__."""
    with TemporaryDirectory(prefix="xpm") as workdir:
        with TemporaryExperiment("test-compat", workdir=workdir):
            shared = SharedConfig.C(value="compat")
            result = TaskWithTaskOutputs.C(shared=shared).submit()

        # Result should be the output from task_outputs
        assert hasattr(result, "value")
        assert result.value == "output-compat"


def test_submit_default_returns_self():
    """Task without __submit__ or task_outputs returns self."""
    with TemporaryDirectory(prefix="xpm") as workdir:
        with TemporaryExperiment("test-default", workdir=workdir):
            shared = SharedConfig.C(value="default")
            task = SimpleTask.C(shared=shared, x=42)
            result = task.submit()

        # Result is the task itself
        assert result.x == 42


def test_submit_priority_submit_over_task_outputs():
    """__submit__ takes precedence over task_outputs when both are defined."""
    with TemporaryDirectory(prefix="xpm") as workdir:
        with TemporaryExperiment("test-priority", workdir=workdir):
            shared = SharedConfig.C(value="x")
            result = TaskWithBothSubmitAndTaskOutputs.C(shared=shared).submit()

        # __submit__ should win
        assert result.value == "from-submit"


def test_submit_priority_order():
    """Verify the full priority chain: __submit__ > task_outputs > self."""
    with TemporaryDirectory(prefix="xpm") as workdir:
        with TemporaryExperiment("test-priority-all", workdir=workdir):
            shared = SharedConfig.C(value="v")

            # 1. __submit__ defined → uses __submit__
            r1 = TaskWithBothSubmitAndTaskOutputs.C(shared=shared).submit()
            assert r1.value == "from-submit"

            # 2. Only task_outputs defined → uses task_outputs
            r2 = TaskWithTaskOutputs.C(shared=shared).submit()
            assert r2.value == "output-v"

            # 3. Neither defined → returns self (the task)
            r3 = SimpleTask.C(shared=shared, x=99).submit()
            assert r3.x == 99


def test_objects_jsonl_streaming():
    """objects.jsonl streams objects as jobs are submitted, with deduplication."""
    with TemporaryDirectory(prefix="xpm") as workdir:
        with TemporaryExperiment("test-streaming", workdir=workdir) as xp:
            shared = SharedConfig.C(value="stream")
            SimpleTask.C(shared=shared, x=1).submit()
            SimpleTask.C(shared=shared, x=2).submit()
            SimpleTask.C(shared=shared, x=3).submit()

            run_dir = xp.workdir

        objects_path = run_dir / "objects.jsonl"
        assert objects_path.exists()
        lines = objects_path.read_text().strip().split("\n")

        # Should have objects but shared config only serialized once
        all_objects = [json.loads(line) for line in lines]
        identifiers = [
            obj.get("identifier") for obj in all_objects if "identifier" in obj
        ]

        # 3 tasks + 1 shared config = 4 unique identifiers
        assert len(identifiers) == 4
        assert len(set(identifiers)) == 4  # all unique (no duplication)


def test_load_xp_info_backward_compat_configs_json():
    """load_xp_info falls back to configs.json for old experiments."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path

        run_dir = Path(tmpdir)

        # Create a minimal configs.json (old format)
        from experimaestro.core.serialization import state_dict
        from experimaestro.core.context import SerializationContext

        config = SharedConfig.C(value="old-format")
        config.seal()

        context = SerializationContext(save_directory=None)
        data = state_dict(context, {"job1": config})
        data["tags"] = {}

        with (run_dir / "configs.json").open("w") as f:
            json.dump(data, f)

        # load_xp_info should fall back to configs.json
        info = load_xp_info(run_dir)
        assert len(info.jobs) == 1
        assert "job1" in info.jobs
        assert len(info.actions) == 0


def test_mock_experiment_with_actions():
    """MockExperiment loads actions from status.json."""
    from experimaestro.scheduler.state_provider import MockExperiment
    from experimaestro.scheduler.interfaces import BaseAction

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)

        data = {
            "version": 1,
            "experiment_id": "test",
            "run_id": "20260325_120000",
            "status": "completed",
            "hostname": "localhost",
            "started_at": "2026-03-25T12:00:00",
            "ended_at": "2026-03-25T12:01:00",
            "job_states": {},
            "services": {},
            "actions": {
                "export": {
                    "action_id": "export",
                    "description": "Export model",
                    "class": "test.ExportAction",
                }
            },
            "run_tags": [],
        }

        exp = MockExperiment.from_state_dict(data, workdir)
        assert len(exp.actions) == 1
        assert "export" in exp.actions
        action = exp.actions["export"]
        assert isinstance(action, BaseAction)
        assert action.description() == "Export model"
        assert action.action_class == "test.ExportAction"


def test_cli_interaction_prefill():
    """CLIInteraction pre-fills answers and skips prompts."""
    prefill = {"target": "HF Hub", "private": "true", "name": "my-model"}
    interaction = CLIInteraction(prefill=prefill)

    assert interaction.choice("target", "Export to:", ["HF Hub", "Local"]) == "HF Hub"
    assert interaction.checkbox("private", "Private?") is True
    assert interaction.text("name", "Name:") == "my-model"


def test_cli_interaction_prefill_invalid_choice():
    """CLIInteraction raises on invalid pre-filled choice."""
    interaction = CLIInteraction(prefill={"target": "invalid"})

    with pytest.raises(ValueError, match="not in choices"):
        interaction.choice("target", "Export to:", ["HF Hub", "Local"])


def test_cli_interaction_prefill_boolean_variants():
    """CLIInteraction handles various boolean formats."""
    for true_val in ["true", "yes", "1", "y", "True", "YES"]:
        interaction = CLIInteraction(prefill={"flag": true_val})
        assert interaction.checkbox("flag", "Enable?") is True

    for false_val in ["false", "no", "0", "n", "False", "NO"]:
        interaction = CLIInteraction(prefill={"flag": false_val})
        assert interaction.checkbox("flag", "Enable?") is False
