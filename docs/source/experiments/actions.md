# Experiment Actions

!!! warning "Alpha Feature"
    Actions are currently in **alpha**. The API may change in future releases.

Actions are user-defined operations that can be registered during experiment
submission and executed after completion. They enable post-experiment workflows
such as exporting models to HF Hub, saving results to specific folders, or
running evaluation pipelines.

## Defining an Action

Actions are `Config` subclasses that implement `describe()` and `execute()`:

```python
from experimaestro import Action, Interaction, Param

class ExportToHub(Action):
    model: Param[Model]
    default_name: Param[str] = field(default="my-model", ignore_default=True)

    def describe(self) -> str:
        return "Export model to Hugging Face Hub"

    def execute(self, interaction: Interaction) -> None:
        target = interaction.choice("target", "Export to:", ["HF Hub", "Local folder"])
        if target == "HF Hub":
            name = interaction.text("name", "Hub model name:", default=self.default_name)
            private = interaction.checkbox("private", "Private repo?", default=False)
            self.model.push_to_hub(name, private=private)
        else:
            folder = interaction.text("folder", "Output folder:")
            self.model.save(Path(folder))
```

## Registering Actions

Actions are registered during task submission via the `__submit__` method.
The `add_action` callable is passed as a parameter:

```python
from experimaestro import Task, Param

class TrainModel(Task):
    model: Param[Model]
    epochs: Param[int]

    def __submit__(self, dep, add_action, **kwargs):
        # Register action for post-experiment use
        add_action(ExportToHub(model=self.model))
        return self  # or dep(some_output_config)

    def execute(self):
        # Training logic
        pass
```

### Migration from `task_outputs`

The `__submit__` method replaces `task_outputs`. Existing `task_outputs` methods
continue to work — the default `__submit__` calls them automatically. To migrate:

```python
# Before (still works):
class MyTask(Task):
    def task_outputs(self, dep):
        return dep(OutputConfig(...))

# After (new style):
class MyTask(Task):
    def __submit__(self, dep, add_action, **kwargs):
        add_action(MyAction(...))
        return dep(OutputConfig(...))
```

## Executing Actions

### CLI

```bash
# List actions for an experiment
experimaestro experiments actions list my-experiment

# Run an action interactively
experimaestro experiments actions run my-experiment ExportToHub

# Pre-fill answers to skip prompts
experimaestro experiments actions run my-experiment ExportToHub \
    --set target="HF Hub" --set name="my-model" --set private=true
```

### TUI

In the experiment monitor TUI, select an experiment and navigate to the
**Actions** tab to see registered actions. Press `Enter` on a selected
action to execute it interactively.

## Interaction Protocol

The `Interaction` interface provides three question types:

| Method | Description | Pre-fill format |
|--------|-------------|-----------------|
| `choice(key, label, choices)` | Select from list | Exact choice string |
| `checkbox(key, label, default)` | Yes/no question | `true`/`false`/`yes`/`no` |
| `text(key, label, default)` | Free text input | Any string |

Each question has a `key` used for pre-filling answers via CLI `--set key=value`.

### Implementing custom Interaction backends

```python
from experimaestro import Interaction

class WebInteraction(Interaction):
    def choice(self, key, label, choices):
        # Show web form, wait for response
        ...

    def checkbox(self, key, label, *, default=False):
        ...

    def text(self, key, label, *, default=""):
        ...
```

## Serialization

Actions are serialized to `objects.jsonl` in the experiment run directory
alongside job configs. They are available immediately after registration
(no need to wait for experiment completion). The `load_xp_info()` function
loads both jobs and actions:

```python
from experimaestro import load_xp_info

info = load_xp_info("/path/to/workspace/experiments/my-experiment/20260325_120000")
print(info.jobs)     # dict[str, Config]
print(info.actions)  # dict[str, Action]
```
