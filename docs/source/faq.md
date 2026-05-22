# Frequently Asked Questions

## Platform support

### Running on Windows

Experimaestro creates symbolic links inside the workspace (e.g. to point at the
latest run of an experiment, to track partial dependencies, and for the
``current`` symlink). On Windows, ``os.symlink`` requires either administrator
privileges or **Developer Mode** to be enabled (Settings → Update & Security →
For developers). If neither is available, the scheduler will fail with an
``OSError`` when it tries to create those links.

Enable Developer Mode (recommended) or run the experiment from an elevated
shell. SSH support is currently not functional on any platform and is unrelated
to this requirement.

## Controlling tasks

### Wait a for a task to complete

```python
# Submit the task
output = mytask.submit()

# Wait until it completes, and returns a boolean (success flag)
output.__xpm__.wait()
```

## Data preparation

### How do I pre-download datasets or resources before running on an offline cluster?

Use {py:class}`~experimaestro.Prepare`: a `Config` subclass whose `prepare()`
method is auto-invoked once before any task that references it. Library
helpers like `datamaestro.prepare_dataset(...)` already return `Prepare`
instances, so you usually do not write one by hand.

To pre-warm caches from a connected node (driver / login node) without
launching any actual tasks:

```bash
uv run experimaestro run-experiment --run-mode prepare my_experiment.py
```

This walks every `Prepare` referenced in your submitted tasks' params,
runs each `prepare()` exactly once in the driver process, and exits.
Afterwards the compute nodes can run the experiment without internet
access:

```bash
uv run experimaestro run-experiment my_experiment.py
```

**Where the data is stored.** PREPARE mode creates **no entries under
`workspace/jobs/`** — the only on-disk effect is what `prepare()` itself
writes (typically a tool-managed cache like
`~/.cache/datamaestro/` or `~/.cache/huggingface/`). In NORMAL mode the same
prep runs before each dependent task, plus each task gets its usual
`workspace/jobs/<task-id>/<hash>/` folder for outputs and logs.

See [Prepare configurations](experiments/config.md#prepare-configurations-data-preparation)
for the full design and the [MNIST demo](https://github.com/experimaestro/experimaestro-demo)
for a worked example.

### Can I write my own `Prepare` for things other than datasets?

Yes — anything that needs a one-shot, idempotent setup before tasks run:
HF model downloads, credential fetching, populating a shared scratch
folder, building a Docker layer cache, etc. Subclass `Prepare`, override
`prepare(self)`, and return it from a helper function that experiments
call:

```python
from experimaestro import Prepare, Param

class HFModelPrep(Prepare):
    model_id: Param[str]

    def prepare(self) -> None:
        from huggingface_hub import snapshot_download
        snapshot_download(self.model_id)  # idempotent

def prepare_model(model_id: str) -> HFModelPrep:
    return HFModelPrep.C(model_id=model_id)
```

Any task that takes an `HFModelPrep` in its params will have the download
triggered automatically before it runs.

## Post-experiment workflows

### How do I run code after my experiment finishes?

Use {py:class}`~experimaestro.Action` (alpha). An Action is a Config whose
`execute(interaction)` runs after the experiment completes — for picking
the best model, pushing to HF Hub, copying artefacts to a results folder,
sending Slack notifications, etc.

Register an Action with the experiment during submission:

```python
from experimaestro import Action, Interaction, Param

class CopyResults(Action):
    runs: Param[list[MyEvalTask]]

    def describe(self) -> str:
        return "Copy results to a destination folder"

    def execute(self, interaction: Interaction) -> None:
        dest = interaction.text("dest", "Destination folder:", default="./results")
        # ... copy logic

# in experiment.py
helper.xp.add_action(CopyResults.C(runs=evaluations))
```

Then list and run the registered actions:

```bash
uv run experimaestro experiments actions list <experiment-id>
uv run experimaestro experiments actions run <experiment-id> <action-id>
```

Actions can also be invoked from the TUI / web UI. See
[Experiment Actions](experiments/actions.md) for the full API and the
[MNIST demo](https://github.com/experimaestro/experimaestro-demo) for the
`ExportBestModel` example.

## Debugging

### How to run a task without scheduling it?

Method 1: creates an instance of a task with its `.instance(context)` method. The context contains information which might
be necessary to create the directory structure for executing the task. If no context is provided, the default context
is used.

```python
context = DirectoryContext("/tmp/taskfolder")
task.instance(context).execute()
```

The main problem with this approach is that resources are shared between experimaestro and the task

### How to Debug a failed task ?
If a task failed, you can rerun it with [debugpy](https://github.com/microsoft/debugpy).

#### Using vsCode
If the task is already generated, you can run it with the [python debugger](https://code.visualstudio.com/docs/python/debugging) directly within vsCode.
- open the task python file `.../HASHID/task_name.py`.
- Run the dubugger Using the following configuration:

In `.vscode/launch.json` :
```json5
 {
            "name": "Python: XPM Task",
            "type": "debugpy",
            "request": "launch",
            "module": "experimaestro",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": [
                "run",
                "params.json"
            ],
            // "python": "${workspaceFolder}/.venv/bin/python",
            "cwd": "${fileDirname}",
            "env": {
                "CUDA_VISIBLE_DEVICES": "1",
            }
}
```
- NOTE: if the task needs GPU support, you may need to open VS-Code on a node with access to a GPU.
