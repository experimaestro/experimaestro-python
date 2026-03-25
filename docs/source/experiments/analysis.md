# Analyzing Experiment Results

Experimaestro automatically saves all job configurations as they are submitted,
preserving shared object references. This makes it easy to load and analyze past
results without manually tracking which configs were used.

## How it works

As jobs are submitted during an experiment, their configs are streamed to
`objects.jsonl` in the run directory. Unlike individual `params.json` files
in each job directory, this shared serialization preserves references between
configs -- if two tasks share the same sub-config, loading them back gives you
the same Python object.

## Loading configs from a past experiment

### Standalone function (simplest)

The easiest way to load experiment data is the `load_xp_info()` function.
Point it at a run directory:

```python
from experimaestro import load_xp_info

info = load_xp_info("/path/to/workspace/experiments/my-experiment/20260319_120000")

for job_id, config in info.jobs.items():
    print(f"Job {job_id}: {config}")

# Actions are also available (see experiments/actions.md)
for action_id, action in info.actions.items():
    print(f"Action {action_id}: {action.describe()}")
```

### Via `WorkspaceStateProvider`

When you need to look up experiments by name (without knowing the run directory),
use `WorkspaceStateProvider`:

```python
from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider

provider = WorkspaceStateProvider(workspace_path)

# Loads from the latest run
info = provider.load_xp_info("my-experiment")

# Or specify a run_id
info = provider.load_xp_info("my-experiment", run_id="20260319_120000")
```

The provider also gives access to `get_tags_map()` for combining tags with
configs when building DataFrames.

## Building a DataFrame for analysis

Combine loaded configs with tags to build a pandas DataFrame:

```python
from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider
import pandas as pd

provider = WorkspaceStateProvider(workspace_path)
info = provider.load_xp_info("my-experiment")
tags_map = provider.get_tags_map("my-experiment")

rows = []
for job_id, config in info.jobs.items():
    row = tags_map.get(job_id, {}).copy()
    row["job_id"] = job_id
    # Extract values from config
    row["learning_rate"] = config.learning_rate
    row["model_name"] = config.model.name
    rows.append(row)

df = pd.DataFrame(rows)
print(df)
```

## Full worked example

### Define tasks

```python
from experimaestro import Config, Task, Param, experiment

class Model(Config):
    name: Param[str]
    hidden_size: Param[int]

class Train(Task):
    model: Param[Model]
    learning_rate: Param[float]

    def execute(self):
        # Training logic here
        pass
```

### Run experiment with tags

```python
from experimaestro import experiment, tag

with experiment(workdir, "grid-search") as xp:
    for lr in [1e-3, 1e-4, 1e-5]:
        for size in [128, 256]:
            model = Model.C(name="transformer", hidden_size=tag(size))
            task = Train.C(model=model, learning_rate=tag(lr))
            task.submit()
```

### Analyze in a notebook

```python
from experimaestro import load_xp_info
import pandas as pd

info = load_xp_info(workdir / "experiments" / "grid-search" / "20260319_120000")
tags_map = ...  # from provider.get_tags_map()

rows = []
for job_id, config in info.jobs.items():
    tags = tags_map.get(job_id, {})
    rows.append({
        "lr": tags.get("learning_rate"),
        "hidden_size": tags.get("hidden_size"),
        "model": config.model.name,
    })

df = pd.DataFrame(rows)
print(df)
#       lr  hidden_size       model
# 0  0.001          128  transformer
# 1  0.001          256  transformer
# ...
```

## Explicit save/load

For saving custom objects beyond auto-saved job configs, use `xp.save()` and
`xp.load()` within an experiment context:

```python
# Save during experiment
with experiment(workdir, "my-experiment") as xp:
    xp.save(my_results, name="final-results")

# Load from another experiment
with experiment(workdir, "analysis") as xp:
    results = xp.load("my-experiment", name="final-results")
```

## Notes

- `objects.jsonl` is only created in `NORMAL` run mode (not dry-run)
- If serialization fails (e.g., unsupported types), a warning is logged but the
  experiment still completes normally
- Shared references are preserved: if two tasks share the same sub-config object,
  `load_xp_info()` returns configs where the sub-config is the same Python object
  (not just equal)
- For backward compatibility, experiments created before `objects.jsonl` was introduced
  will fall back to loading from `configs.json`
