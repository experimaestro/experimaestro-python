# Analyzing Experiment Results

Experimaestro automatically saves all job configurations at the end of each
experiment run, preserving shared object references. This makes it easy to load
and analyze past results without manually tracking which configs were used.

## How it works

When an experiment completes (or fails), all job configs are serialized together
into `configs.json` in the run directory. Unlike individual `params.json` files
in each job directory, this single file preserves shared references between
configs -- if two tasks share the same sub-config, loading them back gives you
the same Python object.

Tags set on configs (via `.tag()`) are also stored and restored on load.

## Loading configs from a past experiment

### Standalone function (simplest)

The easiest way to load configs is the standalone `load_configs()` function.
Just point it at a run directory (or directly at `configs.json`):

```python
from experimaestro import load_configs

# From a run directory
configs = load_configs("/path/to/workspace/experiments/my-experiment/20260319_120000")

# Or directly from configs.json
configs = load_configs("/path/to/configs.json")

for job_id, config in configs.items():
    print(f"Job {job_id}: {config}")
    print(f"  Tags: {config.tags()}")
```

### Via `WorkspaceStateProvider`

When you need to look up experiments by name (without knowing the run directory),
use `WorkspaceStateProvider`:

```python
from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider

provider = WorkspaceStateProvider(workspace_path)

# Loads from the latest run
configs = provider.load_configs("my-experiment")

# Or specify a run_id
configs = provider.load_configs("my-experiment", run_id="20260319_120000")
```

The provider also gives access to `get_tags_map()` for combining tags with
configs when building DataFrames.

## Building a DataFrame for analysis

Combine loaded configs with tags to build a pandas DataFrame:

```python
from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider
import pandas as pd

provider = WorkspaceStateProvider(workspace_path)
configs = provider.load_configs("my-experiment")
tags_map = provider.get_tags_map("my-experiment")

rows = []
for job_id, config in configs.items():
    row = tags_map.get(job_id, {}).copy()
    row["job_id"] = job_id
    # Extract values from config
    row["learning_rate"] = config.learning_rate
    row["model_name"] = config.model.name
    rows.append(row)

df = pd.DataFrame(rows)
print(df)
```

Since tags are also restored on the configs themselves, you can also use
`config.tags()` directly:

```python
from experimaestro import load_configs
import pandas as pd

configs = load_configs("/path/to/run_dir")

rows = []
for job_id, config in configs.items():
    tags = config.tags()
    rows.append({
        "job_id": job_id,
        "lr": float(tags["lr"]),
        "model": config.model.name,
    })

df = pd.DataFrame(rows)
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
from experimaestro import load_configs
import pandas as pd

configs = load_configs(workdir / "experiments" / "grid-search" / "20260319_120000")

rows = []
for job_id, config in configs.items():
    tags = config.tags()
    rows.append({
        "lr": tags["learning_rate"],
        "hidden_size": tags["hidden_size"],
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

- `configs.json` is only created in `NORMAL` run mode (not dry-run)
- If serialization fails (e.g., unsupported types), a warning is logged but the
  experiment still completes normally
- Shared references are preserved: if two tasks share the same sub-config object,
  `load_configs()` returns configs where the sub-config is the same Python object
  (not just equal)
