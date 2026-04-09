# Utility Functions

This page documents utility functions available as a Python API.

## Copying Experiments

The {py:mod}`experimaestro.copy_experiment` module provides functions to copy
experiments between workspaces (local or remote via SSH). This is the same
logic used by the [`experiments copy`](cli.md#copy-command) CLI command.

### Quick Start

```python
from experimaestro.copy_experiment import copy_experiment
from experimaestro.settings import get_workspace

src = get_workspace("cluster", include_remote=True)
dst = get_workspace("local")
result = copy_experiment(src, dst, "my_experiment", "20260121_105710")
print(f"Copied {result.jobs_copied} jobs, skipped {result.jobs_skipped}")
if result.errors:
    print(f"Errors: {result.errors}")
```

### What Gets Copied

- The experiment run directory (`experiments/<id>/<run_id>/`) — metadata, events, configs
- All job directories referenced in `jobs.jsonl` — outputs, logs, params
- Jobs that already exist at the destination are skipped

### Path Rewriting

After copying, workspace-dependent paths in each job's `params.json` are
automatically rewritten to point to the destination workspace. This includes:

- The top-level `"workspace"` field
- All `{"type": "path", ...}` and `{"type": "path.serialized", ...}` entries
  within serialized objects

### Atomic Staging

Each job is first rsynced into a temporary staging directory (`.xpm-staging/`)
inside the destination workspace, then moved to its final location. This prevents
partial copies from leaving broken state.

### API Reference

```{eval-rst}
.. autoclass:: experimaestro.copy_experiment.CopyResult
   :members:

.. autofunction:: experimaestro.copy_experiment.copy_experiment

.. autofunction:: experimaestro.copy_experiment.list_experiments

.. autofunction:: experimaestro.copy_experiment.list_runs

.. autofunction:: experimaestro.copy_experiment.resolve_current_run

.. autofunction:: experimaestro.copy_experiment.format_run_id

.. autofunction:: experimaestro.copy_experiment.read_jobs_jsonl

.. autofunction:: experimaestro.copy_experiment.resolve_workspace_path
```
