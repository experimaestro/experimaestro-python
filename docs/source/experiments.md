# Running experiments

An experiment is a piece of Python code that submits tasks within the
{py:class}`~experimaestro.experiment` context manager. The context manager owns
the workspace, schedules the submitted jobs, resolves their dependencies, and
serves the monitoring interfaces.

## Two ways to run

|  | {py:class}`~experimaestro.experiment` context manager | `experimaestro run` |
|---|---|---|
| Parameters | Written in Python | Read from `YAML` files |
| Workspace | Given explicitly | Selected from the [settings](settings.md) |
| Best for | Notebooks, one-off scripts, tests | Projects, reproducible runs, parameter sweeps |

Directly, in Python:

```python
from experimaestro import experiment

with experiment("/path/to/workspace", "my-experiment", port=12345) as xp:
    model = Learn.C(epochs=100).submit()
```

or through the command line, where the experiment code is a `run(helper, cfg)`
function and its parameters come from a configuration file:

```sh
experimaestro run full.yaml
```

The second form is the recommended one for anything you intend to re-run — see
[The CLI experiment runner](experiments/cli-runner.md).

## Good practices

### Declare your workspace once

A workspace is where jobs and experiments are stored. Rather than repeating
`--workdir` on every command, declare it in
`~/.config/experimaestro/settings.yaml` and let experimaestro select it:

```yaml
workspaces:
  - id: neuralir
    path: ~/experiments/xpmir
    triggers:
      - "neuralir-*"
```

Any experiment whose ID matches a trigger runs in that workspace; `--workspace
neuralir` selects it explicitly. Keep one workspace per project or per storage
tier — jobs are shared within a workspace, so two projects that share tasks
benefit from sharing a workspace. See [Settings](settings.md) for the full
reference, and [Auxiliary folders](settings.md#auxiliary-folders-beta) when
results must be archived to another filesystem.

(sharing-a-workspace)=
### Sharing a workspace

Several users can share a workspace, so that jobs computed by one are reused by
the others. This only requires that every user can write the files of the
workspace:

```bash
# A shared, group-writable workspace. The setgid bit (2) makes every file and
# directory created below it belong to the group
chgrp -R my_team /scratch/shared
chmod -R g+w /scratch/shared
chmod g+s /scratch/shared

# Each user must also create files with group write permission
umask 002
```

If the filesystem supports POSIX ACLs, a default ACL grants the same
permissions without touching the umask:

```bash
setfacl -R -m g:my_team:rwX /scratch/shared
setfacl -R -d -m g:my_team:rwX /scratch/shared
```

Lock files follow the permissions of the workspace directory by default; see
[Lock file permissions](settings.md#lock-file-permissions) to configure them.

Monitoring a shared workspace is safe: the cleanup that runs when a monitor
starts never touches a job that belongs to somebody else. A job whose process
cannot be checked from the current machine — its PID belongs to another host,
or its launcher (e.g. SLURM) is not reachable — is considered active, and job
files owned by another user are never modified.

### Keep task identifiers stable

A job is stored under a directory derived from a hash of its parameters, which
is what makes results reusable across runs. Renaming a parameter, changing its
type, or renaming a task class changes that hash: previous results become
unreachable and are silently recomputed.

- give a parameter a default with `field(ignore_default=X)` (ignored in the
  identifier) or `field(default=X)` (always part of it), never with a bare
  `x: Param[int] = 23`
- pin a task identifier with `__xpmid__` so that renaming the class is free
- when a parameter must change, keep the old one alive with
  {py:class}`~experimaestro.DeprecatedAttribute`

See [Configurations](experiments/config.md) for the details, and
`experimaestro tools deprecated` to inspect the identifiers affected by a change.

### Running on a cluster

Where a task runs is decided by its [launcher](launchers/index.md), and how the
files are accessed by its [connector](connectors/index.md). A specification
(`cuda(mem=24G) & cpu(cores=4)`) is matched against the launchers declared in
your settings, so the same experiment code runs locally and on a cluster.

Two things are worth setting up early:

- **Limit concurrency** with a token when a resource is scarce — the scheduler
  will hold the jobs back rather than let them fail:

  ```python
  with experiment(workdir, "my-experiment") as xp:
      xp.token = xp.token("main", 4)  # at most 4 jobs at a time
  ```

- **Survive walltime limits** by deriving long tasks from
  {py:class}`~experimaestro.ResumableTask`: check `remaining_time()`, save a
  checkpoint and raise {py:class}`~experimaestro.GracefulTimeout` before the
  limit is reached. Such a task is retried automatically (`max_retries`, 3 by
  default, configurable per workspace or per `submit()`) and keeps its
  directory between attempts. See [Tasks](experiments/task.md).

### Monitoring and recovering

An experiment can be followed live, and a workspace inspected long after the
fact:

```bash
# Terminal UI (add --port to get the web UI instead)
experimaestro --workdir /path/to/workspace monitor --console

# The same, for a workspace on a remote machine
experimaestro -w my-cluster monitor --web
```

Both are described in [Interfaces](interfaces.md). Starting a monitor also
performs a workspace cleanup: event files are consolidated, and jobs whose
process died without leaving a marker are marked as failed, so that the next
experiment does not wait on them. Jobs that are no longer part of any
experiment are reported rather than deleted:

```bash
experimaestro --workdir /path/to/workspace workspace orphans
```

### Iterating without losing work

- `--run-mode DRY_RUN` (and `GENERATE_ONLY`, `PREPARE`) walks the experimental
  plan without executing anything — the fastest way to check what a change
  would submit
- `--show` prints the merged YAML configuration, which usually explains an
  unexpected parameter
- keep `dirty_git: error` on experiments meant to be reproducible, so that a
  result can always be traced back to a commit
- tag the parameters that vary in your plan; the [analysis](experiments/analysis.md)
  tools then group results by tag rather than by directory name

## Experiment services

Experiments can expose long-running services (a TensorBoard instance, a web
application) that the monitoring interfaces start, stop, and display. See
[Services](services.md).

## Experiment metadata

Each run records the host it was launched from, in the workspace database and
in `xp/{experiment_id}/informations.json`. It is preserved by database resyncs
and displayed by the interfaces — useful when experiments of the same workspace
are launched from several machines:

```bash
# Shown in brackets after the experiment ID
experimaestro --workdir /path/to/workspace experiments list
# my-experiment [hostname.local] (5/10 jobs)
```
