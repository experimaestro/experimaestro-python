# Workspace Layout

A workspace is a directory where experimaestro stores all experiment data, job
outputs, and metadata. This page describes the directory structure and explains
the purpose of each component.

## Directory Structure

```
WORKSPACE_DIR/
├── .__experimaestro__              # Marker file indicating this is a workspace
├── experiments/                     # Experiment run directories
│   └── {experiment-id}/            # One directory per experiment
│       ├── lock                    # Lock file (prevents concurrent runs)
│       └── {run-id}/               # One directory per run
│           ├── environment.json    # Python environment and git info
│           ├── status.json         # Experiment metadata and services state
│           ├── jobs.jsonl          # Lightweight job information (one per line)
│           ├── jobs/               # Symlinks to job directories
│           ├── results/            # Saved experiment results
│           └── data/               # Serialized configurations
├── jobs/                           # Actual job output directories
│   └── {task-type-id}/
│       └── {job-hash}/
│           ├── params.json         # Job parameters
│           ├── {scriptname}.pid    # Process info (PID, type) while running
│           ├── {scriptname}.done   # Marker file when job succeeds
│           ├── {scriptname}.failed # Marker file when job fails (with reason)
│           ├── {scriptname}.out    # Standard output
│           ├── {scriptname}.err    # Standard error
│           ├── locks.json          # Dynamic dependency locks (tokens)
│           └── .experimaestro/
│               ├── status.json     # Job state, timestamps, progress
│               ├── {scriptname}.lock  # Job lock file
│               ├── task-outputs.jsonl # Dynamic task output events
│               └── events/         # Permanent event storage (after archival)
│                   └── event-{count}.jsonl
├── .events/                        # Temporary event files (watched by scheduler)
│   ├── experiments/
│   │   └── {experiment-id}/
│   │       ├── current             # Symlink to current run directory
│   │       └── events-{count}.jsonl
│   └── jobs/
│       └── {task-type-id}/
│           └── event-{job-id}-{count}.jsonl
├── partials/                       # Shared partial directories
│   └── {task-type-id}/
│       └── {partial-name}/
│           └── {partial-hash}/
├── config/                         # Configuration cache
└── .experimaestro/
    └── experiments/
        ├── events-{count}@{experiment-id}.jsonl  # Event log (active experiments)
        └── {experiment-id}                       # Symlink to current run
```

## Run ID Format

Each experiment run is identified by a **run ID** based on the timestamp when
the experiment started:

- Format: `YYYYMMDD_HHMMSS` (e.g., `20250108_143022`)
- If multiple runs start within the same second, a suffix is added:
  `20250108_143022.1`, `20250108_143022.2`, etc.

This format ensures:
- Runs are naturally sorted chronologically
- Each run has a unique identifier
- Run IDs are human-readable

## Experiment Lock

The lock file at `experiments/{experiment-id}/lock` prevents multiple instances
of the same experiment from running simultaneously. When you start an experiment:

1. Experimaestro acquires an exclusive lock on this file
2. If another process holds the lock, a warning is displayed with the hostname
   of the holder (if available)
3. The process waits until the lock is released

This ensures data integrity and prevents race conditions.

## Environment Information

The `environment.json` file captures the complete runtime environment when the
experiment starts:

```
{
  "python_version": "3.10.12",
  "packages": {
    "experimaestro": "2.0.0",
    "torch": "2.1.0",
    ...
  },
  "editable_packages": {
    "my-project": {
      "version": "0.1.0",
      "path": "/home/user/my-project",
      "git": {
        "branch": "main",
        "commit": "abc123...",
        "dirty": false
      }
    }
  },
  "projects": [...],
  "run": {
    "hostname": "compute-node-01",
    "started_at": "2025-01-08T14:30:22",
    "ended_at": "2025-01-08T15:45:10",
    "status": "completed"
  }
}
```

This information is essential for **reproducibility** - you can recreate the
exact environment that was used for any experiment run.

## History Cleanup

Experimaestro automatically manages experiment history to prevent disk space
accumulation. The cleanup behavior is controlled by settings:

```yaml
# In ~/.config/experimaestro/settings.yaml

# Global defaults
history:
  max_done: 5      # Keep last 5 successful runs per experiment
  max_failed: 1    # Keep last 1 failed run per experiment

workspaces:
  - id: my-workspace
    path: ~/experiments
    # Override for this workspace
    history:
      max_done: 10
      max_failed: 2
```

**Cleanup rules:**
- When an experiment **succeeds**, all previous failed runs are removed
- Only the most recent `max_done` successful runs are kept
- Only the most recent `max_failed` failed runs are kept
- Runs with unknown status are never automatically deleted

## v1 Experiment Layout

Experimaestro v2 can read experiments created with v1 (the `xp/` directory layout).
The state provider automatically detects and handles both layouts:

**v1 layout** (legacy):
```
WORKSPACE_DIR/
└── xp/
    └── {experiment-id}/
        ├── jobs/           # Symlinks to job directories (current run)
        └── jobs.bak/       # Symlinks to job directories (previous run)
```

**v2 layout** (current):
```
WORKSPACE_DIR/
└── experiments/
    └── {experiment-id}/
        └── {run-id}/
            ├── status.json
            └── jobs.jsonl
```

### Migration (Optional)

If you want to migrate v1 experiments to v2 layout:

```bash
# Preview what will be migrated
experimaestro migrate v1-to-v2 /path/to/workspace --dry-run

# Perform the migration
experimaestro migrate v1-to-v2 /path/to/workspace

# Keep remaining files (renamed to xp_MIGRATED_TO_V2)
experimaestro migrate v1-to-v2 /path/to/workspace --keep-old
```

The migration:
- Moves each experiment from `xp/{exp-id}/` to `experiments/{exp-id}/{run-id}/`
- Generates a run ID based on the directory's modification time
- Removes the empty `xp/` directory
- Creates a broken symlink `xp -> /experimaestro_v2_migrated_workspace_do_not_use_v1`

:::{note}
Migration is optional. The TUI, web UI, and CLI commands work with both layouts.
However, new experiments always use the v2 layout.
:::

## Job Execution

When a job is started by the scheduler, several files are created and used
to coordinate execution and track state. The `{scriptname}` is derived from
the task identifier (last component after the last `.`, e.g., `MyTask` from
`my.module.MyTask`).

### Locking

The lock file at `jobs/{task-id}/{job-hash}/.experimaestro/{scriptname}.lock`
ensures exclusive access to a job. Both the scheduler and the job process use
this lock at different phases:

1. **Scheduler lock phase**: The scheduler acquires the lock before setting up
   the job directory, writing `status.json`, launching the process, and writing
   the PID file. The lock is released after the process is launched.
2. **Process lock phase**: The job process acquires the same lock when it starts
   executing the task. It holds the lock until the task completes and the
   terminal marker (`.done`/`.failed`) is written.

There is a brief gap between these two phases where the lock is not held but
the job is still active.

### PID File

The file `{scriptname}.pid` is written by the scheduler (inside `aio_run()`)
while it still holds the job lock. It contains a JSON object describing the
process:

```json
{"type": "local", "pid": 12345}
```

The `type` field identifies the process handler (e.g., `local`, `ssh`, `slurm`).
Process liveness is checked using the launcher-independent `Process` abstraction
(`Process.fromDefinition()`), not directly via `psutil`, so that it works
across different launchers.

### Terminal Markers

When a job finishes, the job process writes one of:
- `{scriptname}.done` — job succeeded
- `{scriptname}.failed` — job failed, contains a JSON object with failure details
  (e.g., `{"reason": "FAILED"}`)

The job process also writes a final `status.json` with updated timestamps.

If the job is killed externally (e.g., SLURM `scancel`, OOM killer), these
markers are **not** written. In that case, cleanup is handled by the scheduler
(if still running) or by a later experimaestro process.

### Event Files

While a job is running, state change events are written to temporary event
files in `.events/jobs/{task-id}/event-{job-id}-{count}.jsonl`. The scheduler
watches this directory to track job progress in real time.

When a job completes, these temporary event files are archived to the permanent
location at `jobs/{task-id}/{job-hash}/.experimaestro/events/` and then deleted
from `.events/`.

:::{important}
The cleanup process that consolidates orphaned event files checks that a job is
not active before deleting its event files. A job is considered active if:
- Its lock is held, OR
- Its PID file references a running process, OR
- No terminal marker (`.done`/`.failed`) exists
:::

## State Tracking

Experimaestro uses a filesystem-based state tracking system instead of a database.
This approach is more robust on network filesystems (NFS) and easier to inspect.

### Status File (`status.json`)

Each experiment run has a `status.json` file containing the experiment metadata and service state:

```json
{
  "version": 1,
  "experiment_id": "my-experiment",
  "run_id": "20250108_143022",
  "events_count": 42,
  "hostname": "compute-node-01",
  "started_at": "2025-01-08T14:30:22.123456",
  "ended_at": "2025-01-08T15:45:10.654321",
  "status": "completed",
  "finished_jobs": 10,
  "failed_jobs": 1,
  "services": {
    "<service_id>": {
      "service_id": "...",
      "description": "...",
      "class": "mypackage.services.MyService",
      "state_dict": {}
    }
  }
}
```

:::{note}
Job details are stored separately in `jobs.jsonl` rather than in `status.json`.
This reduces memory usage and allows for efficient streaming of job information.
:::

### Jobs File (`jobs.jsonl`)

Lightweight job information is stored in a separate JSONL file (one JSON object per line):

```json
{"job_id": "abc123", "task_id": "my.task.Train", "tags": {"experiment": "v1"}, "timestamp": 1736343025.0}
{"job_id": "def456", "task_id": "my.task.Evaluate", "tags": {}, "timestamp": 1736343030.5}
```

Each record contains:
- `job_id`: Unique job identifier
- `task_id`: Task type identifier
- `tags`: Dictionary of job tags
- `timestamp`: When the job was submitted (Unix timestamp)

### Event Log

While an experiment is running, events are streamed to a JSONL file at
`.events/experiments/events-{count}@{experiment-id}.jsonl`:

```json
{"type": "job_submitted", "job_id": "abc123", "task_id": "my.task", "timestamp": 1736343025.0}
{"type": "job_state_changed", "job_id": "abc123", "state": "running", "timestamp": 1736343030.5}
{"type": "service_added", "service_id": "tensorboard", "description": "TensorBoard", "timestamp": 1736343035.0}
```

When the experiment completes, events are consolidated into `status.json` and
the event log is cleaned up.

### Current Run Symlink

A symlink at `.events/experiments/{experiment-id}/current` points to the current
(or most recent) run directory. This allows quick access to the active run
without scanning all run directories

## Related Commands

```bash
# List experiments in a workspace
experimaestro experiments --workdir /path/to/workspace list

# Monitor experiments (TUI)
experimaestro experiments --workdir /path/to/workspace monitor --console

# Monitor experiments (Web UI)
experimaestro experiments --workdir /path/to/workspace monitor --port 12345

# Check for orphan jobs
experimaestro orphans /path/to/workspace
```
