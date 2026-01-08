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
│           ├── status.json         # Complete state snapshot (jobs, services, etc.)
│           ├── jobs/               # Symlinks to job directories
│           ├── results/            # Saved experiment results
│           └── data/               # Serialized configurations
├── jobs/                           # Actual job output directories
│   └── {task-type-id}/
│       └── {job-hash}/
│           ├── params.json         # Job parameters
│           ├── __xpm__/            # Job metadata
│           ├── stdout              # Standard output
│           └── stderr              # Standard error
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
            └── status.json
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

## State Tracking

Experimaestro uses a filesystem-based state tracking system instead of a database.
This approach is more robust on network filesystems (NFS) and easier to inspect.

### Status File (`status.json`)

Each experiment run has a `status.json` file containing the complete state:

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
  "jobs": {
    "<job_id>": {
      "job_id": "...",
      "task_id": "...",
      "state": "done",
      "submittime": "2025-01-08T14:30:25.000000",
      "starttime": "2025-01-08T14:31:00.000000",
      "endtime": "2025-01-08T14:35:00.000000",
      "progress": 1.0
    }
  },
  "tags": {"<job_id>": {"key": "value"}},
  "dependencies": {"<job_id>": ["<depends_on_job_id>"]},
  "services": {
    "<service_id>": {
      "service_id": "...",
      "description": "...",
      "state": "running",
      "state_dict": {}
    }
  }
}
```

### Event Log

While an experiment is running, events are streamed to a JSONL file at
`.experimaestro/experiments/events-{count}@{experiment-id}.jsonl`:

```jsonl
{"type": "job_submitted", "job_id": "...", "task_id": "...", "timestamp": ...}
{"type": "job_state_changed", "job_id": "...", "state": "running", "timestamp": ...}
{"type": "service_added", "service_id": "...", "description": "...", "timestamp": ...}
```

When the experiment completes, events are consolidated into `status.json` and
the event log is cleaned up.

### Current Run Symlink

A symlink at `.experimaestro/experiments/{experiment-id}` points to the current
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
