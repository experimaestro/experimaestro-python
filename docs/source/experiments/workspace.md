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
│           ├── state.json          # Final experiment state
│           ├── jobs.jsonl          # Job submissions log
│           ├── services.json       # Active services during run
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
    └── workspace.db                # SQLite database for state tracking
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

## Migration from v1

If you have an existing workspace with the old v1 layout (`xp/` directory),
you can migrate to the new v2 layout using the CLI:

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

The broken symlink prevents experimaestro v1 from recreating the `xp/` directory
and causes it to fail clearly if accidentally used on a migrated workspace.

:::{warning}
After migrating to v2, do not use experimaestro v1 commands on the workspace.
The orphan cleanup and other commands have different expectations about the
directory structure.
:::

## Database

The SQLite database at `.experimaestro/workspace.db` stores:

- Experiment metadata and run history
- Job states and relationships
- Service registrations
- Partial path tracking for cleanup

The database is automatically synchronized from disk state when needed. You can
force a sync using:

```bash
experimaestro experiments sync --workdir /path/to/workspace
```

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
