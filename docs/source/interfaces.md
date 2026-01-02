# Monitoring Interfaces

Experimaestro provides two interfaces for monitoring experiments and jobs:
a **Web UI** and a **Terminal UI (TUI)**. Both can be used during experiment
execution or to monitor a workspace after the fact.

## Web Interface

The web interface provides a browser-based dashboard for monitoring experiments.

### Starting the Web UI

**During experiment execution:**

```python
from experimaestro import experiment

# Port enables the web interface
with experiment(workdir, "my_experiment", port=12345) as xp:
    # Tasks submitted here can be monitored at http://localhost:12345
    task = MyTask.C(...).submit()
```

**Standalone monitoring:**

```bash
# Monitor an existing workspace
experimaestro experiments monitor --workdir /path/to/workspace --port 12345
```

Then open `http://localhost:12345` in your browser.

### Features

- **Job List**: View all jobs with their status (pending, running, done, error)
- **Real-time Updates**: WebSocket-based updates as job states change
- **Job Details**: Click on a job to view parameters, logs, and output paths
- **Experiment Filtering**: Filter jobs by experiment ID
- **Log Viewing**: View stdout/stderr logs directly in the browser

### Remote Access

If your jobs run on remote machines (e.g., SLURM cluster), use the `--host` option
to make the server accessible:

```bash
experimaestro run-experiment --host 0.0.0.0 --port 12345 experiment.yaml
```

## Terminal UI (TUI)

The TUI provides an interactive terminal-based interface using [Textual](https://textual.textualize.io/).
It's useful when working over SSH or when you prefer staying in the terminal.

### Starting the TUI

**During experiment execution:**

```bash
# Use --console flag to launch TUI instead of web UI
experimaestro run-experiment --console experiment.yaml
```

**Standalone monitoring:**

```bash
experimaestro experiments monitor --workdir /path/to/workspace --console
```

### Features

- **Job Table**: Scrollable list of jobs with status indicators
- **Live Logs**: Real-time log output panel
- **Keyboard Navigation**: Navigate and interact using keyboard shortcuts
- **Status Bar**: Shows experiment status and connection info

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit the TUI |
| `↑`/`↓` | Navigate job list |
| `Enter` | View job details |
| `l` | Toggle log panel |
| `r` | Refresh job list |

### Python API

The TUI can also be launched programmatically:

```python
from experimaestro.tui import ExperimentTUI

tui = ExperimentTUI(
    workdir=Path("/path/to/workspace"),
    show_logs=True,
)
tui.run()
```

## Choosing Between Web and TUI

| Use Case | Recommended |
|----------|-------------|
| Local development | Web UI |
| Remote SSH session | TUI |
| Sharing with team | Web UI (with `--host`) |
| Minimal dependencies | TUI |
| Detailed log analysis | Web UI |
| Quick status check | TUI |

## Database Synchronization

Both interfaces read job information from the workspace database. If jobs
are not appearing, synchronize the database:

```bash
experimaestro experiments sync --workdir /path/to/workspace
```

This is needed when:
- Jobs were created before the database was introduced
- Jobs were created by external processes
- The database is out of sync with the filesystem
