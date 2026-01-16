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

- **Experiments Table**: Lists experiments with ID, hostname, jobs count, status, and duration
- **Job Table**: Scrollable list of jobs with status indicators
- **Live Logs**: Real-time log output panel
- **Keyboard Navigation**: Navigate and interact using keyboard shortcuts
- **Status Bar**: Shows experiment status and connection info

### Keyboard Shortcuts

Press `?` in the TUI to show the help screen with all shortcuts.

**Navigation:**

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `Esc` | Go back / Close dialog |
| `r` | Refresh data |
| `?` | Show help |
| `j` | Switch to Jobs tab |
| `s` | Switch to Services tab |

**Experiments:**

| Key | Action |
|-----|--------|
| `Enter` | Select experiment |
| `d` | Show experiment runs |
| `Ctrl+d` | Delete experiment |
| `k` | Kill all running jobs |
| `S` | Sort by status |
| `D` | Sort by date |

**Jobs:**

| Key | Action |
|-----|--------|
| `l` | View job logs |
| `Ctrl+d` | Delete job |
| `k` | Kill running job |
| `/` | Open search filter |
| `c` | Clear search filter |
| `S` | Sort by status |
| `T` | Sort by task |
| `D` | Sort by date |
| `f` | Copy folder path |

**Log Viewer:**

| Key | Action |
|-----|--------|
| `f` | Toggle follow mode |
| `g` | Go to top |
| `G` | Go to bottom |
| `r` | Sync now (remote) |
| `Esc`/`q` | Close viewer |

**Services:**

| Key | Action |
|-----|--------|
| `s` | Start service |
| `x` | Stop service |
| `u` | Copy URL |

**Orphan Jobs:**

| Key | Action |
|-----|--------|
| `r` | Refresh |
| `T` | Sort by task |
| `Z` | Sort by size |
| `Ctrl+d` | Delete selected |
| `Ctrl+D` | Delete all |
| `f` | Copy folder path |

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

## Remote Monitoring via SSH

When experiments run on remote servers (e.g., HPC clusters, cloud instances),
you can monitor them from your local machine using SSH tunneling. The `ssh-monitor`
command establishes an SSH connection, starts a monitoring server on the remote
host, and communicates via JSON-RPC over the SSH channel.

### Basic Usage

```bash
experimaestro experiments ssh-monitor HOST REMOTE_WORKDIR [OPTIONS]
```

- **HOST**: SSH host in standard format (e.g., `user@server`, `server`, or SSH config alias)
- **REMOTE_WORKDIR**: Path to the workspace directory on the remote server

### Examples

```bash
# Basic SSH monitoring with web UI
experimaestro experiments ssh-monitor myserver /home/user/experiments

# With console TUI instead of web UI
experimaestro experiments ssh-monitor user@cluster.example.com /scratch/experiments --console

# Custom port for web interface
experimaestro experiments ssh-monitor myserver /workspace --port 8080

# With SSH options (e.g., custom port, identity file)
experimaestro experiments ssh-monitor myserver /workspace -o "-p 2222" -o "-i ~/.ssh/cluster_key"

# Specify path to experimaestro on remote host
experimaestro experiments ssh-monitor myserver /workspace --remote-xpm /opt/conda/bin/experimaestro
```

### Options

| Option | Description |
|--------|-------------|
| `--console` | Use terminal TUI instead of web UI |
| `--port PORT` | Port for local web server (default: 12345) |
| `--remote-xpm PATH` | Path to experimaestro executable on remote host |
| `-o, --ssh-option OPT` | Additional SSH options (can be repeated) |

### How It Works

1. **Connection**: The client establishes an SSH connection to the remote host
2. **Server Start**: On the remote host, experimaestro starts in server mode,
   reading commands from stdin and writing responses to stdout
3. **JSON-RPC Protocol**: Commands and responses are exchanged using JSON-RPC 2.0
4. **File Sync**: When needed (e.g., for TensorBoard logs), files are synchronized
   on-demand using rsync over SSH
5. **Real-time Updates**: The server sends notifications when job states change

### Remote experimaestro Installation

By default, the SSH client runs `uv tool run experimaestro==<version>` on the
remote host. This requires:

- `uv` installed on the remote host
- Network access to PyPI (or a local mirror)

Alternatively, specify a pre-installed experimaestro path:

```bash
# If experimaestro is installed in a virtualenv
experimaestro experiments ssh-monitor host /workspace --remote-xpm /path/to/venv/bin/experimaestro

# If installed system-wide
experimaestro experiments ssh-monitor host /workspace --remote-xpm experimaestro
```

### Version Compatibility

The local and remote experimaestro versions should be compatible. The protocol
version is checked on connection, and a warning is shown if versions differ
significantly. For best results, use the same version on both sides.

### SSH Output

Remote server output (logs, warnings) is displayed locally with a colored
`[SSH]` prefix to distinguish it from local output. This helps debug
connection issues or see remote server activity.

### Troubleshooting

**Connection refused or timeout:**
- Verify SSH access: `ssh user@host echo "connected"`
- Check if the remote workspace exists
- Ensure experimaestro is accessible on the remote host

**Jobs not appearing:**
- The remote database may need synchronization:
  ```bash
  ssh user@host "experimaestro experiments sync --workdir /path/to/workspace"
  ```

**Services not working (e.g., TensorBoard):**
- Services are recreated locally from saved state
- Some services may require file synchronization; this happens automatically
- Check that rsync is available on both machines

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
