# Monitoring Interfaces

Experimaestro provides two interfaces for monitoring experiments and jobs:
a **Web UI** and a **Terminal UI (TUI)**. Both can be used during experiment
execution or to monitor a workspace after the fact.

## Web Interface

The web interface provides a browser-based dashboard for monitoring experiments.

```{figure} img/webui-tasks.png
:alt: Web UI tasks screen

Tasks view: filter, sort and monitor jobs by status, tags, duration and CO₂,
with live progress and per-job logs/actions.
```

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
experimaestro --workdir /path/to/workspace monitor --web --port 12345
```

Then open `http://localhost:12345` in your browser.

### Features

- **Experiments List**: Status (failed, running, done, *stopped* — no longer
  running but with unfinished jobs — or empty, as in the TUI), job counts,
  host, duration and CO₂ per experiment, with buttons to **kill** all the running jobs of an experiment
  or **delete** it (both ask for confirmation; deleting keeps the job
  directories, and is refused while the experiment is running); searchable
  by ID or [tag](#experiment-tags) and sortable by status, name or start date
  (newest first by default), by clicking the column headers; *Group by* prefix
  or tag, with collapsible groups showing their number of experiments per
  status
- **Job List**: View all jobs with their status (pending, running, done, error),
  sortable by status, task, submission date or job ID
- **Keyboard Shortcuts**: The TUI keys, outside text fields: `/` to search the
  experiments or filter the jobs (`Esc` clears it), and `S` (status), `D`
  (date), `O` (name) and, for jobs, `T` (task) to sort; the key of the current
  sort reverses it; `g` cycles the grouping of the experiments
- **Job Filter**: The same [filter expressions](#job-filter-expressions) as the
  TUI search bar, evaluated by the server so that both interfaces select the
  same jobs; invalid expressions are reported below the input. Suggestions
  appear while typing (or with `Ctrl+Space` / `↓`): the fields (`@state`,
  `@name` and the tag keys), the operators, then the values of the field —
  job states, task identifiers or tag values of the displayed jobs — and
  `and`/`or`; `↑`/`↓` select one, `Tab` or `Enter` accepts it, `Esc` closes
  the list (a second `Esc` clears the filter)
- **Grouped View**: *Group by* status, task or tag value, with collapsible
  groups showing job counts (and a status breakdown); the flat list is the
  default
- **Tags shown**: the *Tags* menu hides tag keys (e.g. a tag common to all the
  jobs) in the job list of every experiment; hidden tags can still be used to
  filter and group
- **Real-time Updates**: WebSocket-based updates as job states change
- **Job Details**: Click on a job to view parameters, logs, output paths and its
  dependencies with their status
- **Experiment Filtering**: Filter jobs by experiment ID and run
- **Log Viewing**: View stdout/stderr logs directly in the browser
- **Orphan & Stray Jobs**: Jobs no experiment references, with their disk
  usage; kill or delete them one by one, or all at once (with confirmation)
- **Services**: The services of the selected experiment (of all experiments by
  default), also when the experiment is not running; start, stop and open
  them (through `/services/<experiment>/<service>/`), or view their logs
- **Active Services**: The services starting or running, of all the
  experiments, are listed under the navigation bar on every page (with their
  experiment), with links to open them, see their logs or stop them
- **Links**: Each page has its own URL, which can be bookmarked or shared, and
  the browser's back and forward buttons move between the pages (see below)

#### Page URLs

| URL | Page |
|-----|------|
| `/` | Experiments |
| `/xps/<experiment>/jobs` | Jobs of an experiment (`?run=<run>` for a past run) |
| `/xps/<experiment>/graph` | Dependency graph (`?run=<run>` for a past run) |
| `/xps/<experiment>/services` | Services of an experiment |
| `/xps/<experiment>/actions` | Actions of an experiment |
| `/xps/<experiment>/jobs/<task>/<job>` | Details of a job (`.../logs` for its logs) |
| `/all/jobs`, `/all/services`, ... | The same pages, for all the experiments |
| `/orphans` | Orphan and stray jobs |

When the server asks for the token, the login page goes back to the page that
was opened.

Killing and deleting jobs or experiments, and starting and stopping services,
behave the same in the web UI, the TUI and (for jobs) the `experimaestro jobs`
commands: the same operations are refused
(e.g. deleting a running job) and reported with the same messages. Deleting a
job also removes the [partial directories](cli.md#cleanup-partial-directories)
it leaves orphaned.

Both interfaces also show the same job status — the execution state resolved
against the state the experiment tracks — and sort it in the same order:
running, scheduled, ready, waiting, error (timeouts and memory errors first),
done, unscheduled and transient.

### Remote Access

If your jobs run on remote machines (e.g., SLURM cluster), use the `--host` option
to make the server accessible:

```bash
experimaestro run --host 0.0.0.0 --port 12345 experiment.yaml
```

## Terminal UI (TUI)

The TUI provides an interactive terminal-based interface using [Textual](https://textual.textualize.io/).
It's useful when working over SSH or when you prefer staying in the terminal.

### Starting the TUI

**During experiment execution:**

```bash
# Use --console to launch the terminal UI
experimaestro run --console experiment.yaml
```

**Standalone monitoring:**

```bash
experimaestro --workdir /path/to/workspace monitor --console
```

### Features

- **Experiments Table**: Lists experiments with ID (and [tags](#experiment-tags)),
  hostname, jobs count, status, and duration; grouped by prefix or tag with `g`
  (`Enter` on a group collapses or expands it)
- **Job Table**: Scrollable list of jobs with status indicators, filtered with
  [filter expressions](#job-filter-expressions) (`/`)
- **Tree View**: Jobs grouped by status, task and/or tag (`t`)
- **Job Details**: Process, carbon, tags, progress and the job dependencies with
  their status (click a dependency to open its details)
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
| `Ctrl+k` | Kill all running jobs |
| `/` | Search (in the experiment IDs and tags; `Esc` clears it) |
| `S` | Sort by status |
| `D` | Sort by start date (default, newest first) |
| `O` | Sort alphabetically |
| `g` | Group by nothing, prefix or tag (in turn) |

Pressing the key of the current sort again reverses the order.

**Jobs:**

| Key | Action |
|-----|--------|
| `l` | View job logs |
| `Ctrl+d` | Delete job |
| `Ctrl+k` | Kill running job |
| `/` | Open search filter |
| `c` | Clear search filter |
| `S` | Sort by status |
| `T` | Sort by task |
| `D` | Sort by date |
| `O` | Sort alphabetically |
| `#` | Choose the tags shown (hidden tag keys, for all the experiments) |
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
| `Ctrl+S` | Start service |
| `Ctrl+K` | Stop service |
| `l` | View logs |
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

### Clipboard Support

The TUI supports clipboard operations (like `f` to copy path) that work both
locally and in remote/tmux environments.

**How it works:**

1. First tries `pyperclip` for local clipboard access
2. Falls back to [OSC 52](https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h3-Operating-System-Commands)
   escape sequences when pyperclip fails (e.g., over SSH)

**OSC 52 supported terminals:**

- iTerm2 (macOS)
- kitty
- alacritty
- Windows Terminal
- Most modern terminal emulators

**tmux configuration:**

For clipboard to work inside tmux over SSH, add these lines to `~/.tmux.conf`:

```bash
set -g set-clipboard on
set -g allow-passthrough on   # Required for tmux 3.2+
```

Then reload tmux configuration: `tmux source-file ~/.tmux.conf`

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

## Experiment Tags

Experiments can be tagged, with `experiment(..., tags=["eval", "mice"])` or the
`tags` field of the [run YAML file](experiments/cli-runner.md). The tags are
stored with each run and shown in the experiment list of both interfaces, where
the search also matches them. The experiments can be grouped:

- by **prefix**: what comes before the first `-`, `_` or `.` of the experiment
  ID (`msmarco-splade` and `msmarco_bm25` are both in `msmarco`);
- by **tag**: an experiment is in the group of each of its tags, and those
  without tags are in the last group, *(untagged)*.

## Job Filter Expressions

The TUI search bar (`/`) and the web *Filter* input accept the same expressions
(also used by `experimaestro jobs --filter`, see [CLI](cli.md)):

| Expression | Meaning |
|------------|---------|
| `@state = 'done'` | Job state (`unscheduled`, `waiting`, `ready`, `scheduled`, `running`, `done`, `error`) |
| `@name ~ 'pkg\.Learn'` | Task identifier matches a regular expression (anchored at the start) |
| `model = 'bm25'` | Tag value of the job in the displayed experiment run |
| `@state in ['running', 'waiting']` | Value in a list (`not in` for the converse) |
| `... and ...`, `... or ...` | Combine conditions |

For example, `@state = 'error' and model = 'bm25'` selects the failed jobs whose
`model` tag is `bm25`.

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
you can monitor them from your local machine using SSH tunneling. Monitoring a
[remote workspace](cli.md#remote-workspaces) establishes an SSH connection, starts a monitoring server on the remote
host, and communicates via JSON-RPC over the SSH channel.

### Basic Usage

```bash
experimaestro [SSH OPTIONS] --ssh HOST:REMOTE_WORKDIR monitor (--console | --web)
```

- **HOST**: SSH host in standard format (e.g., `user@server`, `server`, or SSH config alias)
- **REMOTE_WORKDIR**: Path to the workspace directory on the remote server

### Examples

```bash
# Basic SSH monitoring (web UI)
experimaestro --ssh myserver:/home/user/experiments monitor --web

# With the console TUI instead of the web UI
experimaestro --ssh user@cluster.example.com:/scratch/experiments monitor --console

# Custom port for web interface
experimaestro --ssh myserver:/workspace monitor --web --port 8080

# With SSH options (e.g., custom port, identity file)
experimaestro --ssh myserver:/workspace -o "-p 2222" -o "-i ~/.ssh/cluster_key" monitor --web

# Specify path to experimaestro on remote host
experimaestro --ssh myserver:/workspace --remote-xpm /opt/conda/bin/experimaestro monitor --web
```

### Options

| Option | Description |
|--------|-------------|
| `--console` / `--web` | Terminal (Textual) UI or web UI (one is required) |
| `--port PORT` | Port for local web server (default: 12345) |

The SSH options (`--remote-xpm`, `-o`, …) are given before the command, see
[Remote Workspaces](cli.md#remote-workspaces).

### How It Works

1. **Connection**: The client establishes an SSH connection to the remote host
2. **Server Start**: On the remote host, experimaestro starts in server mode,
   reading commands from stdin and writing responses to stdout
3. **JSON-RPC Protocol**: Commands and responses are exchanged using JSON-RPC 2.0
4. **File Sync**: When needed (e.g., for TensorBoard logs), files are synchronized
   on-demand using rsync over SSH
5. **Real-time Updates**: The server sends notifications when job states change
   — only for what the UI shows: the experiment list is always kept up to date,
   but job, action and service events are only sent for the experiment shown
   (and for the started services, shown on every page). The same applies to
   the web UI, for each browser page.

### Shared Workspaces

Like local monitoring, remote monitoring runs the workspace cleanup, so that
crashed jobs are recovered without waiting for the next experiment. This is safe
in a workspace shared with other users: a job whose process cannot be checked
from the monitoring machine (its PID belongs to another host, or its launcher is
not reachable) is considered active, and job files owned by another user are
never modified.

See [Sharing a workspace](experiments.md#sharing-a-workspace) for the
filesystem permissions a shared workspace needs, and
[Lock file permissions](settings.md#lock-file-permissions) for the `lock_mode`
setting.

### Configuring SSH Settings in `settings.yaml`

Instead of passing all options on the command line, you can configure SSH settings
per-workspace in your `~/.config/experimaestro/settings.yaml`:

```yaml
workspaces:
  - id: my-cluster
    path: /scratch/experiments
    ssh:
      host: user@cluster
      shell_init: "source /etc/profile; module load python/3.10"
      uv_offline: true
      python: python3
```

Then use `--workspace` to load settings automatically:

```bash
experimaestro -w my-cluster monitor --web
```

See [Remote Workspaces (SSH)](settings.md#remote-workspaces-ssh) for all available settings.

### Remote experimaestro Installation

By default, the SSH client runs `uv tool run experimaestro==<version>` on the
remote host. This requires:

- `uv` installed on the remote host
- Network access to PyPI (or a local mirror)

For environments without direct PyPI access (common on HPC clusters), use
`--uv-offline` or configure `uv_offline: true` in workspace SSH settings.
If the remote host requires environment setup before running commands
(e.g., `module load`), use `--remote-shell-init` or configure `shell_init`
in workspace SSH settings.

Alternatively, specify a pre-installed experimaestro path:

```bash
# If experimaestro is installed in a virtualenv
experimaestro --ssh host:/workspace --remote-xpm /path/to/venv/bin/experimaestro monitor --web

# If installed system-wide
experimaestro --ssh host:/workspace --remote-xpm experimaestro monitor --web
```

(version-compatibility)=
### Version Compatibility

The local and remote experimaestro versions should be compatible. The protocol
version is checked on connection, and the connection is refused if it differs.
For best results, use the same version on both sides.

A remote experimaestro older than the local one may lack some features (e.g.
the dependency graph): the remote server lists what it handles when the
connection starts, and the interfaces then report the missing features (*the
remote experimaestro does not support …*) instead of failing. Update
experimaestro on the remote host, or point `--remote-xpm` to a more recent one.

### SSH Output

Remote server output (logs, warnings) is displayed locally with a colored
`[SSH]` prefix to distinguish it from local output. This helps debug
connection issues or see remote server activity.

### Troubleshooting

**Slow or stuck interface:** trace the requests on both sides. Each one is
logged with its time and answer size: on the remote side, the time spent
answering it; on the local side, the round trip (which includes the wait in
the queue — requests are answered one at a time).

```bash
experimaestro --logging xpm.remote.trace=DEBUG -w my-cluster monitor \
    --remote-logging xpm.remote.trace=DEBUG
```

The `xpm.remote.payload` logger shows the messages themselves (very verbose).

**`uv tool run` hangs (no output):**
- The remote host likely has no outbound HTTPS access to PyPI
- Use `uv_offline: true` in workspace SSH settings or `--uv-offline` CLI flag
- Ensure the package is already cached: `ssh host "uv tool install experimaestro==X.Y.Z"` (run once with network access)

**Wrong Python version or missing modules:**
- Non-interactive SSH skips `.profile`/`.bash_profile`
- Use `shell_init` in workspace SSH settings or `--remote-shell-init` to run setup commands (e.g., `source /etc/profile; module load python/3.10`)

**Connection refused or timeout:**
- Verify SSH access: `ssh user@host echo "connected"`
- Check if the remote workspace exists
- Ensure experimaestro is accessible on the remote host

**Services not working (e.g., TensorBoard):**
- Services are recreated locally from saved state
- Some services may require file synchronization; this happens automatically
- Check that rsync is available on both machines

## Experiment DAG Visualization

The *Graph* tab of the web UI shows the dependency graph of the selected
experiment run: jobs coloured by status, the configurations linking them
(optional), stacks of similar nodes, and a grid search configuration selector.
Clicking a node highlights its lineage and lists its jobs, whose details can
be opened.

```{figure} img/DAG_example.png
:alt: Experimaestro DAG Web Viewer

Dependency graph: the selected stack of evaluations, with its lineage highlighted.
```

The same view can be exported as a standalone HTML page:

```bash
experimaestro experiments dag my_experiment --html dag.html
```

See [Experiment DAG Visualization](cli.md#experiment-dag-visualization-experimaestro-dag) in the CLI reference for details and options.
