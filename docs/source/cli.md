# Command Line Interface

(running-experiments)=
## Running Experiments

The general command to launch experiments is:

```bash
experimaestro run-experiment [OPTIONS] YAML_FILE [ARGS]
```

### run-experiment Options

| Option | Description |
|--------|-------------|
| `--run-mode MODE` | Set run mode: `NORMAL`, `DRY_RUN`, or `GENERATE_ONLY` |
| `--workspace ID` | Use a workspace from settings by ID |
| `--workdir PATH` | Specify working directory directly |
| `--port PORT` | Port for web monitoring interface |
| `--console` | Use terminal TUI instead of web interface |
| `--host HOST` | Server hostname (for remote job access) |
| `--pre-yaml FILE` | Load YAML file before the main one (can be repeated) |
| `--post-yaml FILE` | Load YAML file after the main one (can be repeated) |
| `-c KEY=VALUE` | Override configuration values |
| `--file PATH` | Python file containing the `run` function |
| `--module-name NAME` | Python module containing the `run` function |
| `--env KEY VALUE` | Set environment variable (can be repeated) |
| `--xpm-config-dir PATH` | Custom experimaestro config directory |
| `--show` | Print merged configuration and exit |
| `--debug` | Enable debug logging |

### Run Modes

`experimaestro run-experiment --run-mode [DRY_RUN|NORMAL|GENERATE_ONLY]`

- `DRY_RUN` - Display task hashes and dependencies without launching
- `GENERATE_ONLY` - Generate job folders without [launching](./launchers/index.md) tasks
- `NORMAL` - Generate and launch jobs (default)

## Job Control

Besides the web interface, it is possible to use the command line to check the job
status and control jobs.

### Database Synchronization

**Important:** The job commands read from the workspace database. If jobs are not visible,
you may need to synchronize the database with the filesystem:

```bash
experimaestro experiments sync --workdir /path/to/workspace
```

This is typically needed when:
- Jobs were run before the database was introduced
- Jobs were created by an external process
- The database is out of sync with the filesystem

### Available Commands

| Command | Description |
|---------|-------------|
| `experimaestro jobs list` | List all jobs in the workspace (sorted by date, most recent first) |
| `experimaestro jobs kill` | Kill running jobs |
| `experimaestro jobs clean` | Clean (delete) finished jobs |
| `experimaestro jobs log JOBID` | View job log (stderr by default) |
| `experimaestro jobs path JOBID` | Print the path to a job directory |

### Common Options

All job commands support these options:

- `--workdir PATH`: Specify the workspace directory
- `--workspace ID`: Use a workspace from settings by ID
- `--experiment ID`: Filter by experiment ID
- `--filter EXPR`: Filter using a filter expression (see below)
- `--tags`: Show job tags in output
- `--fullpath`: Show full paths instead of `task.id/hash` format

### List Command Options

The `list` command has additional options:

- `--count N` or `-c N`: Limit output to the N most recent jobs

```bash
# Show only the 10 most recent jobs
experimaestro jobs list -c 10

# Show the 5 most recent running jobs
experimaestro jobs list --filter '@state = "RUNNING"' -c 5
```

### Kill and Clean Commands

The `kill` and `clean` commands require `--perform` to actually execute:

```bash
# Dry run - shows what would be killed
experimaestro jobs kill --filter '@state="RUNNING"'

# Actually kill the jobs
experimaestro jobs kill --filter '@state="RUNNING"' --perform
```

```bash
# Dry run - shows what would be cleaned
experimaestro jobs clean --filter '@state="DONE"'

# Actually clean the jobs (deletes directories and database entries)
experimaestro jobs clean --filter '@state="DONE"' --perform
```

### Log and Path Commands

The `log` and `path` commands take a JOBID in the format `task.name/hash`:

```bash
# View stderr log (default)
experimaestro jobs log mymodule.MyTask/abc123def

# View stdout log
experimaestro jobs log --std mymodule.MyTask/abc123def

# Follow log (like tail -f)
experimaestro jobs log -f mymodule.MyTask/abc123def

# Get job directory path
experimaestro jobs path mymodule.MyTask/abc123def
```

## Filter Expressions

The job filter is a boolean expression that allows filtering jobs by state, name, or tags.

### Special Variables

- `@state`: Job state (values: `UNSCHEDULED`, `WAITING`, `READY`, `RUNNING`, `DONE`, `ERROR`)
- `@name`: Task identifier (e.g., `mymodule.MyTask`)

### Operators

- `=`: Equality comparison
- `~`: Regex matching
- `in`: Membership in a list
- `not in`: Non-membership in a list
- `and`: Logical AND
- `or`: Logical OR

### Examples

Filter by state:
```bash
# List only running jobs
experimaestro jobs list --filter '@state = "RUNNING"'

# List completed jobs
experimaestro jobs list --filter '@state = "DONE"'

# List failed jobs
experimaestro jobs list --filter '@state = "ERROR"'
```

Filter by tag:
```bash
# Filter by a single tag
experimaestro jobs list --filter 'model = "bm25"'

# Filter by multiple tags
experimaestro jobs list --filter 'model = "bm25" and dataset = "msmarco"'
```

Complex filters:
```bash
# Running or waiting jobs
experimaestro jobs list --filter '@state = "RUNNING" or @state = "WAITING"'

# Jobs with model in a list
experimaestro jobs list --filter 'model in ["bm25", "splade"]'

# Jobs NOT matching a value
experimaestro jobs list --filter 'mode not in ["debug", "test"]'

# Combine state and tag filters
experimaestro jobs list --filter '@state = "ERROR" and model = "bm25"'
```

## Tags

Using the `--tags` flag, lists all jobs and shows their tagged arguments:

```bash
experimaestro jobs list --tags
```

Output example:
```
DONE    mymodule.MyTask/abc123 [my_experiment] model=bm25 dataset=msmarco
RUNNING mymodule.MyTask/def456 [my_experiment] model=splade dataset=msmarco
```

### Cleanup Partial Directories

Partial directories are shared checkpoint locations created by `subparameters`.
When all jobs using a partial are deleted, the partial becomes orphaned:

```bash
# Dry run - shows what would be deleted
experimaestro jobs cleanup-partials

# Actually delete orphan partials
experimaestro jobs cleanup-partials --perform
```

## Experiment Management

The `experiments` command group provides tools for managing experiments:

```bash
experimaestro experiments [OPTIONS] COMMAND
```

### Available Commands

| Command | Description |
|---------|-------------|
| `list` | List experiments in the workspace |
| `monitor` | Monitor experiments with web UI or console TUI |
| `monitor-server` | Start SSH monitoring server (internal, for remote monitoring) |
| `sync` | Synchronize workspace database from disk state |

### List Command

List experiments with their status and metadata:

```bash
experimaestro experiments list --workdir /path/to/workspace
```

Output format:
```
experiment-id [hostname] (finished_jobs/total_jobs jobs)
```

- `experiment-id`: The experiment identifier
- `hostname`: The host where the experiment was launched (if recorded)
- Job counts: Number of completed jobs out of total

### Monitor Command

Launch an interactive monitoring interface for experiments:

```bash
# Web interface (default) on port 12345
experimaestro experiments monitor --workdir /path/to/workspace

# Console TUI interface
experimaestro experiments monitor --workdir /path/to/workspace --console

# Custom port for web interface
experimaestro experiments monitor --workdir /path/to/workspace --port 8080

# Force sync before starting
experimaestro experiments monitor --workdir /path/to/workspace --sync
```

#### Remote Monitoring via SSH

Monitor experiments running on a remote server through SSH:

```bash
# Monitor a remote workspace via SSH
experimaestro experiments monitor --ssh user@server --remote-workdir /path/to/workspace

# With console TUI
experimaestro experiments monitor --ssh user@server --remote-workdir /path/to/workspace --console

# With additional SSH options
experimaestro experiments monitor --ssh user@server --remote-workdir /path/to/workspace --ssh-option "-p 2222"
```

The SSH monitoring connects to the remote server, starts a monitoring server process, and communicates via JSON-RPC over SSH stdin/stdout. Files are synchronized on-demand using rsync when needed (e.g., for TensorBoard log directories).

Requirements:
- SSH access to the remote server
- `experimaestro` installed on the remote server (same version recommended)
- `rsync` available on both local and remote machines

See [Monitoring Interfaces](interfaces.md) for more details on the web and TUI interfaces.

## Cleaning Up Orphans

Check for tasks that are not part of any experimental plan in the given
experimental folder:

```
Usage: experimaestro orphans [OPTIONS] PATH

Options:
  --size      Show size of each folder
  --clean     Prune the orphan folders
  --show-all  Show even not orphans
```

## Difference between two configurations

It is possible to look at the differences (that explain that two tasks have a different identifier) by using the `parameters-difference` command

## Refactoring Code

### Fix Bare Default Values

The `refactor default-values` command helps migrate from deprecated bare default values
to the explicit `field(ignore_default=...)` syntax:

```bash
# Dry run - shows what would be changed (default)
experimaestro refactor default-values /path/to/project

# Actually apply the changes
experimaestro refactor default-values /path/to/project --perform
```

This converts patterns like:
```python
x: Param[int] = 23
```

To the explicit form:
```python
x: Param[int] = field(ignore_default=23)
```

The tool:

- Scans Python files for `Param`, `Meta`, or `Option` annotations with bare default values
- Suggests converting to `field(ignore_default=...)` to maintain backwards compatibility
- Runs in dry-run mode by default (use `--perform` to apply changes)
- Handles simple single-line defaults; multi-line defaults require manual intervention

## (Sphinx) Checking undocumented configurations

Use `experimaestro check-documentation objects.inv my_package` to check for undocumented members. The `objects.inv` file should be generated by sphinx
