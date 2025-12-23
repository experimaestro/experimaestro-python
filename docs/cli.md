# Command Line Interface

## Running Experiments

here is the general command used to launch experiments:

`experimaestro run-experiment --run-mode NORMAL --workspace workspace-id some_xp_params.yaml`

Use `experimaestro jobs --help` for more details.

Let's detail some important parts:

### Run modes

`experimaestro run-experiment --run-mode [DRY_RUN|NORMAL|GENERATE_ONLY] `

Several run modes are possible:
- `DRY_RUN` will not launch any Task, but display task hashes and dependencies
- `GENERATE_ONLY` will generate jobs folder without [launching](./launchers/index.md) tasks.
- `NORMAL` will both generate and launch the jobs, effectively running the experiment.

Use `experimaestro run-experiment --help` for more details.

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
| `experimaestro jobs list` | List all jobs in the workspace |
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

## Cleaning up

Check for tasks that are not part of any experimental plan in the given
experimental folder.

```
Usage: experimaestro orphans [OPTIONS] PATH

Options:
  --size      Show size of each folder
  --clean     Prune the orphan folders
  --show-all  Show even not orphans
```

## Difference between two configurations

It is possible to look at the differences (that explain that two tasks have a different identifier) by using the `parameters-difference` command

## (Sphinx) Checking undocumented configurations

Use `experimaestro check-documentation objects.inv my_package` to check for undocumented members. The `objects.inv` file should be generated by sphinx
