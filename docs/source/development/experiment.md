# Experiment and Scheduler Architecture

This document describes the internal architecture of the experimaestro experiment
and scheduler systems, covering job lifecycle, state management, and completion handling.

See also the [Internal API Reference](api.md) for detailed class documentation.

## Overview

The experiment system consists of two main components:

1. **{py:class}`~experimaestro.scheduler.experiment.experiment`** (`experiment.py`): Context manager for experiment setup, tracking, and cleanup
2. **{py:class}`~experimaestro.scheduler.base.Scheduler`** (`base.py`): Singleton managing job lifecycle and execution across all experiments

```
+-------------------------------------------------------------------+
|                           Scheduler                               |
|           (singleton, asyncio event loop in background)           |
|                                                                   |
|   +--------------+   +--------------+   +--------------+          |
|   | Experiment 1 |   | Experiment 2 |   | Experiment 3 |          |
|   |   (context)  |   |   (context)  |   |   (context)  |          |
|   |              |   |              |   |              |          |
|   | +----+----+  |   |    +----+    |   | +----+----+  |          |
|   | |Job1|Job2|  |   |    |Job1|    |   | |Job1|Job2|  |          |
|   | +----+----+  |   |    +----+    |   | +----+----+  |          |
|   +--------------+   +--------------+   +--------------+          |
+-------------------------------------------------------------------+
```

## Experiment Lifecycle

### Initialization (`__init__`)

The {py:class}`~experimaestro.scheduler.experiment.experiment` class is a context manager that sets up the experiment environment:

```python
with experiment(workdir, "my-experiment", port=12345) as xp:
    task.submit()
```

During initialization:
- Creates {py:class}`~experimaestro.scheduler.workspace.Workspace` from settings or path
- Sets up experiment base directory (`experiments/{name}/`)
- Prepares lock path for preventing concurrent runs
- Captures project paths for git info

### Entry (`__enter__`)

When entering the context:

1. **Lock acquisition**: Locks experiment to prevent concurrent runs
2. **Run ID generation**: Creates unique `run_id` (format: `YYYYMMDD_HHMMSS`, with `.N` suffix for collisions)
3. **Directory creation**: Creates run-specific `workdir` (`experiments/{name}/{run_id}/`)
4. **Environment capture**: Records git info, Python version, hostname
5. **Dirty git check**: Optionally warns or errors on uncommitted changes (raises {py:class}`~experimaestro.DirtyGitError`)
6. **Scheduler registration**: Registers with the singleton {py:class}`~experimaestro.scheduler.base.Scheduler`
7. **Event writer initialization**: Starts {py:class}`~experimaestro.scheduler.state_status.EventWriter` for state changes
8. **TaskOutputsWorker start**: Begins processing dynamic task outputs via {py:class}`~experimaestro.scheduler.dynamic_outputs.TaskOutputsWorker`

### Exit (`__exit__`)

When exiting the context:

1. **Wait for jobs**: Unless exception or {py:class}`~experimaestro.scheduler.experiment.GracefulExperimentExit`, waits for all jobs to complete
2. **Stop services**: Stops any registered services
3. **Finalize run**: Writes final status and archives events
4. **Cleanup transient jobs**: Removes transient job directories if clean exit
5. **History cleanup**: Removes old runs based on `max_done`/`max_failed` settings
6. **Release lock**: Releases workspace lock

### File Structure

```
experiments/{name}/
  lock                          # Experiment-level lock file
  {run_id}/
    environment.json            # Environment and git info
    status.json                 # Final run status
    jobs.jsonl                  # Job list (one line per job)
    jobs/                       # Symlinks to job directories
    results/                    # User-defined results
    data/                       # Saved configurations
```

## Scheduler Architecture

The {py:class}`~experimaestro.scheduler.base.Scheduler` is a singleton that runs an asyncio event loop in a background thread.
It manages job lifecycle across all active experiments.

### Key Components

| Component | Purpose |
|-----------|---------|
| `experiments` | Dict of active experiments |
| `jobs` | Dict of all {py:class}`~experimaestro.scheduler.jobs.Job` instances (by identifier) |
| `waitingjobs` | Set of jobs awaiting execution |
| `loop` | The asyncio event loop |
| `exitCondition` | Condition for signaling completions |
| `dependencyLock` | Lock for acquiring dynamic dependencies |

### Singleton Pattern

```python
scheduler = Scheduler.instance()  # Get or create singleton
```

Uses an internal event loop thread for a shared event loop across locking, file watching, and scheduling.

## Job Lifecycle

{py:class}`~experimaestro.scheduler.jobs.Job` instances transition through these {py:class}`~experimaestro.scheduler.interfaces.JobState` values:

```
UNSCHEDULED → WAITING → READY → SCHEDULED → RUNNING → DONE/ERROR
     (0)        (1)      (2)       (3)         (4)      (5/6)
```

| State | Description |
|-------|-------------|
| `UNSCHEDULED` | Initial state, job not yet submitted |
| `WAITING` | Job submitted, waiting for dependencies |
| `READY` | Dependencies resolved, ready to execute |
| `SCHEDULED` | Scheduled for execution |
| `RUNNING` | Job process is executing |
| `DONE` | Job completed successfully |
| `ERROR` | Job failed (with optional {py:class}`~experimaestro.scheduler.interfaces.JobFailureStatus`) |

### Error States

`ERROR` states can have a {py:class}`~experimaestro.scheduler.interfaces.JobFailureStatus`:
- `DEPENDENCY`: Upstream job failed
- `FAILED`: Job execution failed
- `MEMORY`: Out of memory
- `TIMEOUT`: Walltime exceeded (allows retries for resumable tasks)

## Job Submission Pipeline

```
submit() → aio_registerJob() → aio_submit() → aio_start() → completion
    │              │                │              │
    │              │                │              └→ Execute process, wait
    │              │                └→ Create symlinks, notify, start inner
    │              └→ Deduplicate, merge state, check resubmission
    └→ Thread-safe entry point
```

### 1. `submit(job)` - Thread-safe Entry

- Calls `aio_registerJob()` to check for duplicates
- Sets up job future for async tracking
- Returns immediately with existing job if resubmitted

### 2. `aio_registerJob(job)` - Deduplication

Jobs with identical configurations get the same identifier (hash). When resubmitting:

1. Checks if job already exists in scheduler
2. If found, adds experiment to existing job's experiments list
3. Merges {py:class}`~experimaestro.scheduler.jobs.TransientMode` (more conservative wins)
4. Updates `max_retries` if new job has higher value
5. Copies watched outputs from new job to existing
6. Returns existing job (unless needs restart)

### 3. `aio_submit(job)` - Main Execution

- Adds job to `waitingjobs`
- Registers {py:class}`~experimaestro.core.objects.WatchedOutput` instances
- Creates symlink in experiment's `jobspath`
- Notifies {py:class}`~experimaestro.scheduler.base.Listener` of job submission
- Calls `aio_submit_inner()` in retry loop (for `GracefulTimeout`)
- Processes final state via `aio_final_state()`

### 4. `aio_submit_inner(job)` - Load and Check State

- Loads existing state from disk (preserves history)
- Checks if already done or exhausted retries
- Clears transient fields
- Sets state to `WAITING`
- Checks for already running process
- Calls `aio_start()` if job needs to run

### 5. `aio_start(job)` - Actual Execution

1. **Wait for static dependencies**: Sequentially waits for dependent jobs via {py:class}`~experimaestro.scheduler.jobs.JobDependency`
2. **Acquire locks**: Job lock and launcher lock
3. **Acquire dynamic dependencies**: Tokens with deadlock prevention
4. **Create job directory**: Writes metadata files
5. **Execute**: Calls `job.aio_run()` to start process
6. **Wait for completion**: Monitors process
7. **Read final state**: From `.done`/`.failed` marker files
8. **Return final state**

## State Management

All state changes go through the {py:meth}`~experimaestro.scheduler.jobs.Job.set_state` method which ensures:

1. **Automatic timestamp updates**:
   - `WAITING`: sets `submittime`, clears `starttime` and `endtime`
   - `RUNNING`: sets `starttime`
   - `DONE`/`ERROR`: sets `endtime`

2. **Experiment statistics**: Updates `unfinishedJobs` and `failedJobs` counters

3. **Listener notifications**: Notifies registered {py:class}`~experimaestro.scheduler.base.Listener` of state changes

4. **Future reset**: Resets `_final_state` future when transitioning to `WAITING`
   (enables proper handling of job resubmission)

### State Property Encapsulation

The job state is stored in `_state` (private) and accessed via the `state` property.
Direct assignment (`job.state = X`) automatically calls {py:meth}`~experimaestro.scheduler.jobs.Job.set_state` to ensure
proper handling.

## Job Locking

Each job has a lock file (`{job_path}/{scriptname}.lock`) ensuring:

- Only one scheduler instance can run a job at a time
- Status file writes are atomic and consistent

The lock is acquired before starting a job and released after the process starts.
The done handler acquires its own lock (using filelock) when writing the final
status file.

## Dependency Management

### Static Dependencies (Jobs)

- Waited sequentially before job starts via {py:class}`~experimaestro.scheduler.jobs.JobDependency`
- Don't need the dependency lock
- If dependency fails, dependent job fails with `DEPENDENCY` reason

### Dynamic Dependencies (Tokens)

Tokens are acquired with `dependencyLock` to prevent deadlocks:

**Problem**: Multiple jobs acquiring tokens could deadlock (A waits for Token1 while holding Token2, B waits for Token2 while holding Token1).

**Solution**:
- Single `dependencyLock` ensures only one task acquires dynamic deps at a time
- Retry logic: if any lock fails, release all and restart
- First dependency has 0 timeout, others have 0.1s

## Done Handler

Job completion is processed in a dedicated thread pool:

1. **Process task outputs**: Calls `job.done_handler()` to process any registered
   task output callbacks

2. **Write final status**: Acquires the job lock and writes the final status file
   with timestamps and state

3. **Cleanup**: Removes job from scheduler's waiting jobs set

4. **Resolve future**: Sets the result on `_final_state` future to unblock
   `aio_submit()` callers

### Thread Safety

The done handler runs in a separate thread pool (4 workers by default) to avoid
blocking the main event loop. Communication with the event loop uses
`call_soon_threadsafe()` and `run_coroutine_threadsafe()`.

## Transient Jobs

Jobs can be marked as transient with different {py:class}`~experimaestro.scheduler.jobs.TransientMode` values:

| Mode | Description |
|------|-------------|
| `NONE` | Normal job, always runs |
| `TRANSIENT` | Only runs if a non-transient job depends on it |
| `REMOVE` | Like `TRANSIENT`, but job directory is removed after experiment ends |

Transient jobs that aren't needed stay in `UNSCHEDULED` state and skip the done
handler processing. They can be converted to non-transient if a job depends on them.

## Resumable Tasks

Tasks inheriting from {py:class}`~experimaestro.ResumableTask` support automatic retry on timeout:

1. When a timeout occurs, `retry_count` is incremented
2. If `retry_count <= max_retries`, the job restarts automatically
3. Log files are rotated via {py:meth}`~experimaestro.scheduler.jobs.Job.rotate_logs` to preserve previous run's output
4. The job directory is preserved between retries for checkpoint files

```python
class LongTraining(ResumableTask):
    checkpoint: Meta[Path] = field(default_factory=PathGenerator("checkpoint.pth"))

    def execute(self):
        start_epoch = 0
        if self.checkpoint.exists():
            start_epoch = load_checkpoint(self.checkpoint)

        for epoch in range(start_epoch, self.epochs):
            remaining = self.remaining_time()
            if remaining is not None and remaining < 300:
                save_checkpoint(self.checkpoint, epoch)
                raise GracefulTimeout("Not enough time")

            train_one_epoch()
            save_checkpoint(self.checkpoint, epoch)
```

## Dynamic Task Outputs

For tasks that produce outputs during execution (e.g., checkpoints), see {py:class}`~experimaestro.core.objects.WatchedOutput`:

```python
class Training(ResumableTask):
    validation: Param[Validation]

    def execute(self):
        for step in range(100):
            train_step()
            if step % 10 == 0:
                self.validation.compute(step)  # Calls register_task_output

# Usage
def on_checkpoint(checkpoint):
    Evaluate.C(checkpoint=checkpoint).submit()

training = Training.C(...)
training.watch_output(validation.checkpoint, on_checkpoint)
training.submit()
```

The {py:class}`~experimaestro.scheduler.dynamic_outputs.TaskOutputsWorker` processes these in a separate thread:
- Queues callbacks for execution
- Updates `taskOutputQueueSize` via scheduler
- Waits for all outputs before experiment completes
- Events stored in `.experimaestro/task-outputs.jsonl` for replay on restart

## Event System

### Event Types

**Job Events:**
- {py:class}`~experimaestro.scheduler.state_status.JobSubmittedEvent`: Job was submitted (records tags, dependencies)
- {py:class}`~experimaestro.scheduler.state_status.JobStateChangedEvent`: Job state changed (records failure reason, times, exit code)
- {py:class}`~experimaestro.scheduler.state_status.JobProgressEvent`: Job progress updated

**Experiment Events:**
- {py:class}`~experimaestro.scheduler.state_status.RunCompletedEvent`: Experiment run finished
- `ServiceAddedEvent`: Service added to experiment
- `ServiceStateChangedEvent`: Service state changed

**Process Events:**
- `CarbonMetricsEvent`: Carbon tracking data (see {py:class}`~experimaestro.scheduler.state_provider.CarbonMetricsData`)
- `ProcessStartedEvent`: Process started
- `ProcessStateEvent`: Process state changed

### Event Storage

Events are written to JSONL files via {py:class}`~experimaestro.scheduler.state_status.EventWriter` and read via {py:class}`~experimaestro.scheduler.state_status.EventReader`:
```
workspace/.events/
  experiments/{exp_id}/
    events-0.jsonl
    events-1.jsonl
  jobs/{task_id}/
    event-{job_id}-0.jsonl
```

After run completion, events are archived to permanent storage for replay.

## Notification System

Two parallel notification systems:

1. **Legacy listeners**: {py:class}`~experimaestro.scheduler.base.Listener` callbacks (`job_submitted`, `job_state`)
2. **StateProvider listeners**: {py:class}`~experimaestro.scheduler.state_provider.StateProvider` event-based for TUI/Web UI

Notifications run in a thread pool to avoid blocking the scheduler.

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| {py:class}`~experimaestro.scheduler.base.Scheduler` | `base.py` | Main scheduler, manages event loop and job lifecycle |
| {py:class}`~experimaestro.scheduler.experiment.experiment` | `experiment.py` | Context manager for experiment setup |
| {py:class}`~experimaestro.scheduler.jobs.Job` | `jobs.py` | Job state machine with dependencies |
| {py:class}`~experimaestro.scheduler.interfaces.BaseJob` | `interfaces.py` | Base class with state property and timestamps |
| {py:class}`~experimaestro.scheduler.dynamic_outputs.TaskOutputsWorker` | `dynamic_outputs.py` | Processes dynamic task outputs |
| {py:class}`~experimaestro.scheduler.workspace.Workspace` | `workspace.py` | Working directory and run mode management |
| {py:class}`~experimaestro.scheduler.experiment.StateListener` | `experiment.py` | Writes state changes to event files |

## Workspace Structure

```
workspace/
  .events/
    experiments/{exp_id}/events-*.jsonl
    jobs/{task_id}/event-{job_id}-*.jsonl
  jobs/
    {task_id}/
      {job_hash}/
        .experimaestro/
          status.json
          task-outputs.jsonl
        params.json
        script.py
        script.out
        script.err
        script.done/.failed
        script.pid
        locks.json
  experiments/
    {exp_id}/
      lock
      {run_id}/
        environment.json
        status.json
        jobs.jsonl
        jobs/ (symlinks)
        results/
        data/
  partials/
  config/
```
