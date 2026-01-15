# Scheduler Architecture

This document describes the internal architecture of the experimaestro scheduler,
which manages job lifecycle, state transitions, and completion handling.

## Job Lifecycle

Jobs transition through these states:

```
UNSCHEDULED → WAITING → RUNNING → DONE/ERROR
```

- **UNSCHEDULED**: Initial state, job not yet submitted to scheduler
- **WAITING**: Job submitted, waiting for dependencies or to start
- **RUNNING**: Job process is executing
- **DONE**: Job completed successfully
- **ERROR**: Job failed (various failure reasons possible)

## State Management

All state changes go through the `set_state()` method which ensures:

1. **Automatic timestamp updates**:
   - WAITING: sets `submittime`, clears `starttime` and `endtime`
   - RUNNING: sets `starttime`
   - DONE/ERROR: sets `endtime`

2. **Experiment statistics**: Updates `unfinishedJobs` and `failedJobs` counters

3. **Listener notifications**: Notifies registered listeners of state changes

4. **Future reset**: Resets `_final_state` future when transitioning to WAITING
   (enables proper handling of job resubmission)

### State Property Encapsulation

The job state is stored in `_state` (private) and accessed via the `state` property.
Direct assignment (`job.state = X`) automatically calls `set_state(X)` to ensure
proper handling.

## Job Locking

Each job has a lock file (`{job_path}/{scriptname}.lock`) ensuring:

- Only one scheduler instance can run a job at a time
- Status file writes are atomic and consistent

The lock is acquired before starting a job and released after the process starts.
The done handler acquires its own lock (using filelock) when writing the final
status file.

## Done Handler

Job completion is processed in a dedicated thread pool (`DoneHandlerWorker`):

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

Jobs can be marked as transient with different modes:

- **NONE**: Normal job, always runs
- **TRANSIENT**: Only runs if a non-transient job depends on it
- **REMOVE**: Like TRANSIENT, but job directory is removed after experiment ends

Transient jobs that aren't needed stay in UNSCHEDULED state and skip the done
handler processing.

## Resumable Tasks

Tasks inheriting from `ResumableTask` support automatic retry on timeout:

1. When a timeout occurs, `retry_count` is incremented
2. If `retry_count <= max_retries`, the job restarts automatically
3. Log files are rotated to preserve previous run's output
4. The job directory is preserved between retries for checkpoint files

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `Scheduler` | `base.py` | Main scheduler, manages event loop and job lifecycle |
| `Job` | `jobs.py` | Job state machine with dependencies |
| `BaseJob` | `interfaces.py` | Base class with state property and timestamps |
| `DoneHandlerWorker` | `done_handler.py` | Background job completion processing |
| `Experiment` | `experiment.py` | Context manager for experiment setup |

## Event Flow

```
submit() → aio_registerJob() → aio_submit() → aio_start() → DoneHandlerWorker
    │                              │               │              │
    │                              │               │              └→ Write status
    │                              │               └→ Start process, wait for completion
    │                              └→ Load from disk, check state
    └→ Register job, create future
```
