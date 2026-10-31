# Filesystem Protocol

Experimaestro has no central server and no database: the experiment process,
the job processes, monitors (TUI, web UI, SSH server), the CLI and cleanup
routines coordinate **only through files** in the workspace. This document is
the specification of that protocol: which process writes which file, in which
order, under which lock, and what a reader may conclude from what it sees.

It has three parts:

1. **Description** of the current behaviour (layout, primitives, lifecycles,
   readers, mutators);
2. **Invariants** the protocol is meant to guarantee;
3. **Known deviations** (`D-xx`) where the code does not (yet) meet the
   invariants. Tests reproducing a deviation reference its identifier; when a
   deviation is fixed, move it to the description and remove it from the list.

Paths are relative to `src/experimaestro/`. Symbols are given instead of line
numbers, which rot.

## Actors

| Actor | Process | Code |
|---|---|---|
| **Experiment** | user script inside `with experiment(...)`; runs the scheduler loop, the event writer, the callback executor and the folders archive thread | `scheduler/experiment.py`, `scheduler/base.py`, `scheduler/hooks.py`, `scheduler/folders.py` |
| **Job** | the generated `<name>.py` script running `TaskRunner` (rank 0 = *master*, rank > 0 = *slave*) | `run.py`, `scriptbuilder.py`, `taskglobals.py` |
| **Launcher** | local `Popen`, `sbatch`, SSH; provides the `Process` handle | `connectors/`, `launchers/` |
| **Monitor** | `WorkspaceStateProvider` (TUI, web UI, CLI) and the SSH server | `scheduler/workspace_state_provider.py`, `scheduler/state_provider.py`, `scheduler/remote/server.py` |
| **Cleanup** | runs at every `WorkspaceStateProvider` start (unless `no_cleanup`) | `scheduler/cleanup.py` |
| **CLI mutators** | `jobs clean/kill`, `experiments runs prune/tag`, `workspace orphans/stray/migrate` | `cli/`, `scheduler/job_operations.py` |

Several actors of each kind may run at the same time, possibly on different
hosts sharing the workspace over NFS.

## Primitives

### Atomic writes (`utils/atomic.py`)

- `atomic_write_json(path, data, fsync=False)` / `atomic_open`: write to a
  unique `.<name>.<rand>.tmp` in the same directory, then `os.replace`. A
  reader sees either the old or the new content, never a partial one.
  `fsync=True` additionally makes the content durable before the rename
  (needed for markers that must survive a node crash).
- `atomic_create(path)`: exclusive creation through a hard link (NFS-safe
  `O_EXCL`).
- `read_json(path)`: retries on `ESTALE` and invalid JSON (NFS attribute cache).

**Append-only JSONL** files (events, `jobs.jsonl`, `objects.jsonl`,
`task-outputs.jsonl`) are not atomic: a reader may see a trailing partial
line. Readers **must** consume only lines ending in `\n` and keep their offset
before a partial line.

### Locks

All inter-process locks are `filelock.FileLock` (kernel `flock`; on NFS this
depends on the server's lock support). The kernel releases them when the
holder dies, so **a lock never needs stale detection, but its file says
nothing about liveness**. Lock files contain no PID.

| Lock file | Protects | Held by |
|---|---|---|
| `experiments/<xp>/lock` | one live run per experiment; run-id allocation; event consolidation | experiment (whole run); cleanup and `finalize_status` of monitors (timeout 0.1 s, skip if held) |
| `jobs/<task>/<hash>/<name>.lock` (**job lock**, `Job.lockpath`) | starting a job; finalising its status | scheduler during start-up and `finalize_status`; master job process during its whole run |
| `<run>/.status.json.lock` | read-modify-write of a run's `status.json` | `write_status`, `MockExperiment.from_disk` |
| `tokens/<name>.counter/ipc.lock` | token accounting | scheduler while acquiring/releasing a token |
| `<folder>/.tmp.<hash>.lock` | staging of a folder archive/recovery | folders archive thread |
| `.scheduler/<id>/.lock` | the scheduler log directory | its scheduler |
| `carbon/.lock` | carbon records | job (at the end of the run) |

### Process handles

A job's process is identified by `<name>.pid` (JSON `{"type", "pid", "host",
…}`, `Process.tospec()`), turned back into a handle with
`Process.fromDefinition(connector, spec)`, which dispatches on `type` through
the `experimaestro.process` entry points and returns `None` if the process does
not exist. **Never use `psutil` directly.**

## Layout

```
<workspace>/
├── .__experimaestro__               marker (touched by each NORMAL run)
├── experiments/<xp>/
│   ├── lock                         experiment lock
│   ├── current -> <run_id>          symlink to the latest run
│   └── <run_id>/                    YYYYmmdd_HHMMSS[.k]
│       ├── environment.json         atomic; start, host, end status
│       ├── status.json              atomic; run state, job_states, events_count
│       ├── .status.json.lock
│       ├── jobs.jsonl               append; one line per submitted job
│       ├── objects.jsonl            append; serialized configs (truncated at run start)
│       ├── events/event-<n>.jsonl   permanent event files (hard links while live)
│       ├── jobs/<task>/<hash> -> <workspace>/jobs/<task>/<hash>
│       ├── data/<name>/             xp.save()
│       └── results/
├── jobs/<task>/<hash>/
│   ├── <name>.py  [<name>.sh]       generated script (+ SLURM wrapper)
│   ├── params.json                  atomic+fsync; serialized task
│   ├── <name>.pid                   atomic; process spec
│   ├── <name>.lock                  job lock
│   ├── locks.json                   atomic; dynamic dependency locks (tokens)
│   ├── <name>.out / <name>.err      [rotated to <name>.<ts>.out on resumable retry]
│   ├── <name>.done                  atomic+fsync; {"task_outputs": N}
│   ├── <name>.failed                atomic+fsync; {"code", "reason", "message"?}
│   └── .experimaestro/
│       ├── status.json              atomic; job state
│       ├── task-outputs.jsonl       append; dynamic outputs
│       └── events/event-<k>.jsonl   hard links to .events/jobs/…
├── .events/
│   ├── experiments/<xp>-<n>.jsonl   live experiment events
│   └── jobs/<hash8>-<job>-<k>.jsonl live job events
├── .scheduler/<id>/                 scheduler log, service logs, .lock
├── carbon/
└── <connector localpath>/tokens/<token>.counter/
    ├── informations.json            {"total": N}
    ├── ipc.lock
    └── jobs/<task_id>@<hash>.json   one file per token holder
```

Legacy files still read (never written): `configs.json`, `xp/<xp>/` (v1
layout), nested event layouts, `finished_jobs`/`failed_jobs` counters. Nothing
writes or checks a layout version (`WORKSPACE_VERSION` and `status.json`
`version` are unused).

## Lifecycles

### Experiment run

1. Touch `.__experimaestro__`; acquire `experiments/<xp>/lock` (non-blocking
   attempt, then a hint from the latest `environment.json`, then **blocking**).
2. Under the lock: allocate `run_id`, create the run directory, write
   `environment.json`, truncate `objects.jsonl`, open the event writer
   (`.events/experiments/<xp>-0.jsonl`, hard-linked to `events/event-0.jsonl`),
   write `status.json` (`init_status`), re-point `current`.
3. While running:
   - `submit()` appends to `jobs.jsonl`, then streams the config to
     `objects.jsonl`, then writes `ExperimentJobStateEvent(submitted)`;
   - every job state change observed by the scheduler is written as an
     experiment event (deduplicated per job on state + failure reason);
   - every 1280 events the event file rotates; `status.json` is rewritten with
     `events_count` before the switch.
4. Exit (`experiment.__exit__`): wait for tracked work (`xp.wait()`), wait for
   notifications (5 s), then `_finalize_run`: `RunCompletedEvent`, close and
   fsync the event writer, close `objects.jsonl`, consolidate carbon, write the
   final `status.json`, archive the event files (unlink the temporary name if
   the permanent hard link exists, else move); update `environment.json`;
   release the experiment lock; then remove transient job directories and
   prune the run history.

After a crash (`kill -9`), the kernel releases the lock; `status.json` still
says `running`; temporary event files remain. **The experiment does not
consolidate its previous crashed run on start**: cleanup in a monitor does
(see below).

### Job start (scheduler)

`Scheduler.aio_submit_inner` then `aio_start`:

1. Recover the job directory from `folders` if needed; `load_from_disk`;
   replay job events; return if DONE.
2. If `<name>.pid` designates a live process: **re-attach** (wait for it,
   holding no lock).
3. Otherwise wait for static dependencies, enter the dynamic dependency locks
   (held until the process exits), acquire the **job lock**, acquire tokens
   (under `ipc.lock`: re-read holders, purge orphans, create
   `jobs/<task_id>@<hash>.json` atomically).
4. Under the job lock: create the directory, delete old event files, unlink
   `.done`/`.failed`, write `status.json`; rotate logs on a resumable retry;
   write `params.json`, the script(s); launch; write `<name>.pid`, then
   `status.json` (SCHEDULED); add the process to the token files; write
   `locks.json`.
5. Release the job lock and wait for the process exit code.

### Job execution (job process)

`TaskRunner.run`:

1. Install the SIGTERM/SIGINT handler and the atexit cleanup; `chdir` to the
   job directory; determine the rank.
2. Master only: **blocking** acquisition of the job lock (it is free once the
   scheduler finished step 4 above).
3. Load `locks.json`. If `.done` exists, exit (someone else did the job).
   Unlink `.failed`; `status.json` RUNNING; `start_of_job` event.
4. Enter the dynamic dependency locks (check that the token file exists) and
   run the task.
5. Termination:

| Outcome | Order of writes |
|---|---|
| success (return) | `cleanup()` [rm `.pid`, carbon, `status.json`, **release job lock**, end-of-job event] → `.done` |
| success (`sys.exit(0)` from the task) | `.done` → atexit `cleanup()` |
| exception | `.failed{reason: failed}` → `cleanup()` → exit 1 |
| `GracefulTimeout` | `.failed{reason: timeout, message}` → `cleanup()` → exit 1 |
| SIGTERM / SIGINT | handler raises `TaskCancelled`; background thread writes `.failed{reason: cancelled}` → `cleanup()`; main thread joins it (30 s) → exit 1 |

On exit of the dynamic dependency locks context the job deletes its token
files.

### Job completion (scheduler)

1. `process.aio_code()`; then `JobState.from_path`: `.done` wins, else
   `.failed`, else the launcher's view (`get_job_state(code)`, e.g. SLURM
   TIMEOUT), else ERROR.
2. Release tokens and dynamic dependency locks.
3. `finalize_status` under the job lock: `load_from_disk`, apply the new
   state, write `status.json` if changed. TIMEOUT + resumable + retries left →
   restart.
4. Write the experiment event; start `aio_settle`: run the hooks (task outputs
   drain up to `.done["task_outputs"]` lines, `on_completed` for DONE, folder
   archive for DONE) then mark the job as settled.

### Dynamic outputs

The job appends to `.experimaestro/task-outputs.jsonl` (flush, no fsync). The
scheduler reads complete lines from an offset (watchdog + polling) and runs
callbacks serially on the callback executor; at the end, the drain waits up to
60 s for the count written in `.done`. On restart, all lines are replayed.

### Tokens

A token is a directory of holder files. A waiting scheduler is woken by a
watchdog on the token folder, with a polling fallback (0.1 s → 30 s backoff).
Holders are released by (a) the scheduler after the process exits, (b) the job
on context exit, (c) any scheduler that sees the holder's process dead, or
(d) orphan purge (no process info and older than 30 s).

### Folder archive (`scheduler/folders.py`)

On a single worker thread: skip if the destination exists; sweep stale
`.tmp.*`; take `.tmp.<hash>.lock` (timeout 0); copy into `.tmp.<hash>`
(skipping `*.lock`, `*.pid`); `os.replace` to the destination; rewrite
`params.json` (fsync); MOVE mode: `rmtree` the source and replace it with a
symlink. Recovery runs the same staging in reverse.

## Readers

### Job state

`MockJob.load_from_disk` (monitors):

1. `.experimaestro/status.json` (errors ignored);
2. markers via `JobState.from_path`: `.done` (content ignored) wins over
   `.failed` (JSON `failure_status` / `reason`, legacy int, unparseable →
   FAILED);
3. no marker and `<name>.pid` present → RUNNING (no liveness check);
4. events replayed from `.events/jobs/` with `k ≥ events_count`.

Precedence: job events / markers, then the experiment's `job_states`, then
the job's `status.json`. `BaseJob.load_from_disk` (scheduler side) also probes
the process and maps "finished in `status.json` but no marker" to
`ERROR(MISSING_OUTPUTS)`.

Monitors cache jobs; a cached job only changes through job events or
deletion of its event files.

### Experiment state

Current run: `current` symlink, else the newest run directory. Running vs
finished comes from `status.json` `status` plus replayed events. The
experiment lock is **not** consulted by readers.

### Cleanup (monitor start)

- **Experiment events**: under the experiment lock (skip if held), replay the
  temporary events, hard-link/copy them into `<run>/events/`, unlink the
  temporary files, rewrite `status.json`.
- **Stale pid files and job events**: take the lock
  `.experimaestro/<name>.lock` (timeout 0), probe the process; a `None` probe
  (other host, unknown launcher) counts as alive; files of other users are
  never touched. Dead process → unlink `.pid`, write `.failed` if no marker,
  rewrite `status.json`.

### Mutators

| Operation | Guard |
|---|---|
| delete job (`jobs clean`, TUI, web) | cached state finished/unscheduled |
| delete experiment | cached/`status.json` status not RUNNING |
| kill job | cached state; RPC path skips the check |
| `workspace orphans --clean` | none |
| `experiments runs prune` | `get_run_status` |
| `runs tag/untag` | none (read-modify-write of `status.json`) |
| `workspace migrate` (v1→v2) | none |

## Invariants

These are the properties the protocol must guarantee. Each one is meant to
be checked by a test (crash injection or concurrent actors).

**I1 — Single runner.** At most one process executes a given job directory
at any time. *Mechanism*: the job lock, held by the scheduler while starting
and by the master job process while running.

**I2 — Marker before release.** A job publishes its outcome (`.done` or
`.failed`) **before** it releases the job lock and removes its pid file. A
process that acquires the job lock and finds neither pid nor marker may
therefore conclude that the previous run died.

**I3 — Markers are final and exclusive.** `.done` and `.failed` are written
atomically with fsync; only the scheduler, under the job lock and before
starting a new run, may remove them. A job never ends with both.

**I4 — No completed state without a marker.** A job is shown DONE only if
`.done` exists; ERROR(MISSING_OUTPUTS) when the recorded state says finished
but no marker exists.

**I5 — Liveness from locks or processes, never from status fields.** A
reader decides that an experiment or job is *alive* from the corresponding
lock (or `Process.fromDefinition`), not from a `status` field that a crashed
process never updates. Destructive operations (delete, prune, clean, migrate)
must take the lock of what they destroy, non-blocking, and refuse if it is
held.

**I6 — Read-modify-write under one lock.** Every rewrite of a run's
`status.json` holds `<run>/.status.json.lock`.

**I7 — Crash at any point is recoverable.** For every write point of every
actor, killing the actor there leaves a workspace that (a) cleanup brings to a
consistent state without user interaction, and (b) a new experiment run
resumes correctly (jobs restarted or re-attached, no job run twice
concurrently, no completed job lost).

**I8 — Runs are isolated.** Files of one run (events, status) are never
mixed with another run's, including a crashed one.

**I9 — Readers tolerate partial appends.** JSONL readers never consume a
line that does not end in `\n`, and never lose it.

**I10 — Tokens are released exactly once per holder**, and only when the
holder (the job's master process) is gone.

**I11 — Ordered settlement.** For a job run, `on_completed` callbacks see all
dynamic outputs of the run (outputs drained first), and archiving starts only
after the run's callbacks have been dispatched.

## Known deviations

Found by a code survey (October 2026). Each entry should be confirmed by a
test before being fixed; the test is then marked
`xfail(reason="D-xx")` until the fix lands.

### Locks and liveness

**D-01 — Two job lock paths** (I1, I5). The scheduler and the job lock
`jobs/<task>/<hash>/<name>.lock` (`Job.lockpath` in `scheduler/jobs.py`),
while `BaseJob.lockpath` (`scheduler/interfaces.py`, inherited by `MockJob`)
and cleanup (`scheduler/cleanup.py`, stale pid and job events) use
`.experimaestro/<name>.lock`. Cleanup's lock therefore never excludes a running
job or a starting scheduler; only the pid/marker checks protect it.

**D-02 — Job lock released before `.done`** (I2). On a normal return,
`TaskRunner.cleanup()` removes `.pid` and releases the job lock, and `.done` is
written afterwards in the `SystemExit` handler. In between, another scheduler
can take the lock, see neither pid nor marker, and start a second run (and
unlink the `.done` about to be written). The error paths write `.failed`
before the cleanup, and `sys.exit(0)` from the task writes `.done` first, so
the order also differs between the success paths.

**D-03 — Cancelled task marked done** (I3). A task that catches
`TaskCancelled` and calls `sys.exit(0)` reaches `except SystemExit` with code 0,
which writes `.done` on top of `.failed{cancelled}`; `.done` wins. In addition,
on SIGTERM `cleanup()` (job lock release) runs in a thread while the task code
may still run, and `cleaned` is not thread-safe.

**D-04 — Slave ranks release tokens** (I10). Ranks > 0 load `locks.json` and
enter `dynamic_locks.dependency_locks()` (the comment says they get dummy
locks, but no check exists); the first rank to exit unlinks the token file
while the master still runs. Slaves do not take the job lock and may read a
`locks.json` from a previous run (never deleted).

**D-05 — Token orphan check on a path that never exists** (I10). The job URI
of a token holder is `target.basepath` = `<job dir>/<name>` (no extension),
so `async_resolve_orphaned` declares orphaned every holder without process
information older than 30 s.

**D-06 — Re-attach ignores host and PID reuse** (I1). Re-attach and
`MockJob` treat any live process with the recorded PID as the job; the
`host` field is checked only by cleanup, and no start time is recorded.

**D-07 — Liveness from status fields** (I5). Readers decide that an
experiment is running from `status.json` (`is_experiment_running`), never from
the experiment lock; a crashed run stays `running` forever. `get_run_status`
falls back to `status.json` and returns `"completed"` for any status other
than done/failed — including `running` — so history pruning (run after the
experiment lock is released) and `experiments runs prune` can delete a live
or crashed run.

**D-08 — Destructive operations without locks** (I5). Delete job, delete
experiment (incl. `delete_jobs=True` over RPC, which does not check whether
other experiments use the jobs), kill over RPC, `workspace orphans --clean`
(no state check at all), `runs prune` and `workspace migrate` rely on cached
states only. `workspace stray --perform` checks `provider.is_live`, which is
always `False` for the workspace provider.

**D-09 — Experiment lock leaked on `DirtyGitError`.** The error is raised
after the lock is taken, and `__exit__` is not called.

### Status files

**D-10 — `status.json` writers bypass `.status.json.lock`** (I6): event
rotation (`_write_events_count_to_status`), cleanup consolidation,
`_persist_job_deletion`, `runs tag/untag`, and monitor-side `finalize_status`
(which uses the experiment lock instead).

**D-11 — Monitors write into a starting run.** When `status.json` is missing,
`MockExperiment.from_disk` recovers it from the `jobs/` symlinks and writes
it; the experiment creates the run directory before `init_status`, so a
monitor can write a `status.json` into a run being created.

**D-12 — "Detached" not persisted.** `_finalize_run` maps only
completed/failed; `status.json` stays `running` while `environment.json` says
`detached`.

**D-13 — MISSING_OUTPUTS invisible to monitors** (I4). `MockJob`
overrides `load_from_disk` without the MISSING_OUTPUTS branch; the stale
`status.json` state is shown. Cleanup marks dead jobs FAILED rather than
MISSING_OUTPUTS. `mark_done()` writes an empty `.done` (no `task_outputs`).

### Events and runs

**D-14 — Event files shared across runs** (I8). The temporary experiment
event file is `<xp>-<n>.jsonl` with `n` restarting at 0 and no run id, opened
in append mode: a leftover file of a crashed, unconsolidated run is appended
to and hard-linked into the new run, and `current` is re-pointed so the old
run can no longer be consolidated.

**D-15 — Crash between final `status.json` and event archiving.** The final
`status.json` has no `events_count`; leftover temporary files then raise
`OrphanedEventsError`, which needs user confirmation (violates I7a).

**D-16 — Late service events.** If `wait_for_notifications` times out, a late
`ServiceAddedEvent` reopens the event writer after it was closed and archived,
creating a new orphan file.

**D-17 — Partial lines consumed** (I9). `_read_events_from_file`,
`read_new_events`, `_apply_pending_*` and `read_events_since_count` record
`f.tell()` after iterating, so a trailing partial line is dropped and its
bytes counted as read. `_load_from_objects_jsonl` fails on a partial line
(unlike `read_objects`).

**D-18 — Non-atomic symlink swaps.** `current` and the run's `jobs/` links
are replaced by unlink then create; a reader can see them missing. Cleanup
creates `current` before taking the experiment lock.

**D-19 — Monitor caches miss out-of-band changes.** A cached job is refreshed
only by job events; a `.failed` written by cleanup, a purge, or a `.done`
without event leave it stale.

### Hooks and folders

**D-20 — Hooks run concurrently** (I11). `aio_settle` runs the hooks with
`asyncio.gather`: `on_completed` may run before late outputs are drained, and
archiving may copy the directory while callbacks still run.

**D-21 — Folder archive races.** The staging lock file is unlinked after
release (a process that opened the old inode is not excluded); MOVE mode does
`rmtree` then `symlink` without the job lock; the destination is published
before `params.json` is rewritten; recovery is a no-op when the source
directory exists without a marker.

**D-22 — Scheduler log cleanup not exclusive.** It locks only its own
`.scheduler/<id>/.lock` and deletes other directories by mtime, which log
appends do not update.

### Other

- Script files (`<name>.py`, `.sh`) are written non-atomically (harmless while
  under the job lock, given I1).
- No reader in `scheduler/` or `cli/` uses `read_json`; NFS `ESTALE` is not
  retried.
- The SSH server's `read_file` rejects `..` and absolute paths but follows
  relative symlinks inside the workspace.
- `WorkspaceStateProvider` has dead consolidation methods
  (`_consolidate_experiment_events`, `_consolidate_job_events`,
  `_write_experiment_status`, `_archive_event_files`, `_cleanup_event_files`).
