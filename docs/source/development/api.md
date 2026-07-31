# Internal API Reference

This section provides API documentation for internal classes used in the scheduler
and experiment systems. These are primarily useful for developers extending or
debugging experimaestro.

## Scheduler

### Scheduler

```{eval-rst}
.. autoclass:: experimaestro.scheduler.base.Scheduler
   :members: instance, submit, register_experiment, unregister_experiment, addlistener
   :show-inheritance:
   :no-index:
```

## Jobs

### Job

```{eval-rst}
.. autoclass:: experimaestro.scheduler.jobs.Job
   :members: set_state, wait, load_from_disk, finalize_status, rotate_logs
   :show-inheritance:
   :no-index:
```

### JobDependency

```{eval-rst}
.. autoclass:: experimaestro.scheduler.jobs.JobDependency
   :members:
   :show-inheritance:
```

### TransientMode

```{eval-rst}
.. autoclass:: experimaestro.scheduler.jobs.TransientMode
   :members:
```

## Job Interfaces

### BaseJob

```{eval-rst}
.. autoclass:: experimaestro.scheduler.interfaces.BaseJob
   :members:
   :no-index:
```

### JobState

```{eval-rst}
.. autoclass:: experimaestro.scheduler.interfaces.JobState
   :members: notstarted, running, finished, is_error
```

### JobFailureStatus

```{eval-rst}
.. autoclass:: experimaestro.scheduler.interfaces.JobFailureStatus
   :members:
```

## Experiment

### experiment (BaseExperiment)

```{eval-rst}
.. autoclass:: experimaestro.scheduler.experiment.experiment
   :no-index:
   :members: submit, prepare, stop, wait, add_job, add_service, watch_output, save, load
   :show-inheritance:
```

### BaseExperiment

```{eval-rst}
.. autoclass:: experimaestro.scheduler.interfaces.BaseExperiment
   :members:
```

### StateListener

```{eval-rst}
.. autoclass:: experimaestro.scheduler.experiment.StateListener
   :members:
```

## Workspace

### Workspace

```{eval-rst}
.. autoclass:: experimaestro.scheduler.workspace.Workspace
   :members:
```

### RunMode

```{eval-rst}
.. autoclass:: experimaestro.scheduler.workspace.RunMode
   :members:
```

## Dynamic Outputs

### TaskOutputsWorker

```{eval-rst}
.. autoclass:: experimaestro.scheduler.dynamic_outputs.TaskOutputsWorker
   :members:
```

### WatchedOutput

```{eval-rst}
.. autoclass:: experimaestro.core.objects.WatchedOutput
   :members:
```

## Events

### EventBase

```{eval-rst}
.. autoclass:: experimaestro.scheduler.state_status.EventBase
   :members:
```

### JobSubmittedEvent

```{eval-rst}
.. autoclass:: experimaestro.scheduler.state_status.JobSubmittedEvent
   :members:
   :show-inheritance:
```

### JobStateChangedEvent

```{eval-rst}
.. autoclass:: experimaestro.scheduler.state_status.JobStateChangedEvent
   :members:
   :show-inheritance:
```

### JobProgressEvent

```{eval-rst}
.. autoclass:: experimaestro.scheduler.state_status.JobProgressEvent
   :members:
   :show-inheritance:
```

### RunCompletedEvent

```{eval-rst}
.. autoclass:: experimaestro.scheduler.state_status.RunCompletedEvent
   :members:
   :show-inheritance:
```

### EventWriter

```{eval-rst}
.. autoclass:: experimaestro.scheduler.state_status.EventWriter
   :members:
```

### EventReader

```{eval-rst}
.. autoclass:: experimaestro.scheduler.state_status.EventReader
   :members:
```

## State Provider

Read queries (`get_experiments`, `get_jobs`, `get_tags_map`, `get_services`, ...)
are **cached by default** by the offline providers, in a single `QueryCache`
holding the answers. Events keep those answers up to date rather than dropping
them whenever they carry enough information to do so: a job state change updates
the cached job object in place, and a job submission is appended to the cached
lists and maps it belongs to. A query is only read from storage again when an
event cannot be applied to it, when a mutation is performed through the provider,
or when the caller asks for a full re-read:

```python
jobs = provider.get_jobs(experiment_id)                 # cached
jobs = provider.get_jobs(experiment_id, refresh=True)   # re-read the state
```

`refresh=True` is meant for an explicit user action (the `r` key in the TUI, the
refresh button in the web interface); everything else — including the periodic
redraws triggered by events — should use the cached form. The live provider
(`Scheduler`) accepts the argument and ignores it: its state is always current.

Concrete offline providers implement the `_fetch_*` hooks rather than the public
`get_*` methods, which are provided (and cached) by `OfflineStateProvider`.

### StateProvider

```{eval-rst}
.. autoclass:: experimaestro.scheduler.state_provider.StateProvider
   :members:
```

### OfflineStateProvider

```{eval-rst}
.. autoclass:: experimaestro.scheduler.state_provider.OfflineStateProvider
   :members:
   :show-inheritance:
```

### QueryCache

```{eval-rst}
.. autoclass:: experimaestro.scheduler.query_cache.QueryCache
   :members:
```

### CarbonMetricsData

```{eval-rst}
.. autoclass:: experimaestro.scheduler.state_provider.CarbonMetricsData
   :members:
```

## Listeners

### Listener

```{eval-rst}
.. autoclass:: experimaestro.scheduler.base.Listener
   :members:
```

## Exceptions

### DirtyGitError

```{eval-rst}
.. autoclass:: experimaestro.DirtyGitError
   :show-inheritance:
```

### FailedExperiment

```{eval-rst}
.. autoclass:: experimaestro.scheduler.experiment.FailedExperiment
   :show-inheritance:
```

### GracefulExperimentExit

```{eval-rst}
.. autoclass:: experimaestro.scheduler.experiment.GracefulExperimentExit
   :show-inheritance:
```

## Carbon Tracking

### CarbonImpactData

```{eval-rst}
.. autoclass:: experimaestro.carbon.base.CarbonImpactData
   :members:
```
