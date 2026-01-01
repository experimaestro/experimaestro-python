# API Reference

This section provides the API documentation for the experimaestro module.

## Core Classes

The core classes form the foundation of experimaestro's configuration and task system.

### Config

```{eval-rst}
.. autoclass:: experimaestro.Config
   :members: C, XPMConfig, XPMValue, value_class, __validate__, __post_init__, __identifier__, copy_dependencies, register_task_output
   :show-inheritance:
```

### Task

```{eval-rst}
.. autoclass:: experimaestro.Task
   :members: execute, submit, watch_output, on_completed
   :show-inheritance:
```

### ResumableTask

```{eval-rst}
.. autoclass:: experimaestro.ResumableTask
   :members: remaining_time
   :show-inheritance:
```

### LightweightTask

```{eval-rst}
.. autoclass:: experimaestro.LightweightTask
   :members: execute
   :show-inheritance:
```

### InstanceConfig

```{eval-rst}
.. autoclass:: experimaestro.InstanceConfig
   :show-inheritance:
```

## Type Annotations

Type annotations are used to declare parameters in configurations and tasks.

### Param

```{eval-rst}
.. data:: experimaestro.Param

   Type annotation for configuration parameters.

   Parameters annotated with ``Param[T]`` are included in the configuration
   identifier computation and must be set before the configuration is sealed.
```

### Meta

```{eval-rst}
.. data:: experimaestro.Meta

   Type annotation for meta-parameters (ignored in identifier computation).

   Use ``Meta[T]`` for parameters that should not affect the task identity,
   such as output paths or runtime configuration.
```

### Constant

```{eval-rst}
.. data:: experimaestro.Constant

   Type annotation for constant (read-only) parameters.

   Constants must have a default value and cannot be modified after creation.
```

### DataPath

```{eval-rst}
.. data:: experimaestro.DataPath

   Type annotation for data paths that should be serialized.

   Use ``DataPath`` for paths that point to data files that should be
   preserved when serializing/deserializing a configuration.
```

### DependentMarker

```{eval-rst}
.. data:: experimaestro.DependentMarker

   Type alias for dependency marker functions used in ``task_outputs()``
   and dynamic output methods.
```

## Experiment Management

### experiment

```{eval-rst}
.. autoclass:: experimaestro.experiment
   :members: current, submit, wait, add_service
   :show-inheritance:
```

### Workspace

```{eval-rst}
.. autoclass:: experimaestro.Workspace
   :members:
   :show-inheritance:
```

### RunMode

```{eval-rst}
.. autoclass:: experimaestro.RunMode
   :members:
```

## Tagging

### tag

```{eval-rst}
.. autofunction:: experimaestro.tag
```

### tags

```{eval-rst}
.. autofunction:: experimaestro.tags
```

### tagspath

```{eval-rst}
.. autofunction:: experimaestro.tagspath
```

## Utilities

### setmeta

```{eval-rst}
.. autofunction:: experimaestro.setmeta
```

### cache

```{eval-rst}
.. autofunction:: experimaestro.cache
```

### initializer

```{eval-rst}
.. autofunction:: experimaestro.initializer
```

### tqdm

```{eval-rst}
.. autofunction:: experimaestro.tqdm
```

### progress

```{eval-rst}
.. autofunction:: experimaestro.progress
```

## Field Definitions

### field

```{eval-rst}
.. autoclass:: experimaestro.field
   :members:
```

### param_group

```{eval-rst}
.. autofunction:: experimaestro.param_group
```

### subparameters

```{eval-rst}
.. autofunction:: experimaestro.subparameters
```

### PathGenerator

```{eval-rst}
.. autoclass:: experimaestro.PathGenerator
   :members:
```

## Deprecation

### deprecate

```{eval-rst}
.. autofunction:: experimaestro.deprecate
```

## Exceptions

### GracefulTimeout

```{eval-rst}
.. autoclass:: experimaestro.GracefulTimeout
   :show-inheritance:
```

## Serialization

### save

```{eval-rst}
.. autofunction:: experimaestro.save
```

### load

```{eval-rst}
.. autofunction:: experimaestro.load
```

### serialize

```{eval-rst}
.. autofunction:: experimaestro.serialize
```

### deserialize

```{eval-rst}
.. autofunction:: experimaestro.deserialize
```

### from_task_dir

```{eval-rst}
.. autofunction:: experimaestro.from_task_dir
```

### state_dict

```{eval-rst}
.. autofunction:: experimaestro.state_dict
```

### from_state_dict

```{eval-rst}
.. autofunction:: experimaestro.from_state_dict
```

### SerializationContext

```{eval-rst}
.. autoclass:: experimaestro.SerializationContext
   :members:
```

## Launchers

### DirectLauncher

```{eval-rst}
.. autoclass:: experimaestro.launchers.direct.DirectLauncher
   :members:
   :show-inheritance:
```

### SlurmLauncher

```{eval-rst}
.. autoclass:: experimaestro.launchers.slurm.SlurmLauncher
   :members:
   :show-inheritance:
```

### SlurmOptions

```{eval-rst}
.. autoclass:: experimaestro.launchers.slurm.SlurmOptions
   :members:
```

## Connectors

### LocalConnector

```{eval-rst}
.. autoclass:: experimaestro.connectors.local.LocalConnector
   :members:
   :show-inheritance:
```

## Launcher Finder

### find_launcher

```{eval-rst}
.. autofunction:: experimaestro.launcherfinder.find_launcher
```

### parse

```{eval-rst}
.. autofunction:: experimaestro.launcherfinder.parser.parse
```

### HostRequirement

```{eval-rst}
.. autoclass:: experimaestro.launcherfinder.specs.HostRequirement
   :members:
```

### HostSpecification

```{eval-rst}
.. autoclass:: experimaestro.launcherfinder.specs.HostSpecification
   :members:
```

### CudaSpecification

```{eval-rst}
.. autoclass:: experimaestro.launcherfinder.specs.CudaSpecification
   :members:
```

### CPUSpecification

```{eval-rst}
.. autoclass:: experimaestro.launcherfinder.specs.CPUSpecification
   :members:
```

## Experiments CLI

### ExperimentHelper

```{eval-rst}
.. autoclass:: experimaestro.experiments.cli.ExperimentHelper
   :members:
```

### ConfigurationBase

```{eval-rst}
.. autoclass:: experimaestro.experiments.ConfigurationBase
   :members:
```

### configuration

```{eval-rst}
.. autofunction:: experimaestro.experiments.configuration
```

## HuggingFace Integration

### ExperimaestroHFHub

```{eval-rst}
.. autoclass:: experimaestro.huggingface.ExperimaestroHFHub
   :members: from_pretrained, push_to_hub
```

## Submit Hooks

### SubmitHook

```{eval-rst}
.. autoclass:: experimaestro.core.types.SubmitHook
   :members:
```

## Services

### Service

```{eval-rst}
.. autoclass:: experimaestro.scheduler.services.Service
   :members:
```

### WebService

```{eval-rst}
.. autoclass:: experimaestro.scheduler.services.WebService
   :members:
   :show-inheritance:
```

### ServiceState

```{eval-rst}
.. autoclass:: experimaestro.scheduler.services.ServiceState
   :members:
```

## Jobs

### Job

```{eval-rst}
.. autoclass:: experimaestro.scheduler.jobs.Job
   :members:
```

## Base Launcher

### Launcher

```{eval-rst}
.. autoclass:: experimaestro.launchers.Launcher
   :members:
   :show-inheritance:
```

## Click Integration

### forwardoption

```{eval-rst}
.. autoclass:: experimaestro.click.forwardoption
   :members:
```
