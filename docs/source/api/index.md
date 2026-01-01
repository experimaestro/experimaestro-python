# API Reference

This section provides the API documentation for the experimaestro module.

## Core Classes

### Config and Task

```{eval-rst}
.. autoclass:: experimaestro.Config
   :members:
   :show-inheritance:

.. autoclass:: experimaestro.Task
   :members:
   :show-inheritance:

.. autoclass:: experimaestro.ResumableTask
   :members:
   :show-inheritance:

.. autoclass:: experimaestro.LightweightTask
   :members:
   :show-inheritance:

.. autoclass:: experimaestro.InstanceConfig
   :members:
   :show-inheritance:
```

### Type Annotations

```{eval-rst}
.. autodata:: experimaestro.Param

.. autodata:: experimaestro.Meta

.. autodata:: experimaestro.Constant
```

### Experiment Management

```{eval-rst}
.. autofunction:: experimaestro.experiment

.. autoclass:: experimaestro.Workspace
   :members:
   :show-inheritance:
```

### Utilities

```{eval-rst}
.. autofunction:: experimaestro.tag

.. autofunction:: experimaestro.tagspath

.. autofunction:: experimaestro.setmeta

.. autofunction:: experimaestro.cache

.. autofunction:: experimaestro.initializer
```

### Field Definitions

```{eval-rst}
.. autofunction:: experimaestro.field

.. autofunction:: experimaestro.param_group

.. autofunction:: experimaestro.subparameters
```

### Deprecation

```{eval-rst}
.. autofunction:: experimaestro.deprecate
```

### Exceptions

```{eval-rst}
.. autoclass:: experimaestro.GracefulTimeout
   :members:
   :show-inheritance:
```
