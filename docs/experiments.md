# Running experiments

The main class is `experimaestro.experiment`


::: experimaestro.experiment

## Experiment services

::: experimaestro.scheduler.services.Service
::: experimaestro.scheduler.services.WebService
::: experimaestro.scheduler.services.ServiceListener


## Experiment configuration

The module `experimaestro.experiments` contain code factorizing boilerplate for
launching experiments. It allows to setup the experimental environment and
read ``YAML`` configuration files to setup some experimental parameters.

This can be extended to support more specific experiment helpers (see e.g.
experimaestro-ir for an example).

### Example

An `experiment.py` file:

```py3
    from experimaestro.experiments import ExperimentHelper, configuration

    @configuration
    class Configuration:
        #: Default learning rate
        learning_rate: float = 1e-3

    def run(
        helper: ExperimentHelper, cfg: Configuration
    ):
        # Experimental code
        ...
```

With `full.yaml` located in the same folder as `experiment.py`

```yaml
    file: experiment
    learning_rate: 1e-4
```

The experiment can be started with

```sh
    experimaestro run-experiment --run-mode normal full.yaml
```

### Common handling

::: experimaestro.experiments.cli
