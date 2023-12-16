# Running experiments

The main class is `experimaestro.experiment`


::: experimaestro.experiment

## Experiment services

::: experimaestro.scheduler.services.Service
::: experimaestro.scheduler.services.WebService
::: experimaestro.scheduler.services.ServiceListener


## Experiment configuration

The module `experimaestro.experiments` contain code factorizing boilerplate for
launching experiments


### Example

An `experiment.py` file:

```py3
    from xpmir.experiments.ir import PaperResults, ir_experiment, ExperimentHelper
    from xpmir.papers import configuration

    @configuration
    class Configuration:
        #: Default learning rate
        learning_rate: float = 1e-3

    @ir_experiment()
    def run(
        helper: ExperimentHelper, cfg: Configuration
    ) -> PaperResults:
        ...

        return PaperResults(
            models={"my-model@RR10": outputs.listeners[validation.id]["RR@10"]},
            evaluations=tests,
            tb_logs={"my-model@RR10": learner.logpath},
        )
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
