# Grid Search

Grid search allows you to explore a range of parameters for your experiments. Experimaestro provides a unified way to define search spaces directly in your configuration classes and YAML files.

## Defining Search Spaces

To enable grid search for a configuration field, use the `GridSearch[T]` type hint. This allows the field to accept either a single value (of type `T`) or a search space definition.

```python
from experimaestro.experiments import ConfigurationBase, configuration
from experimaestro.experiments.grid import GridSearch

@configuration
class MyConfig(ConfigurationBase):
    learning_rate: GridSearch[float] = 0.001
    batch_size: GridSearch[int] = 32
```

### Inline Lists

You can define a list of values directly in your YAML configuration:

```yaml
id: my-experiment
learning_rate: [0.1, 0.01, 0.001]
#or yaml list fashion
batch_size:
  - 16
  - 32
```

### Value Ranges

You can also define a range of values using a dictionary:

```yaml
id: my-experiment
learning_rate:
  values_range: [0, 10]  # values from 0 to 9
```

## Explicit Grid Search Block

Sometimes you want to define a grid search for parameters that are deeply nested or when you want to keep the search space definition separate from the main configuration values. You can use the `grid_search` block in your YAML:

```yaml
id: my-experiment
lr: 0.01
sub_config:
  param: 10

grid_search:
  lr: [0.1, 0.01]
  sub_config.param: [10, 20, 30]
```

The keys in the `grid_search` block are dot-separated paths to the fields in your configuration.

## Generating Permutations

The grid search is managed manually in your `run` function. This gives you full control over how to manage experiment IDs, logging, and task submission.

The `generate_grid` function returns two lists:
1.  **`configurations`**: A list of finalized configuration objects, each representing one permutation.
2.  **`tags`**: A list of dictionaries containing the specific parameter values that were set for each permutation.

```python
from experimaestro.experiments import ExperimentHelper, ConfigurationBase, configuration
from experimaestro.experiments.grid import GridSearch, generate_grid

@configuration
class MyConfig(ConfigurationBase):
    learning_rate: GridSearch[float] = 0.001

def run(helper: ExperimentHelper, cfg: MyConfig):
    # Generate all permutations and their corresponding tags
    configurations, all_tags = generate_grid(cfg)
    
    for config, tags in zip(configurations, all_tags):
        # Each permutation is a finalized MyConfig object
        # 'tags' contains the specific values used (e.g., {'learning_rate': 0.1})
        
        # Use tags to create a descriptive experiment ID
        tag_suffix = ".".join(f"{k}={v}" for k, v in tags.items())
        sub_id = f"{config.id}.{tag_suffix}" if tag_suffix else config.id
        
        # Run your experiment logic for this config
        ...
```

## How it Works

1.  **Validation**: When Experimaestro loads your YAML, the `GridSearch[T]` type hint uses a Pydantic validator to coerce the input (scalar, list, or range dict) into a `GenericParams` object.
2.  **Discovery**: The `generate_grid` function recursively scans your configuration to find all `GenericParams` that define a search space.
3.  **Permutations**: It computes the Cartesian product of all discovered search spaces.
4.  **Finalization**: For each permutation, it creates a deep copy of your configuration, injects the specific values for that permutation, and converts all `GenericParams` back to their underlying native Python types (`float`, `int`, etc.). This ensures that the rest of your code doesn't need to know about `GridSearch`.
