# Serialization

This page discusses how to save and load configuration objects:

1. Saving configuration
1. How to specify files/directories to be serialized
1. HuggingFace integration


## Saving/Loading objects with configurations

Configuration objects can be loaded and saved. You can even embed them
within any standard Python structure (i.e. dictionary, list, tuple).

```python
from experimaestro import load, save

# Saves the object
save([obj1, obj2, {key: obj3, key2: obj4}], "/my/folder")

# Load the object
[obj1, obj2, obj_dict] = load("/my/folder")
```

- {py:func}`~experimaestro.load` - Load configuration objects from a folder.
- {py:func}`~experimaestro.save` - Save configuration objects to a folder.

You can use `serialization` methods to include init_tasks
in the deserialize process. This makes it easier to distribute
configurations that need to be initialized in a special way.

- {py:func}`~experimaestro.serialize` - Serialize configurations with init tasks.
- {py:func}`~experimaestro.deserialize` - Deserialize configurations with init tasks.

A task configuration/instance can be loaded with {py:func}`~experimaestro.from_task_dir`.

The serialization context is controlled by {py:class}`~experimaestro.SerializationContext`.

If you need more control over saved data, you can use `state_dict`
and `from_state_dict` that respectively returns Python data structures
and loads from them.

- {py:func}`~experimaestro.state_dict` - Convert configurations to Python data structures.
- {py:func}`~experimaestro.from_state_dict` - Load configurations from Python data structures.


### Saving/Loading from running experiment

To ease saving/loading configuration from experiments, one
can use methods from the experiment objects as follows:

```python
from experimaestro import experiment, Param, Config

class MyConfig(Config):
    a: Param[int]

if __name__ == "__main__":
    # Saving configurations
    with experiment("/tmp/load_save", "xp1", port=-1) as xp:
        cfg = MyConfig.C(a=1)
        xp.save([cfg])


    # Loading configurations
    with experiment("/tmp/load_save", "xp2", port=-1) as xp:
        # Loads MyConfig(a=1)
        cfg, = xp.load("xp1")
```


## Specifying paths to be serialized

Configurations can be serialized with the data necessary
to restore their state. This can be useful to share a
model (e.g. with HuggingFace hub).

Use `DataPath` to annotate fields whose file or directory content
should be copied into the save directory during serialization:

```python
from experimaestro import Config, DataPath

class MyConfig(Config):
    to_serialize: DataPath
    """This path will be serialized alongside the configuration"""
```

When saving, each `DataPath` field is copied (using hard links when
possible) into the save directory under a relative path derived from
the field name. On loading, the relative path is resolved back to an
absolute path.

`DataPath` fields are **ignored in identifier computation** — changing
the data path does not change the task identifier.


### Custom data serialization

For more control over which files are serialized and where they are
stored, override the `__xpm_serialize__` method on your `Config` subclass.
This method receives a {py:class}`~experimaestro.SerializationContext` and
returns a dict mapping names to
{py:class}`~experimaestro.core.context.SerializedPath` objects.

By default, it serializes all `DataPath` fields. You can override it to
change destination paths, add extra data files, or skip certain fields:

```python
from pathlib import Path
from experimaestro import Config, Param, DataPath
from experimaestro.core.context import SerializationContext, SerializedPath

class MyModel(Config):
    name: Param[str]
    weights: DataPath

    def __xpm_serialize__(self, context: SerializationContext) -> dict[str, SerializedPath]:
        # Call super() for default DataPath handling
        result = super().__xpm_serialize__(context)

        # Or customize: serialize weights under a different name
        result["weights"] = context.serialize(
            context.var_path + ["model_weights.bin"],
            self.weights,
            self,
        )

        # Add extra files not declared as DataPath
        vocab_path = self.weights.parent / "vocab.txt"
        if vocab_path.exists():
            result["vocab"] = context.serialize(
                context.var_path + ["vocab.txt"],
                vocab_path,
                self,
            )

        return result
```

The {py:class}`~experimaestro.SerializationContext` can also be subclassed
to customize the serialization process globally (e.g. to change how files
are copied or where they are stored). The `serialize` method receives the
config object, enabling per-config path logic.


## HuggingFace integration

```python
# ExperimaestroHFHub implements the interface from ModelHubMixin
# https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.ModelHubMixin
from experimaestro.huggingface import ExperimaestroHFHub

# True if the object should be an instance (and not a configuration)
as_instance = False

# Save and load a configuration
ExperimaestroHFHub(config).push_to_hub(hf_id)
ExperimaestroHFHub.from_pretrained(hf_id_or_folder, as_instance=as_instance)

# Save and load a configuration (with a variant)
ExperimaestroHFHub(config).push_to_hub(hf_id, variant)
ExperimaestroHFHub.from_pretrained(hf_id_or_folder, variant=variant, as_instance=as_instance)
```

{py:class}`~experimaestro.huggingface.ExperimaestroHFHub` - HuggingFace Hub integration for experimaestro configurations. Key methods: `from_pretrained()`, `push_to_hub()`.

### Customizing HuggingFace serialization

Subclass `ExperimaestroHFHub` to customize the definition filename
or the serialization context:

```python
from experimaestro.huggingface import ExperimaestroHFHub
from experimaestro.core.context import SerializationContext

class MyHFHub(ExperimaestroHFHub):
    # Use a custom filename for the definition JSON
    definition_filename = "my_model.json"

    # Use a custom SerializationContext subclass
    serialization_context_class = MySerializationContext
```

- `definition_filename`: The JSON file storing the configuration definition
  (default: `"experimaestro.json"`, with fallback to `"definition.json"` on load)
- `serialization_context_class`: The `SerializationContext` class used during
  serialization (default: `SerializationContext`)
