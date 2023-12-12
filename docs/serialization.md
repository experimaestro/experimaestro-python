# Serialization

This page discusses how to save and load configuration objects:

1. Saving configuration
1. How to specify files/directories to be serialized
1. HuggingFace integration


## Saving/Loading objects with configurations

Configuration objects can be loaded and saved. You can even embed them
within any standard Python structure (i.e. dictionary, list, tuple).

```py3
from experimaestro import load, save

# Saves the object
save([obj1, obj2, {key: obj3, key2: obj4}], "/my/folder")

# Load the object
[obj1, obj2, obj_dict] = load("/my/folder")
```

::: experimaestro.load
::: experimaestro.save

The serialization context is controlled by a specific object
named `SerializationContext`:

::: experimaestro.SerializationContext


If you need more control over saved data, you can use `state_dict`
and `from_state_dict` that respectively returns Python data structures
and loads from them.

::: experimaestro.state_dict
::: experimaestro.from_state_dict


### Saving/Loading from running experiment

To ease saving/loading configuration from experiments, one
can use methods from the experiment objects as follows:

```py
from experimaestro import experiment, Param, Config

class MyConfig(Config):
    a: Param[int]

if __name__ == "__main__":
    # Saving configurations
    with experiment("/tmp/load_save", "xp1", port=-1) as xp:
        cfg = MyConfig(a=1)
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

```py3
from experimaestro import DataPath

class MyConfig(Config):
    to_serialize: DataPath
    """This path will be serialized on the hub"""
```

## HuggingFace integration

```py3
# ExperimaestroHFHub implements the interface from ModelHubMixin
# https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.ModelHubMixin
from experimaestro.huggingface import ExperimaestroHFHub

# True if the object should be an instance (and not a configuration)
as_instance = False

# Save and load a configuration
ExperimaestroHFHub(config).push_to_hub(hf_id)
ExperimaestroHFHub().from_pretrained(hf_id_or_folder, as_instance=as_instance)

# Save and load a configuration (with a variant)
ExperimaestroHFHub(config).push_to_hub(hf_id, variant)
ExperimaestroHFHub().from_pretrained(hf_id_or_folder, variant=variant, as_instance=as_instance)
```

::: experimaestro.huggingface.ExperimaestroHFHub
    options:
      members:
        - from_pretrained
        - push_to_hub
