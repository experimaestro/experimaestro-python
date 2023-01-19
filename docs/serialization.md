# Serialization

Configurations can be serialized with the data necessary
to restore their state. This can be useful to share a
model (e.g. with HuggingFace hub).

```py3
from experimaestro import DataPath

class MyConfig(Config):
    to_serialize: DataPath[Path]
    """This path will be serialized on the hub"""
```

## HuggingFace integration

```py3
# ExperimaestroHFHub implements the interface from ModelHubMixin
# https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.ModelHubMixin
from experimaestro.huggingface import ExperimaestroHFHub

# Save and load a configuration
ExperimaestroHFHub(config).push_to_hub(hf_id)
ExperimaestroHFHub().from_pretrained(hf_id_or_folder)

# Save and load a configuration (with a variant)
ExperimaestroHFHub(config).push_to_hub(hf_id, variant)
ExperimaestroHFHub().from_pretrained(hf_id_or_folder, variant=variant)
```
