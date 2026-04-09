from pathlib import Path
from typing import Union
from experimaestro.core.context import SerializationContext, SerializedPath
from experimaestro.core.objects import ConfigInformation, ConfigMixin
import os

try:
    from huggingface_hub import ModelHubMixin, hf_hub_download, snapshot_download
except ImportError:
    raise ImportError(
        "huggingface-hub is required for ExperimaestroHFHub. "
        "Install it with: pip install experimaestro[huggingface]"
    )


class ExperimaestroHFHub(ModelHubMixin):
    """Defines models that can be uploaded/downloaded from the Hub

    Subclass to customize serialization behavior:
    - ``definition_filename``: Override the JSON definition filename
      (default: ``"experimaestro.json"``)
    - ``serialization_context_class``: Override the SerializationContext class
    """

    #: The filename used for the definition JSON file
    definition_filename: str = "experimaestro.json"

    #: The SerializationContext class to use for serialization
    serialization_context_class: type[SerializationContext] = SerializationContext

    def __init__(self, config: ConfigMixin):
        self.config = config

    def _save_pretrained(self, save_directory: Union[str, Path]):
        save_directory = Path(save_directory)
        assert self.config is not None
        context = self.serialization_context_class(save_directory=save_directory)
        self.config.__xpm__.serialize(
            save_directory,
            definition_filename=self.definition_filename,
            context=context,
        )

    @classmethod
    def _from_pretrained(
        cls,
        model_id,
        revision=None,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=None,
        local_files_only=False,
        token=None,
        *,
        as_instance: bool = False,
        **model_kwargs,
    ):
        if os.path.isdir(model_id):
            save_directory = Path(model_id)

            def data_loader(path: Union[Path, str, SerializedPath]):
                if isinstance(path, SerializedPath):
                    path = path.path
                else:
                    path = Path(path)
                return save_directory / path

        else:

            def data_loader(s_path: Union[Path, str, SerializedPath]):
                if not isinstance(s_path, SerializedPath):
                    s_path = SerializedPath(Path(s_path), False)
                path = s_path.path

                # Folder
                if s_path.is_folder:
                    hf_path = snapshot_download(
                        repo_id=model_id,
                        allow_patterns=f"{s_path.path}/**",
                        revision=revision,
                        cache_dir=cache_dir,
                        proxies=proxies,
                        resume_download=resume_download,
                        token=token,
                        local_files_only=local_files_only,
                    )
                    return Path(hf_path) / path

                hf_path = Path(
                    hf_hub_download(
                        repo_id=model_id,
                        filename=str(path),
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        token=token,
                        local_files_only=local_files_only,
                    )
                )
                return hf_path

        return ConfigInformation.deserialize(
            data_loader,
            as_instance=as_instance,
            partial_loading=True,
            definition_filename=cls.definition_filename,
        )


__all__ = ["ExperimaestroHFHub"]
