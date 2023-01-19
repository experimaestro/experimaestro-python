from pathlib import Path
from typing import Optional, Union
from experimaestro import Config
from experimaestro.core.objects import ConfigInformation
from huggingface_hub import ModelHubMixin, hf_hub_download
import os


class ExperimaestroHFHub(ModelHubMixin):
    """Defines models that can be uploaded/downloaded from the Hub"""

    def __init__(self, config: Config, variant: Optional[str] = None):
        self.config = config
        self.variant = variant

    def _save_pretrained(self, save_directory: Union[str, Path]):
        save_directory = Path(save_directory)
        if self.variant:
            save_directory = save_directory / self.variant
            save_directory.mkdir()
        assert self.config is not None
        self.config.__xpm__.serialize(save_directory)

    @classmethod
    def _from_pretrained(
        cls,
        model_id,
        revision,
        cache_dir,
        force_download,
        proxies,
        resume_download,
        local_files_only,
        token,
        *,
        variant: Optional[str],
        **model_kwargs,
    ):
        if os.path.isdir(model_id):
            save_directory = Path(model_id)

            def data_loader(path: Path):
                if variant:
                    return save_directory / path / variant
                return save_directory / path

        else:

            def data_loader(path: Union[Path, str]):
                path = Path(path)
                hf_path = Path(
                    hf_hub_download(
                        repo_id=model_id,
                        filename=str(path if variant is None else Path(variant) / path),
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

        return ConfigInformation.deserialize(data_loader)
