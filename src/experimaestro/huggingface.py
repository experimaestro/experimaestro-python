from pathlib import Path
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
    """Defines models that can be uploaded/downloaded from the Hub.

    Subclass to customize serialization behavior:

    - ``definition_filename``: Override the JSON definition filename (default: ``"experimaestro.json"``).
    - ``serialization_context_class``: Override the SerializationContext class.
    """

    #: The filename used for the definition JSON file
    definition_filename: str = "experimaestro.json"

    #: The SerializationContext class to use for serialization
    serialization_context_class: type[SerializationContext] = SerializationContext

    def __init__(self, config: ConfigMixin):
        self.config = config

    def _save_pretrained(self, save_directory: str | Path):
        save_directory = Path(save_directory)
        assert self.config is not None
        context = self.serialization_context_class(save_directory=save_directory)
        self.config.__xpm__.serialize(
            save_directory,
            definition_filename=self.definition_filename,
            context=context,
        )
        try:
            self.generate_dag_html(save_directory)
        except Exception as e:
            import logging
            logging.getLogger("experimaestro").debug(
                "Failed to generate dag.html during save: %s", e
            )

    def generate_dag_html(
        self, save_directory: str | Path, html_filename: str = "dag.html"
    ) -> Path | None:
        """Generate a standalone interactive HTML DAG view in the export directory."""
        from experimaestro.ui.graph import load_serialized_graph, standalone_html

        save_path = Path(save_directory)
        def_file = save_path / self.definition_filename
        if not def_file.is_file():
            return None

        graph = load_serialized_graph(def_file)
        statuses = {n.id: "done" for n in graph.nodes.values() if n.kind.is_job}
        html_content = standalone_html(graph, statuses)
        out = save_path / html_filename
        out.write_text(html_content, encoding="utf-8")
        return out

    def get_dag_mermaid(
        self,
        save_directory: str | Path | None = None,
        orientation: str = "TD",
    ) -> str | None:
        """Generate a Mermaid flowchart markdown string for this model's DAG."""
        from experimaestro.ui.graph import load_serialized_graph, to_mermaid

        if save_directory is not None:
            def_file = Path(save_directory) / self.definition_filename
            if def_file.is_file():
                graph = load_serialized_graph(def_file)
                return to_mermaid(graph, orientation=orientation)
        return None

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

            def data_loader(path: Path | str | SerializedPath):
                if isinstance(path, SerializedPath):
                    path = path.path
                else:
                    path = Path(path)
                return save_directory / path

        else:

            def data_loader(s_path: Path | str | SerializedPath):
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
