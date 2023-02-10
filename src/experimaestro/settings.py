import os
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional


@dataclass
class ServerSettings:
    port: Optional[int] = None
    """Port for the server"""

    host: Optional[str] = None
    """Hostname for the server"""

    token: Optional[str] = None
    """Token for the server"""


@dataclass
class Settings:
    server: ServerSettings = field(default_factory=ServerSettings)


@lru_cache()
def get_settings(path: Optional[Path] = None) -> Settings:
    if "PYTEST_CURRENT_TEST" in os.environ:
        return Settings()
    else:
        schema = OmegaConf.structured(Settings)

        path = path or Path("~/.config/experimaestro/settings.yaml").expanduser()
        if not path.is_file():
            return schema

        conf = OmegaConf.load(path)
        return OmegaConf.merge(schema, conf)
