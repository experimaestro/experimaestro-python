"""Streaming serialization of Config objects to objects.jsonl.

Replaces the old configs.json approach by streaming objects incrementally
as jobs are submitted and actions are added, rather than batching at
experiment finalization.

Each line in objects.jsonl is a single serialized object dict. The root
object ID for each job/action is stored in jobs.jsonl or status.json
so it can be looked up during deserialization.
"""

import json
import logging
from pathlib import Path
from typing import Any

from experimaestro.core.context import SerializationContext
from experimaestro.core.objects import ConfigMixin

logger = logging.getLogger("xpm.objects_writer")


class ObjectsWriter:
    """Streams serialized Config objects to a JSONL file.

    Uses a shared :class:`SerializationContext` across all writes so that
    shared Config references are only serialized once. Each serialized
    object is written as a separate line.
    """

    def __init__(self, path: Path):
        self._path = path
        self._context = SerializationContext(save_directory=None)
        self._file = path.open("w")

    def write(self, config: "ConfigMixin") -> None:
        """Serialize a Config object and append its objects to the JSONL file.

        :param config: The Config object to serialize
        """
        from experimaestro.core.serialization import json_object

        # Each call gets a fresh objects list; context.serialized ensures
        # shared objects are only serialized once across all writes
        objects: list[dict[str, Any]] = []
        json_object(self._context, config, objects)

        # Write each object as a separate line
        for obj in objects:
            self._file.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the file handle."""
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
