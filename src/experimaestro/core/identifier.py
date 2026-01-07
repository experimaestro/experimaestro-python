from contextlib import contextmanager
from enum import Enum
from functools import cached_property
import hashlib
import logging
import os
import struct
from typing import Optional, TYPE_CHECKING
from experimaestro.core.objects import Config, ConfigMixin

if TYPE_CHECKING:
    from experimaestro.core.partial import Partial


class ConfigPath:
    """Used to keep track of cycles when computing a hash"""

    def __init__(self):
        self.loops: list[bool] = []
        """Indicates whether a loop was detected up to this node"""

        self.config2index = {}
        """Associate an index in the list with a configuration"""

        self.instance_order: dict[bytes, list[int]] = {}
        """Maps InstanceConfig base identifier to list of object ids in encounter order"""

    def detect_loop(self, config) -> Optional[int]:
        """If there is a loop, return the relative index and update the path"""
        index = self.config2index.get(id(config), None)
        if index is not None:
            for i in range(index, self.depth):
                self.loops[i] = True
            return self.depth - index
        return None

    def has_loop(self):
        return self.loops[-1]

    @property
    def depth(self):
        return len(self.loops)

    @contextmanager
    def push(self, config):
        config_id = id(config)
        assert config_id not in self.config2index

        self.config2index[config_id] = self.depth
        self.loops.append(False)

        try:
            yield
        finally:
            self.loops.pop()
            del self.config2index[config_id]


hash_logger = logging.getLogger("xpm.hash")


def is_ignored(value):
    """Returns True if the value should be ignored by itself"""
    return value is not None and isinstance(value, Config) and (value.__xpm__.meta)


def remove_meta(value):
    """Cleanup a dict/list by removing ignored values"""
    if isinstance(value, list):
        return [el for el in value if not is_ignored(el)]
    if isinstance(value, dict):
        return {key: value for key, value in value.items() if not is_ignored(value)}
    return value


class Identifier:
    def __init__(self, main: bytes):
        self.main = main
        self.has_loops = False

    @cached_property
    def all(self):
        """Returns the overall identifier"""
        return self.main

    def __hash__(self) -> int:
        return hash(self.main)

    def state_dict(self):
        return self.main.hex()

    def __eq__(self, other: object):
        if not isinstance(other, Identifier):
            return False
        return self.main == other.main

    @staticmethod
    def from_state_dict(data: dict[str, str] | str):
        if isinstance(data, str):
            return Identifier(bytes.fromhex(data))

        return Identifier(bytes.fromhex(data["main"]))

    def __repr__(self):
        return self.main.hex()


class IdentifierComputer:
    """This class is in charge of computing a config/task identifier

    Args:
        config: The configuration to compute the identifier for
        config_path: Used to track cycles when computing identifiers
        version: Hash computation version (defaults to XPM_HASH_COMPUTER env var or 2)
        partial: If provided, only include parameters that are not excluded
            by this Partial instance (for partial identifier computation)
    """

    OBJECT_ID = b"\x00"
    INT_ID = b"\x01"
    FLOAT_ID = b"\x02"
    STR_ID = b"\x03"
    PATH_ID = b"\x04"
    NAME_ID = b"\x05"
    NONE_ID = b"\x06"
    LIST_ID = b"\x07"
    TASK_ID = b"\x08"
    DICT_ID = b"\x09"
    ENUM_ID = b"\x0a"
    CYCLE_REFERENCE = b"\x0b"
    INIT_TASKS = b"\x0c"
    INSTANCE_ID = b"\x0d"

    def __init__(
        self,
        config: "ConfigMixin",
        config_path: ConfigPath,
        *,
        version=None,
        partial: "Partial" = None,
    ):
        # Hasher for parameters
        self._hasher = hashlib.sha256()
        self.config = config
        self.config_path = config_path
        self.version = version or int(os.environ.get("XPM_HASH_COMPUTER", 2))
        self.partial = partial
        if hash_logger.isEnabledFor(logging.DEBUG):
            hash_logger.debug(
                "starting hash (%s): %s", hash(str(self.config)), self.config
            )

    def identifier(self) -> Identifier:
        main = self._hasher.digest()
        if hash_logger.isEnabledFor(logging.DEBUG):
            hash_logger.debug("hash (%s): %s", hash(str(self.config)), str(main))
        return Identifier(main)

    def _hashupdate(self, bytes: bytes):
        """Update the hash computers with some bytes"""
        if hash_logger.isEnabledFor(logging.DEBUG):
            hash_logger.debug(
                "updating hash (%s): %s", hash(str(self.config)), str(bytes)
            )
        self._hasher.update(bytes)

    def update(self, value, *, myself=False):  # noqa: C901
        """Update the hash

        :param value: The value to add to the hash
        :param myself: True if the value is the configuration for which we wish
            to compute the identifier, defaults to False
        :raises NotImplementedError: If the value cannot be processed
        """
        if value is None:
            self._hashupdate(IdentifierComputer.NONE_ID)
        elif isinstance(value, float):
            self._hashupdate(IdentifierComputer.FLOAT_ID)
            self._hashupdate(struct.pack("!d", value))
        elif isinstance(value, int):
            self._hashupdate(IdentifierComputer.INT_ID)
            self._hashupdate(struct.pack("!q", value))
        elif isinstance(value, str):
            self._hashupdate(IdentifierComputer.STR_ID)
            self._hashupdate(value.encode("utf-8"))
        elif isinstance(value, list):
            values = [el for el in value if not is_ignored(el)]
            self._hashupdate(IdentifierComputer.LIST_ID)
            self._hashupdate(struct.pack("!d", len(values)))
            for x in values:
                self.update(x)
        elif isinstance(value, Enum):
            self._hashupdate(IdentifierComputer.ENUM_ID)
            k = value.__class__
            self._hashupdate(
                f"{k.__module__}.{k.__qualname__}:{value.name}".encode("utf-8"),
            )
        elif isinstance(value, dict):
            self._hashupdate(IdentifierComputer.DICT_ID)
            items = [
                (key, value) for key, value in value.items() if not is_ignored(value)
            ]
            items.sort(key=lambda x: x[0])
            for key, value in items:
                self.update(key)
                self.update(value)

        # Handles configurations
        elif isinstance(value, ConfigMixin):
            # For deprecated configs with __convert__, compute identifier from converted config
            # This must happen BEFORE hashing OBJECT_ID to ensure identical identifiers
            if myself:
                xpmtype = value.__xpmtype__
                if xpmtype.deprecated and hasattr(value, "__convert__"):
                    hash_logger.debug(
                        "Computing hash via __convert__ for deprecated %s", xpmtype
                    )
                    converted = value.__convert__()
                    # Delegate entirely to the converted config
                    self.update(converted, myself=True)
                    return

            # Encodes the identifier
            self._hashupdate(IdentifierComputer.OBJECT_ID)

            # If we encode another config, then
            if not myself:
                if loop_ix := self.config_path.detect_loop(value):
                    # Loop detected: use cycle reference
                    self._hashupdate(IdentifierComputer.CYCLE_REFERENCE)
                    self._hashupdate(struct.pack("!q", loop_ix))

                else:
                    # Just use the object identifier
                    value_id = IdentifierComputer.compute(
                        value, version=self.version, config_path=self.config_path
                    )
                    self._hashupdate(value_id.all)

                    # Handle InstanceConfig: add instance order only if duplicate detected
                    from experimaestro.core.objects import InstanceConfig

                    if isinstance(value, InstanceConfig):
                        # Use the base identifier to track instances
                        base_id = value_id.all
                        obj_id = id(value)

                        # Track instances with this base identifier
                        if base_id not in self.config_path.instance_order:
                            # First occurrence - just record it (NO-OP for backwards compatibility)
                            self.config_path.instance_order[base_id] = [obj_id]
                        else:
                            # Check if this specific object was already seen
                            instances = self.config_path.instance_order[base_id]
                            if obj_id not in instances:
                                # New instance with same params - add to list
                                instances.append(obj_id)

                            # Get position in the list
                            position = instances.index(obj_id)

                            # Only add instance order if not the first (position > 0)
                            if position > 0:
                                self._hashupdate(IdentifierComputer.INSTANCE_ID)
                                self._hashupdate(struct.pack("!q", position))

                # And that's it!
                return

            # Process tasks
            if value.__xpm__.task is not None and (value.__xpm__.task is not value):
                hash_logger.debug("Computing hash for task %s", value.__xpm__.task)
                self._hashupdate(IdentifierComputer.TASK_ID)
                self.update(value.__xpm__.task)

            xpmtype = value.__xpmtype__
            self._hashupdate(xpmtype.identifier.name.encode("utf-8"))

            # Process arguments (sort by name to ensure uniqueness)
            arguments = sorted(xpmtype.arguments.values(), key=lambda a: a.name)
            for argument in arguments:
                # Skip arguments excluded by partial (for partial identifiers)
                if self.partial is not None:
                    if self.partial.is_excluded(argument.groups):
                        continue

                # Ignored argument
                if argument.ignored:
                    argvalue = value.__xpm__.values.get(argument.name, None)

                    # ... unless meta is set to false
                    if (
                        argvalue is None
                        or not isinstance(argvalue, Config)
                        or (argvalue.__xpm__.meta is not False)
                    ):
                        continue

                if argument.generator:
                    continue

                # Argument value
                # Skip if the argument is not a constant, and
                # - optional argument: both value and default are None
                # - the argument value is equal to the default value AND
                #   ignore_default_in_identifier is True
                try:
                    argvalue = getattr(value, argument.name, None)
                except KeyError:
                    logging.warning(
                        "Parameter %s has not been set in %s created at %s",
                        argument.name,
                        value,
                        value.__xpm__._initinfo,
                    )
                    raise
                if not argument.constant and (
                    (
                        not argument.required
                        and argument.default is None
                        and argvalue is None
                    )
                    or (
                        argument.ignore_default_in_identifier
                        and argument.default is not None
                        and argument.default == remove_meta(argvalue)
                    )
                ):
                    # No update if same value (and not constant)
                    continue

                if (
                    argvalue is not None
                    and isinstance(argvalue, Config)
                    and argvalue.__xpm__.meta
                ):
                    continue

                # Hash name
                self.update(argument.name)

                # Hash value
                self._hashupdate(IdentifierComputer.NAME_ID)
                self.update(argvalue)

            # Add init tasks
            if value.__xpm__.init_tasks:
                self._hashupdate(IdentifierComputer.INIT_TASKS)
                for init_task in value.__xpm__.init_tasks:
                    self.update(init_task)
        else:
            raise NotImplementedError("Cannot compute hash of type %s" % type(value))

    @staticmethod
    def compute(
        config: "ConfigMixin", config_path: ConfigPath | None = None, version=None
    ) -> Identifier:
        """Compute the identifier for a configuration

        :param config: the configuration for which we compute the identifier
        :param config_path: used to track down cycles between configurations
        :param version: version for the hash computation (None for the last one)
        """

        # Try to use the cached value first
        # (if there are no loops)
        if config.__xpm__._sealed:
            identifier = config.__xpm__._identifier
            if identifier is not None and not identifier.has_loops:
                return identifier

        config_path = config_path or ConfigPath()

        with config_path.push(config):
            self = IdentifierComputer(config, config_path, version=version)
            self.update(config, myself=True)
            identifier = self.identifier()
            identifier.has_loops = config_path.has_loop()

        return identifier

    @staticmethod
    def compute_partial(
        config: "ConfigMixin",
        partial: "Partial",
        config_path: ConfigPath | None = None,
        version=None,
    ) -> Identifier:
        """Compute a partial identifier for a configuration

        A partial identifier excludes certain parameter groups, allowing
        configurations that differ only in those groups to share the same
        partial identifier (and thus the same partial directory).

        :param config: the configuration for which we compute the identifier
        :param partial: the Partial instance defining which groups
            to include/exclude
        :param config_path: used to track down cycles between configurations
        :param version: version for the hash computation (None for the last one)
        """
        config_path = config_path or ConfigPath()

        with config_path.push(config):
            computer = IdentifierComputer(
                config, config_path, version=version, partial=partial
            )
            computer.update(config, myself=True)
            identifier = computer.identifier()
            identifier.has_loops = config_path.has_loop()

        return identifier
