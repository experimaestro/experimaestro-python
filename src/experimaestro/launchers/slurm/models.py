"""Data models for SLURM cluster configuration with override support.

This module provides a clean separation between:
1. Cluster data - raw information from SLURM commands (read-only)
2. Field-based configuration - tracks value sources (cluster, override, inferred)
3. Configured entities - partition/feature objects using Fields for value management

The Field class provides:
- Value access with source tracking (cluster default, user override, inferred)
- Override and reset functionality
- Inference support (value computed from other fields)
- Read-only protection for cluster values

The data flow is:
    SLURM commands -> ClusterData -> ConfiguredPartition/Feature (with Fields) -> save overrides
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

import yaml

T = TypeVar("T")


# =============================================================================
# Field System for Value Tracking
# =============================================================================


class FieldSource(Enum):
    """Source of a field's current value."""

    CLUSTER = auto()  # From SLURM cluster data (read-only)
    DEFAULT = auto()  # Default value (no cluster, no override)
    OVERRIDE = auto()  # User-set value
    INFERRED = auto()  # Computed from other fields/settings


@dataclass
class SettingDefinition:
    """Definition of a toggleable setting."""

    key: str  # Unique key for the setting
    category: str  # Category for grouping (e.g., "Partitions", "Launcher")
    label: str  # Display label
    description: str  # Help text
    get_value: Callable[[], bool]  # Get current value
    set_value: Callable[[bool], None]  # Set value


class Field(Generic[T]):
    """A field that tracks value source and supports override/inference.

    Fields can have values from multiple sources with this priority:
    1. Inferred (if inference is enabled and infer_func returns non-None)
    2. Override (user-set value)
    3. Cluster (from SLURM, read-only)
    4. Default (fallback)

    Usage:
        field = Field[str](cluster_value="v100", default="unknown")
        field.set("a100")  # Override the value
        field.value  # Returns "a100"
        field.is_overridden  # Returns True
        field.reset()  # Back to cluster value
    """

    def __init__(
        self,
        *,
        cluster_value: T | None = None,
        default: T | None = None,
        read_only: bool = False,
    ):
        self._cluster_value = cluster_value
        self._default = default
        self._override_value: T | None = None
        self._infer_func: Callable[[], T | None] | None = None
        self._infer_enabled: bool = False
        self._read_only = read_only

    @property
    def value(self) -> T | None:
        """Get the effective value based on priority."""
        if self._infer_enabled and self._infer_func:
            result = self._infer_func()
            if result is not None:
                return result
        if self._override_value is not None:
            return self._override_value
        if self._cluster_value is not None:
            return self._cluster_value
        return self._default

    @property
    def cluster_value(self) -> T | None:
        """Get the cluster value (read-only source)."""
        return self._cluster_value

    @property
    def source(self) -> FieldSource:
        """Get the source of the current effective value."""
        if self._infer_enabled and self._infer_func:
            result = self._infer_func()
            if result is not None:
                return FieldSource.INFERRED
        if self._override_value is not None:
            return FieldSource.OVERRIDE
        if self._cluster_value is not None:
            return FieldSource.CLUSTER
        return FieldSource.DEFAULT

    @property
    def is_overridden(self) -> bool:
        """Check if value is from user override."""
        return self.source == FieldSource.OVERRIDE

    @property
    def is_inferred(self) -> bool:
        """Check if value is inferred from other fields."""
        return self.source == FieldSource.INFERRED

    @property
    def infer_enabled(self) -> bool:
        """Check if inference mode is enabled (regardless of result)."""
        return self._infer_enabled

    @property
    def can_edit(self) -> bool:
        """Check if value can be edited (not read-only, not in infer mode)."""
        return not self._read_only and not self._infer_enabled

    def set(self, value: T) -> None:
        """Set the override value. Disables inference mode."""
        if self._read_only:
            raise ValueError("Field is read-only")
        self._override_value = value
        self._infer_enabled = False

    def set_infer_func(self, func: Callable[[], T | None]) -> None:
        """Set the function used for inference."""
        self._infer_func = func

    def enable_inference(self) -> None:
        """Enable inference mode. Value will be computed from infer_func."""
        if self._infer_func is None:
            raise ValueError("No inference function set")
        self._infer_enabled = True

    def disable_inference(self) -> None:
        """Disable inference mode. Falls back to override or cluster value."""
        self._infer_enabled = False

    def reset(self) -> None:
        """Reset to cluster/default value (clear override and disable inference)."""
        self._override_value = None
        self._infer_enabled = False

    @property
    def override_value(self) -> T | None:
        """Get the raw override value (for saving). None if not overridden."""
        return self._override_value


class ListField(Field[list[T]]):
    """Field specialized for list values.

    Always returns a list (empty if no value set).
    Handles list comparison and merging for inference.
    """

    def __init__(
        self,
        *,
        cluster_value: list[T] | None = None,
        read_only: bool = False,
    ):
        super().__init__(cluster_value=cluster_value, default=None, read_only=read_only)

    @property
    def value(self) -> list[T]:
        """Get the effective value (always returns a list)."""
        result = super().value
        return list(result) if result is not None else []

    @property
    def cluster_value(self) -> list[T]:
        """Get the cluster value (always returns a list)."""
        return list(self._cluster_value) if self._cluster_value else []

    def set(self, value: list[T]) -> None:
        """Set the override value. Makes a copy of the list."""
        if self._read_only:
            raise ValueError("Field is read-only")
        self._override_value = list(value) if value else None
        self._infer_enabled = False

    @property
    def is_overridden(self) -> bool:
        """Check if value is from user override (non-empty list)."""
        return (
            self._override_value is not None
            and len(self._override_value) > 0
            and not self._infer_enabled
        )


class BoolField(Field[bool]):
    """Field specialized for boolean values with a default."""

    def __init__(
        self,
        *,
        default: bool = True,
        read_only: bool = False,
    ):
        super().__init__(cluster_value=None, default=default, read_only=read_only)

    @property
    def value(self) -> bool:
        """Get the effective value (always returns a bool)."""
        result = super().value
        return result if result is not None else self._default

    @property
    def is_overridden(self) -> bool:
        """Check if value differs from default."""
        return (
            self._override_value is not None and self._override_value != self._default
        )


class IntField(Field[int]):
    """Field specialized for integer values with a default."""

    def __init__(
        self,
        *,
        cluster_value: int | None = None,
        default: int = 0,
        read_only: bool = False,
    ):
        super().__init__(
            cluster_value=cluster_value, default=default, read_only=read_only
        )
        self._int_default = default

    @property
    def value(self) -> int:
        """Get the effective value (always returns an int)."""
        result = super().value
        return result if result is not None else self._int_default

    @property
    def is_overridden(self) -> bool:
        """Check if value differs from default."""
        return (
            self._override_value is not None
            and self._override_value != self._int_default
        )


# =============================================================================
# Cluster Data (from SLURM commands)
# =============================================================================


@dataclass
class ClusterPartition:
    """Raw partition information from SLURM cluster.

    This represents the actual state of a partition as reported by SLURM.
    Users cannot modify these values directly - they come from the cluster.
    """

    name: str
    nodes: str  # Node list pattern
    cpus_per_node: int
    memory_mb: int  # Memory in MB
    gpus_per_node: int
    gpu_type: str | None  # e.g., "v100", "a100" (from GRES)
    time_limit: str  # SLURM format: "D-HH:MM:SS" or "HH:MM:SS"
    time_limit_seconds: int | None
    available: bool
    features: list[str] = field(default_factory=list)
    # QoS/Account restrictions (from scontrol show partition)
    allow_qos: list[str] | None = None  # None means ALL allowed
    deny_qos: list[str] = field(default_factory=list)
    allow_accounts: list[str] | None = None  # None means ALL allowed
    deny_accounts: list[str] = field(default_factory=list)

    @property
    def memory_bytes(self) -> int:
        return self.memory_mb * (1024**2)

    def is_qos_allowed(self, qos_name: str) -> bool:
        """Check if a QoS is allowed on this partition."""
        if qos_name in self.deny_qos:
            return False
        if self.allow_qos is None:
            return True  # All allowed
        return qos_name in self.allow_qos

    def is_account_allowed(self, account_name: str) -> bool:
        """Check if an account is allowed on this partition."""
        if account_name in self.deny_accounts:
            return False
        if self.allow_accounts is None:
            return True  # All allowed
        return account_name in self.allow_accounts


@dataclass
class ClusterQoS:
    """Raw QoS information from SLURM cluster."""

    name: str
    max_wall: str | None  # Time limit string
    max_wall_seconds: int | None
    priority: int
    # GRES limits (e.g., ["gpu:v100=32", "gpu:a100=16"])
    gres_limits: list[str] = field(default_factory=list)
    # Raw TRES limits for reference
    max_tres_per_job: str | None = None
    max_tres_per_user: str | None = None

    @property
    def gpu_types(self) -> list[str]:
        """Extract GPU types from gres_limits (e.g., 'gpu:v100=32' -> 'v100')."""
        import re

        types = []
        for limit in self.gres_limits:
            match = re.match(r"gpu:(\w+)=", limit)
            if match:
                gpu_type = match.group(1).lower()
                if gpu_type not in types:
                    types.append(gpu_type)
        return types


@dataclass
class ClusterAccount:
    """User's account association from SLURM cluster."""

    account: str
    partition: str | None  # None means all partitions
    qos_list: list[str] = field(default_factory=list)


class ConfiguredQoS:
    """QoS configuration with user-defined tags.

    Wraps ClusterQoS with user-configurable fields like tags.
    """

    def __init__(self, cluster: ClusterQoS):
        self._cluster = cluster
        self.tags: ListField[str] = ListField()

    @property
    def name(self) -> str:
        return self._cluster.name

    @property
    def max_wall(self) -> str | None:
        return self._cluster.max_wall

    @property
    def max_wall_seconds(self) -> int | None:
        return self._cluster.max_wall_seconds

    @property
    def priority(self) -> int:
        return self._cluster.priority

    @property
    def gres_limits(self) -> list[str]:
        return self._cluster.gres_limits

    @property
    def gpu_types(self) -> list[str]:
        return self._cluster.gpu_types

    def to_save_dict(self) -> dict[str, Any] | None:
        """Get data for saving (only overridden values)."""
        data: dict[str, Any] = {}
        if self.tags.is_overridden:
            data["tags"] = self.tags.override_value
        return data if data else None

    def load_from_dict(self, data: dict[str, Any]) -> None:
        """Load configuration from saved data."""
        if tags_list := data.get("tags"):
            self.tags.set(tags_list)


class ConfiguredAccount:
    """Account configuration with user-defined tags.

    Wraps ClusterAccount with user-configurable fields like tags.
    """

    def __init__(self, cluster: ClusterAccount):
        self._cluster = cluster
        self.tags: ListField[str] = ListField()

    @property
    def account(self) -> str:
        return self._cluster.account

    @property
    def partition(self) -> str | None:
        return self._cluster.partition

    @property
    def qos_list(self) -> list[str]:
        return self._cluster.qos_list

    @property
    def key(self) -> str:
        """Unique key for this account association."""
        return f"{self.account}:{self.partition or 'all'}"

    def to_save_dict(self) -> dict[str, Any] | None:
        """Get data for saving (only overridden values)."""
        data: dict[str, Any] = {}
        if self.tags.is_overridden:
            data["tags"] = self.tags.override_value
        return data if data else None

    def load_from_dict(self, data: dict[str, Any]) -> None:
        """Load configuration from saved data."""
        if tags_list := data.get("tags"):
            self.tags.set(tags_list)


@dataclass
class ClusterData:
    """Complete SLURM cluster data from commands.

    This is read-only data representing the actual cluster state.
    """

    cluster_name: str
    partitions: dict[str, ClusterPartition] = field(default_factory=dict)
    qos: dict[str, ClusterQoS] = field(default_factory=dict)
    accounts: list[ClusterAccount] = field(default_factory=list)
    default_account: str | None = None
    features: set[str] = field(default_factory=set)
    # SLURM configuration from scontrol show config (priority weights, etc.)
    slurm_config: dict[str, any] = field(default_factory=dict)


# =============================================================================
# Configured Entities (using Field-based value tracking)
# =============================================================================


class ConfiguredFeature:
    """Feature configuration using Field-based value tracking.

    Features can map to GPU types/memory, CPU cores, memory, and restrict QoS/accounts.
    All values are tracked with source information (cluster, override, inferred).
    """

    def __init__(self, name: str):
        self.name = name
        # CPU cores
        self.cores: IntField = IntField(default=0)
        # Memory in GB
        self.memory_gb: IntField = IntField(default=0)
        # GPU type mapping (e.g., "v100", "a100")
        self.gpu_type: Field[str | None] = Field(default=None)
        # Number of GPUs
        self.gpu_count: IntField = IntField(default=0)
        # GPU memory in GB
        self.gpu_memory_gb: IntField = IntField(default=0)
        # Restrict to specific QoS
        self.allowed_qos: ListField[str] = ListField()
        # Restrict to specific accounts
        self.allowed_accounts: ListField[str] = ListField()
        # Tags for filtering
        self.tags: ListField[str] = ListField()

    def has_any_config(self) -> bool:
        """Check if this feature has any overridden configuration."""
        return (
            self.cores.is_overridden
            or self.memory_gb.is_overridden
            or self.gpu_type.is_overridden
            or self.gpu_count.is_overridden
            or self.gpu_memory_gb.is_overridden
            or self.allowed_qos.is_overridden
            or self.allowed_accounts.is_overridden
            or self.tags.is_overridden
        )

    def to_save_dict(self) -> dict[str, Any] | None:
        """Get data for saving (only overridden values). Returns None if nothing to save."""
        data: dict[str, Any] = {}
        if self.cores.is_overridden:
            data["cores"] = self.cores.override_value
        if self.memory_gb.is_overridden:
            data["memory_gb"] = self.memory_gb.override_value
        if self.gpu_type.override_value:
            data["gpu_type"] = self.gpu_type.override_value
        if self.gpu_count.is_overridden:
            data["gpu_count"] = self.gpu_count.override_value
        if self.gpu_memory_gb.is_overridden:
            data["gpu_memory_gb"] = self.gpu_memory_gb.override_value
        if self.allowed_qos.is_overridden:
            data["allowed_qos"] = self.allowed_qos.override_value
        if self.allowed_accounts.is_overridden:
            data["allowed_accounts"] = self.allowed_accounts.override_value
        if self.tags.is_overridden:
            data["tags"] = self.tags.override_value
        return data if data else None

    def load_from_dict(self, data: dict[str, Any] | str) -> None:
        """Load configuration from saved data."""
        if isinstance(data, str):
            # Simple string means just GPU type (backward compat)
            self.gpu_type.set(data)
            return

        if (cores := data.get("cores", 0)) > 0:
            self.cores.set(cores)
        if (memory := data.get("memory_gb", 0)) > 0:
            self.memory_gb.set(memory)
        if gpu_type := data.get("gpu_type"):
            self.gpu_type.set(gpu_type)
        if (gpu_count := data.get("gpu_count", 0)) > 0:
            self.gpu_count.set(gpu_count)
        if (gpu_mem := data.get("gpu_memory_gb", 0)) > 0:
            self.gpu_memory_gb.set(gpu_mem)
        if qos_list := data.get("allowed_qos"):
            self.allowed_qos.set(qos_list)
        if accounts_list := data.get("allowed_accounts"):
            self.allowed_accounts.set(accounts_list)
        if tags_list := data.get("tags"):
            self.tags.set(tags_list)


class ConfiguredPartition:
    """Partition configuration using Field-based value tracking.

    Combines cluster data with user overrides, tracking the source of each value.
    Supports inference of QoS/accounts from partition features.
    """

    def __init__(self, cluster: ClusterPartition, config: "SlurmConfig"):
        self._cluster = cluster
        self._config = config  # Parent config for feature lookups

        # User-configurable fields
        self.enabled: BoolField = BoolField(default=True)
        self.cores: IntField = IntField(cluster_value=cluster.cpus_per_node, default=0)
        self.memory_gb: IntField = IntField(
            cluster_value=cluster.memory_mb // 1024, default=0
        )
        self.gpu_type: Field[str | None] = Field(cluster_value=cluster.gpu_type)
        self.gpu_memory_gb: IntField = IntField(default=0)
        self.priority: IntField = IntField(default=10)
        self.tags: ListField[str] = ListField()
        self.allowed_qos: ListField[str] = ListField()
        self.allowed_accounts: ListField[str] = ListField()

    # Read-only cluster properties
    @property
    def name(self) -> str:
        return self._cluster.name

    @property
    def cpus_per_node(self) -> int:
        return self._cluster.cpus_per_node

    @property
    def memory_mb(self) -> int:
        return self._cluster.memory_mb

    @property
    def memory_bytes(self) -> int:
        return self._cluster.memory_bytes

    @property
    def gpus_per_node(self) -> int:
        return self._cluster.gpus_per_node

    @property
    def gpu_memory_bytes(self) -> int:
        """Get GPU memory in bytes (from user config only)."""
        return self.gpu_memory_gb.value * (1024**3)

    @property
    def time_limit(self) -> str:
        return self._cluster.time_limit

    @property
    def time_limit_seconds(self) -> int | None:
        return self._cluster.time_limit_seconds

    @property
    def features(self) -> list[str]:
        return self._cluster.features

    @property
    def cluster_allow_qos(self) -> list[str] | None:
        """Get cluster-level QoS restrictions (None = all allowed)."""
        return self._cluster.allow_qos

    @property
    def cluster_allow_accounts(self) -> list[str] | None:
        """Get cluster-level account restrictions (None = all allowed)."""
        return self._cluster.allow_accounts

    def _compute_inferred_cores(self) -> int:
        """Compute cores from partition's features."""
        for feature_name in self.features:
            feature = self._config.features.get(feature_name)
            if feature and feature.cores.value > 0:
                return feature.cores.value
        return 0

    def _compute_inferred_memory(self) -> int:
        """Compute memory (GB) from partition's features."""
        for feature_name in self.features:
            feature = self._config.features.get(feature_name)
            if feature and feature.memory_gb.value > 0:
                return feature.memory_gb.value
        return 0

    def _compute_inferred_qos(self) -> list[str]:
        """Compute QoS list from partition's features."""
        qos_set: set[str] = set()
        for feature_name in self.features:
            feature = self._config.features.get(feature_name)
            if feature and feature.allowed_qos.value:
                qos_set.update(feature.allowed_qos.value)
        return sorted(qos_set)

    def _compute_inferred_accounts(self) -> list[str]:
        """Compute accounts list from partition's features."""
        accounts_set: set[str] = set()
        for feature_name in self.features:
            feature = self._config.features.get(feature_name)
            if feature and feature.allowed_accounts.value:
                accounts_set.update(feature.allowed_accounts.value)
        return sorted(accounts_set)

    def _compute_inferred_gpu_type(self) -> str | None:
        """Compute GPU type from partition's features."""
        for feature_name in self.features:
            feature = self._config.features.get(feature_name)
            if feature and feature.gpu_type.value:
                return feature.gpu_type.value
        return None

    def _compute_inferred_gpu_memory(self) -> int:
        """Compute GPU memory from partition's features."""
        for feature_name in self.features:
            feature = self._config.features.get(feature_name)
            if feature and feature.gpu_memory_gb.value > 0:
                return feature.gpu_memory_gb.value
        return 0

    @property
    def cores_is_inferred(self) -> bool:
        """Check if cores is inferred from global setting."""
        return (
            self._config.infer_cores_from_features.value
            and not self.cores.is_overridden
        )

    @property
    def memory_is_inferred(self) -> bool:
        """Check if memory is inferred from global setting."""
        return (
            self._config.infer_memory_from_features.value
            and not self.memory_gb.is_overridden
        )

    @property
    def gpu_type_is_inferred(self) -> bool:
        """Check if GPU type is inferred from global setting."""
        return (
            self._config.infer_gpu_type_from_features.value
            and not self.gpu_type.is_overridden
        )

    @property
    def gpu_memory_is_inferred(self) -> bool:
        """Check if GPU memory is inferred from global setting."""
        return (
            self._config.infer_gpu_memory_from_features.value
            and not self.gpu_memory_gb.is_overridden
        )

    @property
    def effective_cores(self) -> int:
        """Get effective cores (override, inferred, or cluster)."""
        if self.cores.is_overridden:
            return self.cores.value
        if self._config.infer_cores_from_features.value:
            inferred = self._compute_inferred_cores()
            if inferred > 0:
                return inferred
        return self._cluster.cpus_per_node

    @property
    def effective_memory_gb(self) -> int:
        """Get effective memory in GB (override, inferred, or cluster)."""
        if self.memory_gb.is_overridden:
            return self.memory_gb.value
        if self._config.infer_memory_from_features.value:
            inferred = self._compute_inferred_memory()
            if inferred > 0:
                return inferred
        return self._cluster.memory_mb // 1024

    @property
    def effective_gpu_type(self) -> str | None:
        """Get effective GPU type (inferred, override, or cluster)."""
        if (
            self._config.infer_gpu_type_from_features.value
            and not self.gpu_type.is_overridden
        ):
            inferred = self._compute_inferred_gpu_type()
            if inferred:
                return inferred
        return self.gpu_type.value

    @property
    def effective_gpu_memory_gb(self) -> int:
        """Get effective GPU memory in GB (inferred, override, or 0)."""
        if (
            self._config.infer_gpu_memory_from_features.value
            and not self.gpu_memory_gb.is_overridden
        ):
            inferred = self._compute_inferred_gpu_memory()
            if inferred > 0:
                return inferred
        return self.gpu_memory_gb.value

    @property
    def qos_is_inferred(self) -> bool:
        """Check if QoS is inferred (inference enabled and no override)."""
        return (
            self._config.infer_qos_from_features.value
            and not self.allowed_qos.is_overridden
        )

    @property
    def accounts_is_inferred(self) -> bool:
        """Check if accounts are inferred (inference enabled and no override)."""
        return (
            self._config.infer_accounts_from_features.value
            and not self.allowed_accounts.is_overridden
        )

    @property
    def effective_qos(self) -> list[str]:
        """Get effective QoS list (override, inferred, or empty)."""
        # Override takes priority
        if self.allowed_qos.is_overridden:
            return self.allowed_qos.value
        # Then inference
        if self._config.infer_qos_from_features.value:
            return self._compute_inferred_qos()
        return self.allowed_qos.value

    @property
    def effective_accounts(self) -> list[str]:
        """Get effective accounts list (override, inferred, or empty)."""
        # Override takes priority
        if self.allowed_accounts.is_overridden:
            return self.allowed_accounts.value
        # Then inference
        if self._config.infer_accounts_from_features.value:
            return self._compute_inferred_accounts()
        return self.allowed_accounts.value

    def get_allowed_qos(self) -> list[str] | None:
        """Get effective allowed QoS list.

        Returns user override/inferred if set, else cluster restrictions.
        None means all QoS are allowed.
        """
        effective = self.effective_qos
        if effective:
            return effective
        return self._cluster.allow_qos

    def get_allowed_accounts(self) -> list[str] | None:
        """Get effective allowed accounts list.

        Returns user override/inferred if set, else cluster restrictions.
        None means all accounts are allowed.
        """
        effective = self.effective_accounts
        if effective:
            return effective
        return self._cluster.allow_accounts

    def is_qos_allowed(self, qos_name: str) -> bool:
        """Check if a QoS is allowed on this partition."""
        effective = self.effective_qos
        if effective:
            return qos_name in effective
        return self._cluster.is_qos_allowed(qos_name)

    def is_account_allowed(self, account_name: str) -> bool:
        """Check if an account is allowed on this partition."""
        effective = self.effective_accounts
        if effective:
            return account_name in effective
        return self._cluster.is_account_allowed(account_name)

    def to_save_dict(self) -> dict[str, Any] | None:
        """Get data for saving (only overridden values). Returns None if nothing to save."""
        data: dict[str, Any] = {}

        if not self.enabled.value:
            data["enabled"] = False
        if self.gpu_type.is_overridden:
            data["gpu_type"] = self.gpu_type.override_value
        if self.gpu_memory_gb.is_overridden:
            data["gpu_memory_gb"] = self.gpu_memory_gb.override_value
        if self.priority.is_overridden:
            data["priority"] = self.priority.override_value
        if self.tags.is_overridden:
            data["tags"] = self.tags.override_value
        if self.allowed_qos.is_overridden:
            data["allowed_qos"] = self.allowed_qos.override_value
        if self.allowed_accounts.is_overridden:
            data["allowed_accounts"] = self.allowed_accounts.override_value

        return data if data else None

    def load_from_dict(self, data: dict[str, Any]) -> None:
        """Load configuration from saved data."""
        if "enabled" in data:
            self.enabled.set(data["enabled"])
        if gpu_type := data.get("gpu_type"):
            self.gpu_type.set(gpu_type)
        if (gpu_mem := data.get("gpu_memory_gb", 0)) > 0:
            self.gpu_memory_gb.set(gpu_mem)
        if "priority" in data:
            self.priority.set(data["priority"])
        if tags := data.get("tags"):
            self.tags.set(tags)
        if qos_list := data.get("allowed_qos"):
            self.allowed_qos.set(qos_list)
        if accounts_list := data.get("allowed_accounts"):
            self.allowed_accounts.set(accounts_list)


class SlurmConfig:
    """Main SLURM configuration using Field-based value tracking.

    This is the primary interface for accessing and modifying configuration.
    Replaces the old EffectiveConfig by unifying cluster data and user overrides.
    """

    def __init__(self, cluster: ClusterData):
        self._cluster = cluster

        # Global settings
        self.cluster_name = cluster.cluster_name
        self.default_account: Field[str | None] = Field(
            cluster_value=cluster.default_account
        )
        self.consider_priority: BoolField = BoolField(default=True)

        # Global inference settings
        self.infer_cores_from_features: BoolField = BoolField(default=False)
        self.infer_memory_from_features: BoolField = BoolField(default=False)
        self.infer_gpu_type_from_features: BoolField = BoolField(default=False)
        self.infer_gpu_memory_from_features: BoolField = BoolField(default=False)
        self.infer_qos_from_features: BoolField = BoolField(default=False)
        self.infer_accounts_from_features: BoolField = BoolField(default=False)

        # Launcher settings
        self.gpu_proportional_resources: BoolField = BoolField(default=False)

        # Features (created on demand)
        self.features: dict[str, ConfiguredFeature] = {}

        # QoS (created from cluster data)
        self.qos: dict[str, ConfiguredQoS] = {}
        for name, cq in cluster.qos.items():
            self.qos[name] = ConfiguredQoS(cq)

        # Accounts (created from cluster data)
        self.accounts: dict[str, ConfiguredAccount] = {}
        for ca in cluster.accounts:
            acc = ConfiguredAccount(ca)
            self.accounts[acc.key] = acc

        # Partitions (created from cluster data)
        self.partitions: dict[str, ConfiguredPartition] = {}
        for name, cp in cluster.partitions.items():
            self.partitions[name] = ConfiguredPartition(cp, self)

    @property
    def cluster(self) -> ClusterData:
        """Access to underlying cluster data."""
        return self._cluster

    @property
    def accounts_list(self) -> list[ConfiguredAccount]:
        """Get account associations."""
        return list(self.accounts.values())

    @property
    def all_features(self) -> set[str]:
        """Get all features from cluster, partitions, and config."""
        all_features = set(self._cluster.features)
        for partition in self.partitions.values():
            all_features.update(partition.features)
        # Include features defined in config (may not be in cluster)
        all_features.update(self.features.keys())
        return all_features

    @property
    def enabled_partitions(self) -> dict[str, ConfiguredPartition]:
        """Get only enabled partitions."""
        return {name: p for name, p in self.partitions.items() if p.enabled.value}

    def get_feature(self, name: str) -> ConfiguredFeature:
        """Get or create a feature configuration."""
        if name not in self.features:
            self.features[name] = ConfiguredFeature(name)
        return self.features[name]

    def get_partition(self, name: str) -> ConfiguredPartition | None:
        """Get a partition by name."""
        return self.partitions.get(name)

    def get_gpu_type_for_feature(self, feature: str) -> str | None:
        """Get the GPU type mapped to a feature."""
        f = self.features.get(feature)
        return f.gpu_type.value if f else None

    def get_settings(self) -> list[SettingDefinition]:
        """Get list of global configurable settings."""
        settings = [
            SettingDefinition(
                key="infer_cores",
                category="Partitions",
                label="Infer cores from features",
                description="CPU cores computed from feature configurations",
                get_value=lambda: self.infer_cores_from_features.value,
                set_value=lambda v: self.infer_cores_from_features.set(v),
            ),
            SettingDefinition(
                key="infer_memory",
                category="Partitions",
                label="Infer memory from features",
                description="Memory computed from feature configurations",
                get_value=lambda: self.infer_memory_from_features.value,
                set_value=lambda v: self.infer_memory_from_features.set(v),
            ),
            SettingDefinition(
                key="infer_gpu_type",
                category="Partitions",
                label="Infer GPU type from features",
                description="GPU type computed from feature configurations",
                get_value=lambda: self.infer_gpu_type_from_features.value,
                set_value=lambda v: self.infer_gpu_type_from_features.set(v),
            ),
            SettingDefinition(
                key="infer_gpu_memory",
                category="Partitions",
                label="Infer GPU memory from features",
                description="GPU memory computed from feature configurations",
                get_value=lambda: self.infer_gpu_memory_from_features.value,
                set_value=lambda v: self.infer_gpu_memory_from_features.set(v),
            ),
        ]
        # Only add QoS/Accounts inference if they exist
        if self.qos:
            settings.append(
                SettingDefinition(
                    key="infer_qos",
                    category="Partitions",
                    label="Infer QoS from features",
                    description="QoS lists computed from feature configurations",
                    get_value=lambda: self.infer_qos_from_features.value,
                    set_value=lambda v: self.infer_qos_from_features.set(v),
                )
            )
        if self.accounts:
            settings.append(
                SettingDefinition(
                    key="infer_accounts",
                    category="Partitions",
                    label="Infer accounts from features",
                    description="Account lists computed from feature configurations",
                    get_value=lambda: self.infer_accounts_from_features.value,
                    set_value=lambda v: self.infer_accounts_from_features.set(v),
                )
            )
        settings.append(
            SettingDefinition(
                key="consider_priority",
                category="Launcher",
                label="Consider priority in selection",
                description="Higher priority partitions preferred",
                get_value=lambda: self.consider_priority.value,
                set_value=lambda v: self.consider_priority.set(v),
            )
        )
        settings.append(
            SettingDefinition(
                key="gpu_proportional_resources",
                category="Launcher",
                label="GPU-proportional CPU resources",
                description="CPU cores/memory allocated proportionally to GPU count",
                get_value=lambda: self.gpu_proportional_resources.value,
                set_value=lambda v: self.gpu_proportional_resources.set(v),
            )
        )
        return settings

    @classmethod
    def from_cluster_and_file(
        cls, cluster: ClusterData, config_path: Path | None = None
    ) -> "SlurmConfig":
        """Create config from cluster data and optional config file."""
        config = cls(cluster)
        if config_path and config_path.exists():
            config.load_from_yaml(config_path)
        return config

    def load_from_yaml(self, path: Path) -> None:  # noqa: C901
        """Load user configuration from YAML file."""
        if not path.exists():
            return

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Load global settings
        if default_account := data.get("default_account"):
            self.default_account.set(default_account)
        if "consider_priority" in data:
            self.consider_priority.set(data["consider_priority"])
        if data.get("infer_cores_from_features"):
            self.infer_cores_from_features.set(True)
        if data.get("infer_memory_from_features"):
            self.infer_memory_from_features.set(True)
        if data.get("infer_gpu_type_from_features"):
            self.infer_gpu_type_from_features.set(True)
        if data.get("infer_gpu_memory_from_features"):
            self.infer_gpu_memory_from_features.set(True)
        if data.get("infer_qos_from_features"):
            self.infer_qos_from_features.set(True)
        if data.get("infer_accounts_from_features"):
            self.infer_accounts_from_features.set(True)
        if data.get("gpu_proportional_resources"):
            self.gpu_proportional_resources.set(True)

        # Load feature configurations
        for name, fdata in data.get("features", {}).items():
            feature = self.get_feature(name)
            feature.load_from_dict(fdata)

        # Backward compatibility: load old feature_mappings format
        for name, gpu_type in data.get("feature_mappings", {}).items():
            if name not in self.features:
                feature = self.get_feature(name)
                feature.gpu_type.set(gpu_type)

        # Load QoS configurations
        for name, qdata in data.get("qos", {}).items():
            if name in self.qos:
                self.qos[name].load_from_dict(qdata)

        # Load account configurations
        for key, adata in data.get("accounts", {}).items():
            if key in self.accounts:
                self.accounts[key].load_from_dict(adata)

        # Load partition configurations
        for name, pdata in data.get("partitions", {}).items():
            if name in self.partitions:
                self.partitions[name].load_from_dict(pdata)

    def save_to_yaml(self, path: Path) -> None:  # noqa: C901
        """Save user configuration to YAML file (only overridden values)."""
        data: dict[str, Any] = {
            "cluster_name": self.cluster_name,
        }

        # Save global settings if overridden
        if self.consider_priority.is_overridden:
            data["consider_priority"] = self.consider_priority.value
        if self.infer_cores_from_features.value:
            data["infer_cores_from_features"] = True
        if self.infer_memory_from_features.value:
            data["infer_memory_from_features"] = True
        if self.infer_gpu_type_from_features.value:
            data["infer_gpu_type_from_features"] = True
        if self.infer_gpu_memory_from_features.value:
            data["infer_gpu_memory_from_features"] = True
        if self.infer_qos_from_features.value:
            data["infer_qos_from_features"] = True
        if self.infer_accounts_from_features.value:
            data["infer_accounts_from_features"] = True
        if self.gpu_proportional_resources.value:
            data["gpu_proportional_resources"] = True
        if self.default_account.is_overridden:
            data["default_account"] = self.default_account.override_value

        # Save feature configurations
        features_data = {}
        for name in sorted(self.features.keys()):
            feature = self.features[name]
            if fdata := feature.to_save_dict():
                features_data[name] = fdata
        if features_data:
            data["features"] = features_data

        # Save QoS configurations (only those with tags)
        qos_data = {}
        for name in sorted(self.qos.keys()):
            qos = self.qos[name]
            if qdata := qos.to_save_dict():
                qos_data[name] = qdata
        if qos_data:
            data["qos"] = qos_data

        # Save account configurations (only those with tags)
        accounts_data = {}
        for key in sorted(self.accounts.keys()):
            acc = self.accounts[key]
            if adata := acc.to_save_dict():
                accounts_data[key] = adata
        if accounts_data:
            data["accounts"] = accounts_data

        # Save partition configurations (all partitions to track "seen" state)
        partitions_data = {}
        for name in sorted(self.partitions.keys()):
            partition = self.partitions[name]
            # Save overrides or empty dict to mark partition as "seen"
            partitions_data[name] = partition.to_save_dict() or {}
        if partitions_data:
            data["partitions"] = partitions_data

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    def get_new_partitions(self, saved_path: Path) -> list[str]:
        """Get list of partitions in cluster but not in saved config file."""
        if not saved_path.exists():
            return list(self.partitions.keys())

        with open(saved_path) as f:
            saved_data = yaml.safe_load(f) or {}

        saved_partitions = set(saved_data.get("partitions", {}).keys())
        return [name for name in self.partitions if name not in saved_partitions]


# =============================================================================
# Legacy compatibility aliases (deprecated)
# =============================================================================

# For backward compatibility with existing code that uses the old API
EffectiveConfig = SlurmConfig
EffectivePartition = ConfiguredPartition
