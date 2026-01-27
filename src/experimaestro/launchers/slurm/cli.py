"""CLI commands for SLURM launcher configuration."""

from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import re
import sys

import click

# Import data models
from .models import (
    ClusterPartition,
    ClusterQoS,
    ClusterAccount,
    ClusterData,
)


def get_default_cache_file() -> Path:
    """Get platform-specific default cache file path for SLURM data."""
    if sys.platform == "win32":
        # Windows: %LOCALAPPDATA%\experimaestro\slurm_cache.json
        base = Path(os.environ.get("LOCALAPPDATA", "~/.cache")).expanduser()
        return base / "experimaestro" / "slurm_cache.json"
    elif sys.platform == "darwin":
        # macOS: ~/Library/Caches/experimaestro/slurm_cache.json
        return Path("~/Library/Caches/experimaestro/slurm_cache.json").expanduser()
    else:
        # Linux/Unix: $XDG_CACHE_HOME/experimaestro/slurm_cache.json
        base = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
        return base / "experimaestro" / "slurm_cache.json"


class CacheMissError(Exception):
    """Raised when a command is not found in cache and use_cache is True."""

    pass


# Cache keys for SLURM JSON data
CACHE_KEY_SINFO = "sinfo"
CACHE_KEY_PARTITIONS = "partitions"
CACHE_KEY_QOS = "qos"
CACHE_KEY_ASSOCIATIONS = "associations"
CACHE_KEY_CONFIG = "config"

# SLURM commands with their cache keys and output format
# Format: (command, output_type) where output_type is "json" or "config"
SLURM_COMMANDS: dict[str, tuple[str, str]] = {
    # sinfo: cluster info, nodes, features, GRES
    CACHE_KEY_SINFO: ("sinfo --json 2>/dev/null", "json"),
    # partitions: partition definitions, QoS/account restrictions
    CACHE_KEY_PARTITIONS: ("scontrol show partition --json 2>/dev/null", "json"),
    # qos: QoS info with TRES limits (for GPU type detection)
    CACHE_KEY_QOS: ("sacctmgr show qos --json 2>/dev/null", "json"),
    # associations: user associations (accounts, partitions, QoS)
    CACHE_KEY_ASSOCIATIONS: (
        "sacctmgr show assoc user=$USER --json 2>/dev/null",
        "json",
    ),
    # config: SLURM configuration (priority weights, etc.) - key=value format
    CACHE_KEY_CONFIG: ("scontrol show config 2>/dev/null", "config"),
}


@dataclass
class SlurmCommandCache:
    """Cache for SLURM JSON data.

    The cache stores parsed JSON directly with semantic keys:
    - sinfo: sinfo --json output
    - partitions: scontrol show partition --json output
    - qos: sacctmgr show qos --json output
    - associations: sacctmgr show assoc user=$USER --json output
    - config: scontrol show config (parsed key=value pairs)
    """

    cache_file: Path | None = None
    use_cache: bool = False
    _data: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self):
        if self.cache_file:
            self.cache_file = Path(self.cache_file)
            # If directory given, use default filename inside it
            if self.cache_file.is_dir() or (
                not self.cache_file.suffix and not self.cache_file.exists()
            ):
                self.cache_file = self.cache_file / "slurm_cache.json"
            if self.use_cache:
                self._load_cache()

    def _load_cache(self):
        """Load cached JSON data from disk."""
        if self.cache_file and self.cache_file.exists():
            with open(self.cache_file) as f:
                self._data = json.load(f)

    def save_cache(self):
        """Save JSON data to cache."""
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(self._data, f, indent=2)

    @property
    def cached_command_count(self) -> int:
        """Return the number of cached data keys."""
        return len(self._data)

    def get_json(self, key: str) -> dict:
        """Get cached JSON data by key."""
        if key in self._data:
            return self._data[key]

        if self.use_cache:
            raise CacheMissError(
                f"Data not found in cache: {key}"
                f"\nCache has keys: {list(self._data.keys())}"
            )
        return {}

    def set_json(self, key: str, data: dict):
        """Store JSON data in cache."""
        self._data[key] = data

    def run_json_command(self, key: str, command: str) -> dict:
        """Run a JSON command and cache the result, or return cached data."""
        if key in self._data:
            return self._data[key]

        if self.use_cache:
            raise CacheMissError(
                f"Data not found in cache: {key}"
                f"\nCache has keys: {list(self._data.keys())}"
            )

        import subprocess

        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
        )
        output = result.stdout
        data = json.loads(output) if output.strip() else {}
        self._data[key] = data
        return data

    def run_config_command(self, key: str, command: str) -> dict:
        """Run scontrol show config and parse key=value output into a dict."""
        if key in self._data:
            return self._data[key]

        if self.use_cache:
            raise CacheMissError(
                f"Data not found in cache: {key}"
                f"\nCache has keys: {list(self._data.keys())}"
            )

        import subprocess

        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
        )
        output = result.stdout

        # Parse key = value lines
        data = {}
        for line in output.splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                # Handle "Key = Value" format
                parts = line.split("=", 1)
                if len(parts) == 2:
                    key_name = parts[0].strip()
                    value = parts[1].strip()
                    # Try to convert numeric values
                    if value.isdigit():
                        data[key_name] = int(value)
                    elif value.replace(".", "", 1).isdigit():
                        data[key_name] = float(value)
                    else:
                        data[key_name] = value

        self._data[key] = data
        return data


def parse_time_to_seconds(time_str: str | None) -> int | None:
    """Parse SLURM time format to seconds.

    Formats: "D-HH:MM:SS", "HH:MM:SS", "MM:SS", "SS", "UNLIMITED"
    """
    if not time_str or time_str == "UNLIMITED":
        return None

    # Remove any trailing whitespace
    time_str = time_str.strip()

    days = 0
    if "-" in time_str:
        day_part, time_str = time_str.split("-", 1)
        days = int(day_part)

    parts = time_str.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        hours, minutes, seconds = 0, int(parts[0]), int(parts[1])
    else:
        hours, minutes, seconds = 0, 0, int(parts[0])

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def parse_gres(gres_str: str) -> tuple[int, str | None]:
    """Parse GRES string to get GPU count and type.

    Examples:
        "gpu:4(S:0-1)" -> (4, None)
        "gpu:h100:4(S:0-1)" -> (4, "h100")
        "(null)" -> (0, None)
    """
    if not gres_str or gres_str == "(null)":
        return 0, None

    # Match patterns like "gpu:4" or "gpu:h100:4"
    match = re.match(r"gpu:(?:(\w+):)?(\d+)", gres_str)
    if match:
        gpu_type = match.group(1)  # May be None
        gpu_count = int(match.group(2))
        return gpu_count, gpu_type

    return 0, None


def _parse_slurm_list_field(data: dict, *keys) -> list | None:
    """Try multiple keys and parse as list (comma-separated string or list).

    SLURM JSON uses various field names across versions - this tries multiple
    possibilities like nested (qos.allowed) or flat (allow_qos, allowed_qos).

    Returns None for "ALL" or empty values (meaning no restriction).
    """
    for key in keys:
        if "." in key:
            # Nested key like "qos.allowed"
            parts = key.split(".", 1)
            val = data.get(parts[0], {})
            if isinstance(val, dict):
                val = val.get(parts[1])
            else:
                continue
        else:
            val = data.get(key)

        if val is None:
            continue
        if isinstance(val, list):
            return val if val else None
        if isinstance(val, str):
            val = val.strip()
            if not val:
                return None
            if val.upper() == "ALL":
                return None  # ALL means no restriction
            return [v.strip() for v in val.split(",") if v.strip()]
    return None


def detect_cluster_info(cache: SlurmCommandCache | None = None) -> ClusterData:  # noqa: C901
    """Detect SLURM cluster configuration by running JSON commands.

    Args:
        cache: Optional cache for storing/retrieving command outputs
    """
    # Create a temporary cache if none provided
    if cache is None:
        cache = SlurmCommandCache()

    # Get sinfo data (JSON) - contains cluster name and resource details
    sinfo_data = cache.run_json_command(
        CACHE_KEY_SINFO, SLURM_COMMANDS[CACHE_KEY_SINFO][0]
    )

    # Get cluster name from sinfo meta
    cluster_name = sinfo_data.get("meta", {}).get("slurm", {}).get("cluster", "unknown")

    # Collect features per partition from sinfo entries
    partition_features: dict[str, set[str]] = {}

    def _parse_features(features_val) -> list[str]:
        """Parse features from various formats."""
        if isinstance(features_val, dict):
            features_str = features_val.get("total", "")
            return [f.strip() for f in features_str.split(",") if f.strip()]
        elif isinstance(features_val, str):
            return [f.strip() for f in features_val.split(",") if f.strip()]
        elif isinstance(features_val, list):
            return features_val
        return []

    # Try nodes first (some SLURM versions have this)
    for node in sinfo_data.get("nodes", []):
        node_partitions = node.get("partitions", [])
        node_features = _parse_features(node.get("features", []))
        for part_name in node_partitions:
            if part_name not in partition_features:
                partition_features[part_name] = set()
            partition_features[part_name].update(node_features)

    # Also collect from sinfo entries (main source for features)
    for entry in sinfo_data.get("sinfo", []):
        # Get partition name from partition object or node.partition_name
        part_obj = entry.get("partition", {})
        if isinstance(part_obj, dict):
            part_name = part_obj.get("name", "")
        else:
            part_name = entry.get("node", {}).get("partition_name", "")
        if not part_name:
            continue

        entry_features = _parse_features(entry.get("features", {}))
        if entry_features:
            if part_name not in partition_features:
                partition_features[part_name] = set()
            partition_features[part_name].update(entry_features)

    # Build lookup of sinfo data by configured node list (for resource info)
    # sinfo entries don't have partition.name directly, so we match by node list
    sinfo_by_nodes: dict[str, dict] = {}
    for entry in sinfo_data.get("sinfo", []):
        # Get the configured nodes from partition.nodes.configured
        part_obj = entry.get("partition", {})
        if isinstance(part_obj, dict):
            nodes_config = part_obj.get("nodes", {}).get("configured", "")
            if nodes_config and nodes_config not in sinfo_by_nodes:
                sinfo_by_nodes[nodes_config] = entry

    # Get partition definitions from scontrol (primary source for partition names and restrictions)
    partition_data = cache.run_json_command(
        CACHE_KEY_PARTITIONS, SLURM_COMMANDS[CACHE_KEY_PARTITIONS][0]
    )

    partitions: dict[str, ClusterPartition] = {}

    for part in partition_data.get("partitions", []):
        name = part.get("name", "")
        if not name:
            continue

        # Get state from partition.state
        state = part.get("partition", {}).get("state", ["UP"])
        if isinstance(state, str):
            state = [state]
        available = "UP" in [s.upper() for s in state]

        # Get resource info from scontrol (cpus.total, nodes, etc.)
        cpus_data = part.get("cpus", {})
        cpus = cpus_data.get("total", 0)

        # Get node list
        nodes_str = part.get("nodes", {}).get("configured", "")

        # Get TRES info for GPU detection
        tres_data = part.get("tres", {})
        tres_configured = tres_data.get("configured", "")

        # Parse memory from TRES (format: "cpu=N,mem=NM,node=N,...")
        memory_mb = 0
        if tres_configured:
            for item in tres_configured.split(","):
                if item.startswith("mem="):
                    mem_str = item[4:]
                    # Handle units: M, G, T
                    if mem_str.endswith("M"):
                        memory_mb = int(mem_str[:-1])
                    elif mem_str.endswith("G"):
                        memory_mb = int(mem_str[:-1]) * 1024
                    elif mem_str.endswith("T"):
                        memory_mb = int(mem_str[:-1]) * 1024 * 1024
                    else:
                        try:
                            memory_mb = int(mem_str)
                        except ValueError:
                            pass
                    break

        # Parse GPU from TRES (format includes "gres/gpu=N" or "gres/gpu:type=N")
        gpu_count = 0
        gpu_type = None
        if tres_configured:
            for item in tres_configured.split(","):
                if item.startswith("gres/gpu"):
                    # Could be "gres/gpu=4" or "gres/gpu:h100=4"
                    if ":" in item and "=" in item:
                        # gres/gpu:h100=4
                        type_part = item.split(":")[1]
                        if "=" in type_part:
                            gpu_type = type_part.split("=")[0]
                            gpu_count = int(type_part.split("=")[1])
                    elif "=" in item:
                        # gres/gpu=4
                        gpu_count = int(item.split("=")[1])
                    break

        # Get time limit from maximums or timeouts
        time_limit_seconds = None
        max_time = part.get("maximums", {}).get("time", {})
        if isinstance(max_time, dict):
            if max_time.get("set", False) and not max_time.get("infinite", False):
                time_limit_seconds = max_time.get("number", 0)
        elif isinstance(max_time, int):
            time_limit_seconds = max_time

        time_limit_str = ""
        if time_limit_seconds:
            days = time_limit_seconds // 86400
            hours = (time_limit_seconds % 86400) // 3600
            mins = (time_limit_seconds % 3600) // 60
            if days > 0:
                time_limit_str = f"{days}-{hours:02d}:{mins:02d}:00"
            else:
                time_limit_str = f"{hours:02d}:{mins:02d}:00"

        # Get features from nodes
        features_list = list(partition_features.get(name, []))

        # Enrich with sinfo data if available (match by node list)
        sinfo_entry = sinfo_by_nodes.get(nodes_str, {})
        if sinfo_entry:
            # Get better CPU/memory data from sinfo
            sinfo_cpus = sinfo_entry.get("cpus", {}).get("maximum", 0)
            if sinfo_cpus:
                cpus = sinfo_cpus
            sinfo_mem = sinfo_entry.get("memory", {}).get("maximum", 0)
            if isinstance(sinfo_mem, dict):
                sinfo_mem = sinfo_mem.get("number", 0)
            if sinfo_mem:
                memory_mb = sinfo_mem

            # Get GRES from sinfo
            gres_data = sinfo_entry.get("gres", {})
            if isinstance(gres_data, dict):
                gres_str = gres_data.get("total", "")
            else:
                gres_str = str(gres_data)
            if gres_str:
                sinfo_gpu_count, sinfo_gpu_type = parse_gres(gres_str)
                if sinfo_gpu_count:
                    gpu_count = sinfo_gpu_count
                if sinfo_gpu_type:
                    gpu_type = sinfo_gpu_type

            # Get features from sinfo entry
            sinfo_features = sinfo_entry.get("features", {})
            if isinstance(sinfo_features, dict):
                features_str = sinfo_features.get("total", "")
                if features_str:
                    for f in features_str.split(","):
                        f = f.strip()
                        if f and f not in features_list:
                            features_list.append(f)

        # GPU type can be set manually in the generated slurm.yaml config
        # based on the features shown in the info command

        # Parse QoS and account restrictions
        allow_qos = _parse_slurm_list_field(
            part, "qos.allowed", "allow_qos", "allowed_qos", "qos_allow"
        )
        deny_qos = _parse_slurm_list_field(part, "qos.deny", "deny_qos", "qos_deny")
        allow_accounts = _parse_slurm_list_field(
            part,
            "accounts.allowed",
            "allow_accounts",
            "allowed_accounts",
            "accounts_allow",
        )
        deny_accounts = _parse_slurm_list_field(
            part, "accounts.deny", "deny_accounts", "accounts_deny"
        )

        partitions[name] = ClusterPartition(
            name=name,
            nodes=nodes_str,
            cpus_per_node=cpus,
            memory_mb=memory_mb,
            gpus_per_node=gpu_count,
            gpu_type=gpu_type,
            time_limit=time_limit_str,
            time_limit_seconds=time_limit_seconds or 0,
            available=available,
            features=features_list,
            allow_qos=allow_qos,
            deny_qos=deny_qos or [],
            allow_accounts=allow_accounts,
            deny_accounts=deny_accounts or [],
        )

    # Fallback: derive partitions from nodes when scontrol partition data is empty
    # This handles older SLURM versions or configurations where partition JSON is unavailable
    if not partitions:
        # Aggregate node data per partition
        partition_nodes: dict[str, list[dict]] = {}
        for node in sinfo_data.get("nodes", []):
            node_partitions = node.get("partitions", [])
            for part_name in node_partitions:
                if part_name not in partition_nodes:
                    partition_nodes[part_name] = []
                partition_nodes[part_name].append(node)

        for part_name, nodes in partition_nodes.items():
            # Aggregate resources: use max values from nodes in this partition
            max_cpus = 0
            max_memory_mb = 0
            max_gpus = 0
            gpu_type = None
            all_features: set[str] = set()
            node_names: list[str] = []
            has_available_node = False

            for node in nodes:
                node_names.append(node.get("name", ""))
                node_cpus = node.get("cpus", 0)
                node_mem = node.get("real_memory", 0)  # Memory in MB
                if node_cpus > max_cpus:
                    max_cpus = node_cpus
                if node_mem > max_memory_mb:
                    max_memory_mb = node_mem

                # Check node state for availability
                node_state = node.get("state", "")
                if isinstance(node_state, str):
                    if node_state.lower() not in ("down", "drain", "drained", "fail"):
                        has_available_node = True
                else:
                    has_available_node = True  # Assume available if state is complex

                # Parse GRES for GPU info
                gres_str = node.get("gres", "")
                if gres_str:
                    node_gpu_count, node_gpu_type = parse_gres(gres_str)
                    if node_gpu_count > max_gpus:
                        max_gpus = node_gpu_count
                    if node_gpu_type and not gpu_type:
                        gpu_type = node_gpu_type

                # Collect features
                node_features = node.get("features", [])
                if isinstance(node_features, dict):
                    features_str = node_features.get("total", "")
                    for f in features_str.split(","):
                        f = f.strip()
                        if f:
                            all_features.add(f)
                elif isinstance(node_features, str):
                    for f in node_features.split(","):
                        f = f.strip()
                        if f:
                            all_features.add(f)
                elif isinstance(node_features, list):
                    all_features.update(node_features)

            partitions[part_name] = ClusterPartition(
                name=part_name,
                nodes=",".join(node_names),
                cpus_per_node=max_cpus,
                memory_mb=max_memory_mb,
                gpus_per_node=max_gpus,
                gpu_type=gpu_type,
                time_limit="",  # Not available from nodes
                time_limit_seconds=None,
                available=has_available_node,
                features=list(all_features),
                allow_qos=None,  # Not available from nodes
                deny_qos=[],
                allow_accounts=None,
                deny_accounts=[],
            )

    # Get QoS info (JSON)
    qos_data = cache.run_json_command(CACHE_KEY_QOS, SLURM_COMMANDS[CACHE_KEY_QOS][0])
    qos: dict[str, ClusterQoS] = {}

    for q in qos_data.get("qos", []):
        name = q.get("name", "")
        if not name:
            continue

        # Max wall time - path is limits.max.wall_clock.per.job
        # number is in MINUTES
        max_wall_job = (
            q.get("limits", {})
            .get("max", {})
            .get("wall_clock", {})
            .get("per", {})
            .get("job", {})
        )
        max_wall_seconds = None
        if isinstance(max_wall_job, dict):
            if max_wall_job.get("set", False) and not max_wall_job.get(
                "infinite", False
            ):
                max_wall_minutes = max_wall_job.get("number", 0)
                max_wall_seconds = max_wall_minutes * 60  # Convert to seconds

        max_wall_str = None
        if max_wall_seconds:
            days = max_wall_seconds // 86400
            hours = (max_wall_seconds % 86400) // 3600
            mins = (max_wall_seconds % 3600) // 60
            if days > 0:
                max_wall_str = f"{days}-{hours:02d}:{mins:02d}:00"
            else:
                max_wall_str = f"{hours:02d}:{mins:02d}:00"

        # Priority
        priority_data = q.get("priority", {})
        priority = 0
        if isinstance(priority_data, dict):
            if priority_data.get("set", False):
                priority = priority_data.get("number", 0)
        elif isinstance(priority_data, int):
            priority = priority_data

        # Get TRES limits
        # Check per.user first, then per.account, then total
        tres_per_user = (
            q.get("limits", {})
            .get("max", {})
            .get("tres", {})
            .get("per", {})
            .get("user", [])
        )
        tres_per_account = (
            q.get("limits", {})
            .get("max", {})
            .get("tres", {})
            .get("per", {})
            .get("account", [])
        )
        tres_total = q.get("limits", {}).get("max", {}).get("tres", {}).get("total", [])

        # Use whichever has data
        tres_list = tres_per_user or tres_per_account or tres_total

        # Extract GRES limits from TRES (type=gres, name=gpu:v100)
        gres_limits = []
        tres_parts = []
        for t in tres_list:
            tres_type = t.get("type", "")
            tres_name = t.get("name", "")
            tres_count = t.get("count", 0)
            if tres_name:
                tres_parts.append(f"{tres_type}/{tres_name}={tres_count}")
                # Collect GRES limits (e.g., "gpu:v100=32")
                if tres_type == "gres":
                    gres_limits.append(f"{tres_name}={tres_count}")
            else:
                tres_parts.append(f"{tres_type}={tres_count}")

        max_tres_str = ",".join(tres_parts) if tres_parts else None

        qos[name] = ClusterQoS(
            name=name,
            max_wall=max_wall_str,
            max_wall_seconds=max_wall_seconds,
            priority=priority,
            gres_limits=gres_limits,
            max_tres_per_job=max_tres_str,
            max_tres_per_user=None,  # Already merged into max_tres_str
        )

    # Get user associations (JSON)
    assoc_data = cache.run_json_command(
        CACHE_KEY_ASSOCIATIONS, SLURM_COMMANDS[CACHE_KEY_ASSOCIATIONS][0]
    )
    accounts: list[ClusterAccount] = []
    default_account = None

    for assoc in assoc_data.get("associations", []):
        account = assoc.get("account", "")
        partition = assoc.get("partition", "") or None
        qos_list = assoc.get("qos", [])
        if isinstance(qos_list, str):
            qos_list = [q.strip() for q in qos_list.split(",") if q.strip()]

        if assoc.get("is_default", False):
            default_account = account

        accounts.append(
            ClusterAccount(account=account, partition=partition, qos_list=qos_list)
        )

    # Get all features from partition_features
    features = set()
    for pf in partition_features.values():
        features.update(pf)

    # Get SLURM configuration (priority weights, etc.)
    slurm_config = cache.run_config_command(
        CACHE_KEY_CONFIG, SLURM_COMMANDS[CACHE_KEY_CONFIG][0]
    )

    return ClusterData(
        cluster_name=cluster_name,
        partitions=partitions,
        qos=qos,
        accounts=accounts,
        default_account=default_account,
        features=features,
        slurm_config=slurm_config,
    )


@click.group("slurm")
def cli():
    """SLURM launcher commands."""
    pass


def _run_slurm_command(command: str, output_type: str) -> dict:
    """Run a SLURM command and return parsed output."""
    import subprocess

    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
    )

    if output_type == "json":
        if result.stdout.strip():
            return json.loads(result.stdout)
        return {}
    elif output_type == "config":
        # Parse key = value lines
        data = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    key_name = parts[0].strip()
                    value = parts[1].strip()
                    if value.isdigit():
                        data[key_name] = int(value)
                    elif value.replace(".", "", 1).isdigit():
                        data[key_name] = float(value)
                    else:
                        data[key_name] = value
        return data
    return {}


@cli.command("output-cache")
def output_cache():
    """Output SLURM cluster cache as JSON to stdout.

    Runs SLURM commands and outputs parsed data as JSON for offline testing.
    Redirect to a file to create a cache that can be used with --use-cache.

    Examples:
        experimaestro launchers slurm output-cache > slurm_cache.json

        # Then use the cache offline:
        experimaestro launchers slurm configure --cache-file slurm_cache.json --use-cache
    """
    import sys

    print(f"Running {len(SLURM_COMMANDS)} SLURM commands...", file=sys.stderr)  # noqa: T201

    cache = {}
    for i, (key, (command, output_type)) in enumerate(SLURM_COMMANDS.items(), 1):
        cmd_display = command.split()[0:3]
        print(  # noqa: T201
            f"  [{i}/{len(SLURM_COMMANDS)}] {key}: {' '.join(cmd_display)}...",
            file=sys.stderr,
        )

        try:
            data = _run_slurm_command(command, output_type)
            cache[key] = data

            # Show some stats
            if key == CACHE_KEY_SINFO:
                nodes = len(data.get("nodes", []))
                sinfo = len(data.get("sinfo", []))
                print(  # noqa: T201
                    f"         -> OK ({nodes} nodes, {sinfo} sinfo entries)",
                    file=sys.stderr,
                )
            elif key == CACHE_KEY_PARTITIONS:
                parts = len(data.get("partitions", []))
                print(f"         -> OK ({parts} partitions)", file=sys.stderr)  # noqa: T201
            elif key == CACHE_KEY_QOS:
                qos_count = len(data.get("qos", []))
                print(f"         -> OK ({qos_count} QoS)", file=sys.stderr)  # noqa: T201
            elif key == CACHE_KEY_ASSOCIATIONS:
                assocs = len(data.get("associations", []))
                print(f"         -> OK ({assocs} associations)", file=sys.stderr)  # noqa: T201
            elif key == CACHE_KEY_CONFIG:
                config_count = len(data)
                print(  # noqa: T201
                    f"         -> OK ({config_count} config entries)", file=sys.stderr
                )
            else:
                print("         -> OK", file=sys.stderr)  # noqa: T201

        except json.JSONDecodeError as e:
            print(f"         -> WARNING: invalid JSON: {e}", file=sys.stderr)  # noqa: T201
            cache[key] = {}
        except Exception as e:
            print(f"         -> ERROR: {e}", file=sys.stderr)  # noqa: T201
            cache[key] = {}

    # Output JSON to stdout
    print(json.dumps(cache, indent=2))  # noqa: T201
    print("\nDone. Redirect output to a file.", file=sys.stderr)  # noqa: T201


@cli.command("configure")
@click.option(
    "--cache-file",
    type=click.Path(path_type=Path),
    default=None,
    help="JSON file for cached SLURM command outputs",
)
@click.option(
    "--use-cache",
    is_flag=True,
    help="Use cached SLURM outputs instead of running commands",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to configuration YAML (default: ~/.config/experimaestro/slurm.yaml)",
)
def configure(cache_file: Path | None, use_cache: bool, config_path: Path | None):
    """Interactive TUI for SLURM launcher configuration.

    Opens a terminal UI to configure partitions, GPU types, QoS mappings,
    and other settings. Configuration is saved to slurm.yaml.

    Existing configuration in slurm.yaml is preserved and merged with
    cluster information - new partitions are added but existing settings
    are not overwritten without confirmation.
    """
    from .tui import run_tui

    if config_path:
        config_path = config_path.expanduser()
    else:
        config_path = Path("~/.config/experimaestro/slurm.yaml").expanduser()

    if cache_file:
        cache_file = cache_file.expanduser()

    run_tui(
        cache_file=cache_file,
        use_cache=use_cache,
        config_path=config_path,
    )
