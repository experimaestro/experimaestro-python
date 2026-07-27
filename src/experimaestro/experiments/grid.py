from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    Type,
    get_args,
    get_origin,
    get_type_hints,
    Annotated,
)
from pydantic import BeforeValidator
from itertools import product
import logging
import copy
import attr

logger = logging.getLogger(__name__)

T = TypeVar("T")

@attr.define()
class GenericParams:
    """A class to represent a parameter that can be a single value, a list of values, or a range of values."""
    value: Any = None
    values_list: Optional[List[Any]] = None
    values_range: Optional[Tuple[int, ...]] = None
    values_mult: Optional[Tuple[Any, ...]] = None

    @property
    def is_grid(self) -> bool:
        """Returns True if this parameter represents a search space."""
        return len(self.as_list()) > 1

    def as_list(self) -> List[Any]:
        """Returns the search space as a list of values."""
        if self.values_list:
            return list(self.values_list)
        if self.values_range:
            return list(range(*self.values_range))
        if self.values_mult:
            if len(self.values_mult) != 3:
                raise ValueError("range_mult must have exactly 3 elements: [start, n_iter, multiplier]")
            start, n_iter, multiplier = self.values_mult

            # Convert strings (e.g. '1e-4') to numbers to prevent python string repetition
            def to_num(val):
                if isinstance(val, str):
                    try:
                        return float(val) if ("." in val or "e" in val.lower()) else int(val)
                    except ValueError:
                        return val
                return val

            start = to_num(start)
            multiplier = to_num(multiplier)
            return [start * (multiplier ** i) for i in range(int(n_iter))]
        if self.value is not None:
            return [self.value]
        return []

    @classmethod
    def from_any(cls, obj: Any, target_type: Type = Any) -> "GenericParams":
        """Coerces a value into a GenericParams object."""
        def converter(value: Any) -> Any:
            """Attempts to convert a value to the target_type."""
            if target_type is Any:
                return value

            types_to_try = []
            if get_origin(target_type) is Union:
                types_to_try.extend(get_args(target_type))
            else:
                types_to_try.append(target_type)

            for t in types_to_try:
                if t is type(None):
                    continue
                try:
                    return t(value)
                except (ValueError, TypeError):
                    continue

            return value

        # 1. Already the right type
        if isinstance(obj, cls):
            return obj

        # 2. It's a raw value
        if isinstance(obj, (str, int, float, bool)):
            return cls(value=converter(obj))

        # 3. It's a list
        if isinstance(obj, (list, tuple)):
            return cls(values_list=[converter(v) for v in obj])

        # 4. It's a dict
        if isinstance(obj, dict):
            d = dict(obj)

            # Check for unrecognized keys
            RECOGNIZED_KEYS = {"value", "values_list", "values_range", "range", "values_mult", "range_mult"}
            unrecognized = set(d.keys()) - RECOGNIZED_KEYS
            if unrecognized:
                options_str = ", ".join(sorted(RECOGNIZED_KEYS))
                raise ValueError(
                    f"Unrecognized keys in GridSearch parameter: {', '.join(sorted(unrecognized))}. "
                    f"Possible options are: {options_str}"
                )

            value = d.get("value")
            values_list = d.get("values_list")

            if value is not None:
                value = converter(value)

            if values_list is not None:
                values_list = [converter(v) for v in values_list]

            # Support both 'values_range' and 'range'
            range_val = d.get("values_range") or d.get("range")

            # Support both 'values_mult' and 'range_mult'
            mult_val = d.get("values_mult") or d.get("range_mult")

            return cls(
                value=value,
                values_list=values_list,
                values_range=(
                    tuple(range_val) if range_val is not None else None
                ),
                values_mult=(
                    tuple(mult_val) if mult_val is not None else None
                ),
            )

        return cls(value=converter(obj))

def _coerce_to_generic_params(v: Any) -> Any:
    if isinstance(v, GenericParams):
        return v
    return GenericParams.from_any(v)

GridSearch = Annotated[
    Union[T, GenericParams],
    BeforeValidator(_coerce_to_generic_params),
]
"""
Type alias for configuration fields supporting inline grid search.
"""

def set_nested_attr(obj: Any, path: str, value: Any):
    """Sets a nested attribute on an object."""
    keys = path.split(".")
    current = obj
    for key in keys[:-1]:
        current = getattr(current, key)
    setattr(current, keys[-1], value)

def get_nested_attr_type(obj: Any, path: str) -> Any:
    """
    Traverses a nested object to find the type hint of the final attribute.
    Raises ValueError if the path is invalid.
    """
    keys = path.split(".")
    current_obj = obj
    for i, key in enumerate(keys[:-1]):
        if not hasattr(current_obj, key):
            raise ValueError(
                f"Invalid grid search path '{path}': "
                f"'{key}' not found in {type(current_obj).__name__} "
                f"(at level {'.'.join(keys[:i]) if i > 0 else 'root'})"
            )
        current_obj = getattr(current_obj, key)

    last_key = keys[-1]
    if not hasattr(current_obj, last_key):
        raise ValueError(
            f"Invalid grid search path '{path}': "
            f"'{last_key}' not found in {type(current_obj).__name__} "
            f"(at level {'.'.join(keys[:-1]) if len(keys) > 1 else 'root'})"
        )

    try:
        type_hints = get_type_hints(type(current_obj))
        return type_hints.get(last_key, Any)
    except Exception:
        return Any

def discover_grid_params(obj: Any, prefix: str = "") -> Dict[str, GenericParams]:
    """Recursively find all GenericParams instances in an attrs object"""
    found = {}
    if attr.has(type(obj)):
        for f in attr.fields(type(obj)):
            val = getattr(obj, f.name)
            path = f"{prefix}.{f.name}" if prefix else f.name
            if isinstance(val, GenericParams):
                found[path] = val
            elif val is not None and not isinstance(val, (str, int, float, bool)):
                # Avoid recursing into primitives or enums (which are strings/ints)
                from enum import Enum
                if not isinstance(val, Enum):
                    found.update(discover_grid_params(val, path))
    return found

def finalize_config(obj: Any):
    """Recursively convert all non-grid GenericParams to scalars"""
    if attr.has(type(obj)):
        for f in attr.fields(type(obj)):
            val = getattr(obj, f.name)
            if isinstance(val, GenericParams) and not val.is_grid:
                setattr(obj, f.name, val.value)
            elif val is not None and not isinstance(val, (str, int, float, bool)):
                from enum import Enum
                if not isinstance(val, Enum):
                    finalize_config(val)

def generate_grid(cfg: Any) -> Tuple[List[Any], List[dict]]:
    """
    Generates a list of configuration permutations for a grid search, based
    on a `grid_search` dictionary in the main configuration object or inline definitions.

    returns:
     configs: List of all configurations
     tags: a list of dicts with the same length as configs, where each dict
           contains the parameter values that were set for that config.
    """
    # 1. Discover inline grid parameters
    grid_params = discover_grid_params(cfg)

    # 2. Merge with explicit grid_search block
    if hasattr(cfg, "grid_search") and cfg.grid_search:
        for path, gp in cfg.grid_search.items():
            if isinstance(gp, GenericParams):
                grid_params[path] = gp
            else:
                grid_params[path] = GenericParams.from_any(gp)

    # If no grid parameters found, just return the original config.
    if not grid_params:
        logger.info("no params to grid search, returning raw config")
        new_cfg = copy.deepcopy(cfg)
        finalize_config(new_cfg)
        return [new_cfg], [{}]

    param_paths = list(grid_params.keys())

    value_options = []
    for path in param_paths:
        # get target type for this parameter from the config class using the path
        target_type = get_nested_attr_type(cfg, path)

        def converter(value: Any) -> Any:
            if target_type is Any:
                return value
            types_to_try = [
                t
                for t in (
                    get_args(target_type)
                    if get_origin(target_type) is Union
                    else [target_type]
                )
                if t is not type(None)
            ]
            for t in types_to_try:
                try:
                    return t(value)
                except (ValueError, TypeError):
                    continue
            return value

        gp_from_framework = (
            grid_params[path]
            if isinstance(grid_params[path], GenericParams)
            else GenericParams.from_any(grid_params[path])
        )
        raw_values = gp_from_framework.as_list()
        converted_values = [converter(v) for v in raw_values]
        value_options.append(converted_values)

    # Generate Cartesian product of all parameter values
    grid_combinations = product(*value_options)

    output_configs = []
    tags = []
    base_cfg = copy.deepcopy(cfg)
    # Clear grid_search to make generated configs clean
    if hasattr(base_cfg, "grid_search"):
        base_cfg.grid_search = {}

    logger.info("Building grid search configs")
    for combination in grid_combinations:
        cfg_tags = {}
        new_cfg = copy.deepcopy(base_cfg)
        for i, (path, value) in enumerate(zip(param_paths, combination)):
            if len(value_options[i]) > 1:
                cfg_tags[path] = value
            set_nested_attr(new_cfg, path, value)

        # Convert any remaining single-value GenericParams to scalars
        finalize_config(new_cfg)

        output_configs.append(new_cfg)
        tags.append(cfg_tags)

    return output_configs, tags
