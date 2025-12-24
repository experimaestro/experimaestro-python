"""Mypy plugin for experimaestro.

This plugin provides type hints support for experimaestro's Config system,
particularly for the Config.C pattern and proper parameter type inference.

The plugin handles:
- Config.C, Config.XPMConfig, Config.XPMValue class properties
- Adding __init__ with proper Param field signatures
- Adding ConfigMixin to the class hierarchy for method access
- Handling task_outputs return type for submit()

Usage in mypy.ini or pyproject.toml:
    [mypy]
    plugins = experimaestro.mypy

Or in pyproject.toml:
    [tool.mypy]
    plugins = ["experimaestro.mypy"]
"""

from __future__ import annotations

from typing import Callable, List, Optional

from mypy.nodes import (
    TypeInfo,
    Var,
    Argument,
    ARG_NAMED_OPT,
    ARG_NAMED,
)
from mypy.plugin import Plugin, ClassDefContext
from mypy.plugins.common import add_attribute_to_class, add_method_to_class
from mypy.types import (
    Instance,
    TypeType,
    NoneType,
)
from mypy.mro import calculate_mro, MroError

# Full names of Config and its subclasses that need C/XPMConfig attributes
CONFIG_FULLNAMES = {
    "experimaestro.core.objects.config.Config",
    "experimaestro.core.objects.config.LightweightTask",
    "experimaestro.core.objects.config.Task",
    "experimaestro.core.objects.config.ResumableTask",
    "experimaestro.Config",
    "experimaestro.Task",
    "experimaestro.LightweightTask",
    "experimaestro.ResumableTask",
    "experimaestro.core.objects.Config",
    "experimaestro.core.objects.Task",
    "experimaestro.core.objects.LightweightTask",
    "experimaestro.core.objects.ResumableTask",
}

# ConfigMixin full name for method inheritance
CONFIGMIXIN_FULLNAME = "experimaestro.core.objects.config.ConfigMixin"

# Full names for Param annotations (required by default)
PARAM_FULLNAMES = {
    "experimaestro.core.arguments.Param",
    "experimaestro.Param",
}

# Full names for Meta/Option annotations (always optional, ignored in identifier)
META_FULLNAMES = {
    "experimaestro.core.arguments.Meta",
    "experimaestro.Meta",
    "experimaestro.core.arguments.Option",
    "experimaestro.Option",
}

# Full names for Constant annotations (excluded from __init__)
CONSTANT_FULLNAMES = {
    "experimaestro.core.arguments.Constant",
    "experimaestro.Constant",
}


def is_config_subclass(info: TypeInfo) -> bool:
    """Check if a TypeInfo represents a Config subclass.

    Args:
        info: The TypeInfo to check

    Returns:
        True if the type is Config or a subclass of Config
    """
    if info.fullname in CONFIG_FULLNAMES:
        return True
    for base in info.mro:
        if base.fullname in CONFIG_FULLNAMES:
            return True
    return False


# Fields to skip when building __init__ signature
SKIP_FIELDS = {
    "C",
    "XPMConfig",
    "XPMValue",
    "__xpm__",
    "__xpmtype__",
    "__xpmid__",
    "_deprecated_from",
}


def _is_config_class(base: TypeInfo) -> bool:
    """Check if a TypeInfo is a Config subclass.

    Returns True for user-defined Config subclasses.
    """
    for mro_base in base.mro:
        if mro_base.fullname in CONFIG_FULLNAMES:
            return True
    return False


def _get_annotation_type_str(name: str, base: TypeInfo) -> Optional[str]:
    """Get the type annotation string for a field.

    Tries multiple sources to find the original annotation:
    1. The AST unanalyzed_type (preserves original)
    2. The variable's type string
    """
    # Check the AST first to get unanalyzed types
    if base.defn is not None:
        for stmt in base.defn.defs.body:
            from mypy.nodes import AssignmentStmt

            if isinstance(stmt, AssignmentStmt):
                for lvalue in stmt.lvalues:
                    from mypy.nodes import NameExpr

                    if isinstance(lvalue, NameExpr) and lvalue.name == name:
                        # Try unanalyzed_type first (preserves the original annotation)
                        if stmt.unanalyzed_type is not None:
                            return str(stmt.unanalyzed_type)
                        # Fall back to analyzed type
                        if stmt.type is not None:
                            return str(stmt.type)

    # Fall back to checking the symbol's type
    if name in base.names:
        sym = base.names[name]
        if sym.node is not None and isinstance(sym.node, Var):
            var = sym.node
            if var.type is not None:
                return str(var.type)

    return None


def _is_constant_field(name: str, base: TypeInfo) -> bool:
    """Check if a field is declared as Constant[T].

    Constant fields should be excluded from __init__.
    """
    type_str = _get_annotation_type_str(name, base)
    if type_str is None:
        return False

    # Normalize type string - remove optional markers (?)
    # mypy represents types like "Constant?[str?]"
    type_lower = type_str.lower().replace("?", "")

    # Check for Constant annotation in the type string
    if "constant[" in type_lower:
        return True
    for fullname in CONSTANT_FULLNAMES:
        if fullname.lower() in type_lower:
            return True
    return False


def _is_meta_field(name: str, base: TypeInfo) -> bool:
    """Check if a field is declared as Meta[T] or Option[T].

    Meta fields should always be optional in __init__.
    """
    type_str = _get_annotation_type_str(name, base)
    if type_str is None:
        return False

    # Normalize type string - remove optional markers (?)
    # mypy represents types like "Meta?[Path?]"
    type_lower = type_str.lower().replace("?", "")

    # Check for Meta/Option annotation in the type string
    if "meta[" in type_lower or "option[" in type_lower:
        return True
    for fullname in META_FULLNAMES:
        if fullname.lower() in type_lower:
            return True
    return False


def _get_param_fields(info: TypeInfo) -> List[tuple]:
    """Extract Param and Meta fields from a class and its bases.

    Returns list of (name, type, has_default) tuples.

    Only includes fields from Config subclasses to avoid picking up
    attributes from other base classes like nn.Module.
    Excludes Constant fields which should not be in __init__.
    """
    fields = []
    seen = set()

    # Walk MRO to get inherited fields (in reverse to get proper order)
    for base in reversed(info.mro):
        if base.fullname == "builtins.object":
            continue
        if base.fullname in CONFIG_FULLNAMES:
            # Skip Config/Task base classes - we only want user-defined fields
            continue
        if base.fullname == CONFIGMIXIN_FULLNAME:
            # Skip ConfigMixin - it has methods, not params
            continue

        # Only include fields from Config subclasses
        # This skips bases like nn.Module that don't inherit from Config
        if not _is_config_class(base):
            continue

        for name, sym in base.names.items():
            if name in seen or name in SKIP_FIELDS:
                continue
            if sym.node is None or not isinstance(sym.node, Var):
                continue

            var = sym.node
            if var.type is None:
                continue

            # Skip private/dunder fields
            if name.startswith("_"):
                continue

            # Skip Constant fields - they should not be in __init__
            if _is_constant_field(name, base):
                continue

            # Meta fields are always optional
            # Param fields are optional only if they have a default
            is_meta = _is_meta_field(name, base)
            has_default = var.has_explicit_value or is_meta

            seen.add(name)
            fields.append((name, var.type, has_default))

    return fields


def _add_init_method(ctx: ClassDefContext) -> None:
    """Add an __init__ method with proper Param field signatures."""
    info = ctx.cls.info

    # Get all Param fields from this class and bases
    fields = _get_param_fields(info)

    # Build __init__ arguments
    args = []
    for name, field_type, has_default in fields:
        # All experimaestro params are keyword-only
        # Fields with defaults are optional
        kind = ARG_NAMED_OPT if has_default else ARG_NAMED

        # Create argument
        arg = Argument(
            variable=Var(name, field_type),
            type_annotation=field_type,
            initializer=None,
            kind=kind,
        )
        args.append(arg)

    # Add the __init__ method if we have any args
    if args:
        add_method_to_class(
            ctx.api,
            ctx.cls,
            "__init__",
            args,
            NoneType(),
        )


def _get_task_outputs_return_type(info: TypeInfo) -> Optional[Instance]:
    """Check if the class has a task_outputs method and return its return type.

    If the class defines task_outputs, submit() should return that type instead
    of Self.
    """
    # Look for task_outputs method in the class
    if "task_outputs" in info.names:
        sym = info.names["task_outputs"]
        if sym.node is not None:
            # Try to get the return type from the method signature
            from mypy.nodes import FuncDef

            if isinstance(sym.node, FuncDef):
                ret_type = sym.node.type
                if ret_type is not None:
                    from mypy.types import CallableType

                    if isinstance(ret_type, CallableType):
                        return ret_type.ret_type
    return None


def _add_configmixin_to_bases(ctx: ClassDefContext) -> None:
    """Add ConfigMixin to the class bases if not already present.

    This allows mypy to see all ConfigMixin methods on Config subclasses.
    """
    info = ctx.cls.info

    # Check if ConfigMixin is already in the MRO
    for base in info.mro:
        if base.fullname == CONFIGMIXIN_FULLNAME:
            return  # Already has ConfigMixin

    # Try to look up ConfigMixin
    try:
        configmixin_sym = ctx.api.lookup_fully_qualified_or_none(CONFIGMIXIN_FULLNAME)
        if configmixin_sym is None or not isinstance(configmixin_sym.node, TypeInfo):
            return

        configmixin_info = configmixin_sym.node
        configmixin_instance = Instance(configmixin_info, [])

        # Add ConfigMixin to bases if not already present
        configmixin_in_bases = any(
            isinstance(b, Instance) and b.type.fullname == CONFIGMIXIN_FULLNAME
            for b in info.bases
        )
        if not configmixin_in_bases:
            info.bases.append(configmixin_instance)

            # Recalculate MRO
            try:
                calculate_mro(info)
            except MroError:
                # If MRO calculation fails, remove the base we added
                info.bases.pop()
    except Exception:
        # If lookup fails, continue without adding ConfigMixin
        pass


def _add_submit_method(ctx: ClassDefContext) -> None:
    """Add submit() method that returns Self (or task_outputs return type).

    The actual submit() signature from ConfigMixin:
        def submit(self, *, workspace=None, launcher=None, run_mode=None,
                   init_tasks=[], max_retries=None)
    """
    info = ctx.cls.info

    # Check if the class has task_outputs
    task_outputs_type = _get_task_outputs_return_type(info)

    # submit() returns task_outputs return type if defined, otherwise Self
    if task_outputs_type is not None:
        return_type = task_outputs_type
    else:
        return_type = Instance(info, [])

    # Build submit() arguments - all optional kwargs
    from mypy.types import AnyType, TypeOfAny

    any_type = AnyType(TypeOfAny.explicit)
    submit_args = []
    for arg_name in ("workspace", "launcher", "run_mode", "init_tasks", "max_retries"):
        arg = Argument(
            variable=Var(arg_name, any_type),
            type_annotation=any_type,
            initializer=None,
            kind=ARG_NAMED_OPT,
        )
        submit_args.append(arg)

    add_method_to_class(
        ctx.api,
        ctx.cls,
        "submit",
        submit_args,
        return_type,
    )


def _process_config_class(ctx: ClassDefContext) -> None:
    """Process a Config subclass to add type hints.

    This adds:
    - ConfigMixin to the class hierarchy for method access
    - C, XPMConfig, XPMValue as class attributes returning Type[Self]
    - An __init__ method with proper Param field signatures
    - A submit() method that returns Self (or task_outputs return type)
    """
    info = ctx.cls.info

    # Add ConfigMixin to bases for method access (tag, instance, etc.)
    _add_configmixin_to_bases(ctx)

    # Create Type[Self] for this class
    class_type = Instance(info, [])
    type_type = TypeType(class_type)

    # Add C, XPMConfig, XPMValue as class attributes returning the class type
    for attr_name in ("C", "XPMConfig", "XPMValue"):
        if attr_name not in info.names:
            add_attribute_to_class(
                ctx.api,
                ctx.cls,
                attr_name,
                type_type,
            )

    # Add __init__ with proper field signatures
    _add_init_method(ctx)

    # Add submit() method that returns Self (or task_outputs type)
    _add_submit_method(ctx)


class ExperimaestroPlugin(Plugin):
    """Mypy plugin for experimaestro type hints.

    This plugin handles:
    - Converting @classproperty decorated methods to proper class attributes
    - Type inference for Config.C and Config.XPMConfig patterns
    - Adding __init__ methods with proper Param field signatures
    """

    def get_base_class_hook(
        self, fullname: str
    ) -> Callable[[ClassDefContext], None] | None:
        """Hook called when a class inherits from Config.

        This allows us to process classproperty attributes and add __init__.
        """
        if fullname in CONFIG_FULLNAMES:
            return _process_config_class
        return None


def plugin(_version: str):
    """Entry point for mypy plugin.

    Args:
        _version: The mypy version string (unused but required by mypy API)

    Returns:
        The ExperimaestroPlugin class
    """
    return ExperimaestroPlugin
