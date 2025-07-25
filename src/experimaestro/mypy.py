from mypy.plugin import Plugin, AttributeContext
from mypy.nodes import Var, TypeInfo, ARG_NAMED
from mypy.types import (
    CallableType,
    Instance,
    TypeType,
    AnyType,
    TypeOfAny,
    TypeAliasType,
    get_proper_type,
)

PARAM_FULLNAME = "experimaestro.core.arguments.Param"
CONFIG_FULLNAME = "experimaestro.core.objects.Config"


def plugin(version: str):
    return ExperimaestroPlugin


class ExperimaestroPlugin(Plugin):
    def get_class_attribute_hook(self, fullname: str):
        if fullname.endswith(".C") or fullname.endswith(".XPMConfig"):
            return handle_C_attribute
        return None


def extract_param_inner(t):
    t = get_proper_type(t)

    if isinstance(t, TypeAliasType) and t.alias.fullname == PARAM_FULLNAME:
        return t.args[0]
    return None


def handle_C_attribute(ctx: AttributeContext):
    """Intercept A.C and return a synthetic TypeType with Param-based __init__."""
    typ = get_proper_type(ctx.type)
    if not isinstance(typ, CallableType):
        return ctx.default_attr_type

    cls_info = typ.ret_type.type
    print(type(cls_info))

    # Collect Param[...] attributes from A and its bases
    arg_names, arg_types, arg_kinds = [], [], []
    for base in cls_info.mro:
        for name, sym in base.names.items():
            print("--->", sym)
            node = sym.node
            if isinstance(node, Var) and node.type:
                inner_t = extract_param_inner(node.type)
                if inner_t:
                    arg_names.append(name)
                    arg_types.append(inner_t)
                    arg_kinds.append(ARG_NAMED)

    if not arg_names:
        return ctx.default_attr_type

    api = ctx.api
    anyt = AnyType(TypeOfAny.special_form)
    # return_type = Instance(cls_info, [anyt] * len(cls_info.defn.type_vars))

    # __init__ signature
    init_sig = CallableType(
        arg_types=arg_types,
        arg_kinds=arg_kinds,
        arg_names=arg_names,
        ret_type=None,
        fallback=api.named_type("builtins.function"),
    )

    # Create a synthetic TypeInfo for A.C
    class_name = f"{cls_info.name}.C"
    symbol_table = api.basic_symbol_table()
    fake_info = TypeInfo(
        symbol_table, api.lookup_fully_qualified("builtins.object").node
    )
    fake_info.set_name(class_name)
    fake_info.bases = [api.named_type("builtins.object")]
    fake_info.mro = [fake_info] + fake_info.bases[0].type.mro
    api.add_method(fake_info, "__init__", init_sig)

    fake_instance = Instance(fake_info, [])
    return TypeType(fake_instance, fallback=api.named_type("builtins.type"))
