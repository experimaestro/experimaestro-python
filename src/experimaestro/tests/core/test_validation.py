"""Tests for type validation"""

import pytest
from pathlib import Path
from pydantic_core import ValidationError
from experimaestro import Task, field, Identifier, Constant, Param, Config, Meta
from enum import Enum
from experimaestro.core.types import Type, TupleType, UnionType
from experimaestro.generators import PathGenerator
from experimaestro.scheduler import Job, JobContext
from experimaestro.scheduler.workspace import RunMode
from ..utils import TemporaryExperiment
from experimaestro.xpmutils import EmptyContext

# Mark all tests in this module as config tests
pytestmark = pytest.mark.config

valns = Identifier("validation")


def expect_validate(value):
    value.__xpm__.validate()


def expect_notvalidate(value):
    with pytest.raises((ValueError, KeyError)):
        value.__xpm__.validate()


class A(Config):
    value: Param[int]


class B(Config):
    a: Param[A]


class C(Config):
    path: Meta[Path] = field(default_factory=PathGenerator("outdir"))
    pass


def test_validation_simple():
    expect_validate(A.C(value=1))


def test_validation_missing():
    expect_notvalidate(A.C())


def test_validation_simple_nested():
    b = B.C()
    b.a = A.C(value=1)
    expect_validate(b)


def test_validation_missing_nested():
    b = B.C()
    b.a = A.C()
    expect_notvalidate(b)


def test_validation_type():
    class A(Config):
        __xpmid__ = valns.type.a
        pass

    class B(Config):
        __xpmid__ = valns.type.b

    class C(Config):
        a: Param[A]
        __xpmid__ = valns.type.c

    with pytest.raises(ValueError):
        C.C(a=B.C())

    with pytest.raises(ValueError):
        c = C.C()
        c.a = B.C()


def test_validation_subtype():
    class A(Config):
        __xpmid__ = valns.subtype.a

    class A1(A):
        __xpmid__ = valns.subtype.a1

    class B(Config):
        __xpmid__ = valns.subtype.b
        a: Param[A]

    expect_validate(B.C(a=A1.C()))


def test_validation_path_generator():
    """Test of path generator"""

    class A(Config):
        __xpmid__ = valns.path.a
        value: Meta[Path] = field(default_factory=PathGenerator("file.txt"))

    a = A.C()
    a.__xpm__.validate()
    with TemporaryExperiment("constant") as xp:
        jobcontext = Job(a)
        a.__xpm__.seal(JobContext(jobcontext))
        assert isinstance(a.value, Path)
        assert a.value.name == "file.txt"
        assert a.value.parents[0].name == a.__xpm__.identifier.all.hex()
        assert a.value.parents[1].name == str(a.__xpmtype__.identifier)
        assert a.value.parents[2].name == "jobs"
        assert a.value.parents[3] == xp.workspace.path


def test_validation_constant():
    """Test of constant"""

    class A(Config):
        __xpmid__ = valns.constant.a
        value: Constant[int] = 1

    a = A.C()
    a.__xpm__.validate()
    with TemporaryExperiment("constant"):
        joba = Job(a)
        a.__xpm__.seal(JobContext(joba))
        assert a.value == 1


class Parent(Config):
    x: Param[int]


class Child(Parent):
    pass


def test_validation_child():
    expect_validate(Child.C(x=1))


# --- Path argument checks


class PathParent(Config):
    x: Meta[Path] = field(default_factory=PathGenerator("x"))


def test_validation_path_option():
    c = PathParent.C()
    expect_validate(c)


# --- Default value


def test_validation_seal():
    """Test value sealing"""

    class A(Config):
        a: Param[int]

    a = A.C(a=2)
    a.__xpm__.seal(EmptyContext())

    with pytest.raises(AttributeError):
        a.a = 1


def test_validation_validation_enum():
    """Path arguments should be ignored"""

    class EnumParam(Enum):
        FIRST = 0
        SECOND = 1

    class EnumConfig(Config):
        a: Param[EnumParam]

    expect_validate(EnumConfig.C(a=EnumParam.FIRST))

    # A non-enum value is rejected (ValueError-family via pydantic validation)
    with pytest.raises(ValueError):
        EnumConfig.C(a=1)


# --- Task as argument


class TaskParentConfig(Config):
    pass


class taskconfig(TaskParentConfig, Task):
    pass


class TaskConfigConsumer(Config):
    x: Param[TaskParentConfig]


def test_validation_taskargument():
    x = taskconfig.C()
    with TemporaryExperiment("fake"):
        x.submit(run_mode=RunMode.DRY_RUN)
        expect_validate(TaskConfigConsumer.C(x=x))


# --- Nested structures (list, dict, set)


class ListConfig(Config):
    items: Param[list[A]]


class DictConfig(Config):
    items: Param[dict[str, A]]


def test_validation_list_valid():
    """Validate recurses into list elements"""
    expect_validate(ListConfig.C(items=[A.C(value=1), A.C(value=2)]))


def test_validation_list_missing():
    """Validate catches missing required params inside list elements"""
    expect_notvalidate(ListConfig.C(items=[A.C(value=1), A.C()]))


def test_validation_dict_valid():
    """Validate recurses into dict values"""
    expect_validate(DictConfig.C(items={"x": A.C(value=1), "y": A.C(value=2)}))


def test_validation_dict_missing():
    """Validate catches missing required params inside dict values"""
    expect_notvalidate(DictConfig.C(items={"x": A.C(value=1), "y": A.C()}))


# --- Union types (pydantic-core backed) ------------------------------------
#
# UnionType used to swallow all member errors and, for a non-matching dict,
# silently return None. It now resolves natively (left-to-right, first match).


class UnionConfig(Config):
    x: Param[int | str]


def test_union_accepts_members():
    assert UnionConfig.C(x=1).x == 1
    assert UnionConfig.C(x="hello").x == "hello"


def test_union_left_to_right_first_match():
    """1.0 coerces to int 1 (first member wins)."""
    value = UnionConfig.C(x=1.0).x
    assert value == 1
    assert isinstance(value, int)


def test_union_rejects_non_member():
    with pytest.raises(ValueError):
        UnionConfig.C(x=[])


def test_union_non_matching_dict_raises_not_none():
    """Regression: a dict matching no member used to return None silently."""
    tp = Type.fromType(int | str)
    assert isinstance(tp, UnionType)
    with pytest.raises(ValueError):
        tp.validate({"a": 1})


# --- Tuple types (pydantic-core backed) ------------------------------------
#
# tuple[...] annotations used to fall through to GenericType with no element
# or arity checks; they are now validated structurally and recursively.


def test_fromtype_builds_tuple_type():
    assert isinstance(Type.fromType(tuple[int, str]), TupleType)
    variadic = Type.fromType(tuple[int, ...])
    assert isinstance(variadic, TupleType) and variadic.variadic


def test_tuple_fixed_valid():
    tp = Type.fromType(tuple[int, str])
    assert tp.validate((1, "a")) == (1, "a")


def test_tuple_fixed_wrong_element_type():
    tp = Type.fromType(tuple[int, str])
    with pytest.raises(ValueError):
        tp.validate((1, 2))  # second element must be a str, got int


def test_tuple_fixed_wrong_arity():
    tp = Type.fromType(tuple[int, str])
    with pytest.raises(ValueError):
        tp.validate((1,))


def test_tuple_variadic_valid():
    tp = Type.fromType(tuple[int, ...])
    assert tp.validate((1, 2, 3)) == (1, 2, 3)


def test_tuple_variadic_wrong_element():
    tp = Type.fromType(tuple[int, ...])
    with pytest.raises(ValueError):
        tp.validate((1, "x", 3))


def test_tuple_nested_tuples_validate_recursively():
    """A tuple nested inside a tuple is validated recursively (schemas compose)."""
    tp = Type.fromType(tuple[tuple[int, str], str])
    assert tp.validate(((1, "a"), "b")) == ((1, "a"), "b")
    with pytest.raises(ValueError):
        tp.validate(((1, 2), "b"))  # inner position 1 must be a str
    with pytest.raises(ValueError):
        tp.validate(((1,), "b"))  # inner arity is wrong


def test_tuple_with_builtin_list_member_deep_validated():
    """Builtin list[int] members are deep-validated too."""
    tp = Type.fromType(tuple[list[int], str])
    assert tp.validate(([1, 2], "a")) == ([1, 2], "a")
    with pytest.raises(ValueError):
        tp.validate((["x"], "a"))


def test_tuple_of_config_elements():
    class Inner(Config):
        value: Param[int]

    tp = Type.fromType(tuple[Inner, Inner])
    a, b = Inner.C(value=1), Inner.C(value=2)
    assert tp.validate((a, b)) == (a, b)
    with pytest.raises(ValueError):
        tp.validate((a, 3))


# --- Coercion corpus -------------------------------------------------------
#
# int/float/bool use native pydantic schemas (lax): numeric strings parse and
# the legacy bool(value) bug ("false" -> True) is fixed. Plain ints/floats/bools
# coerce identically to before, so identifiers do not drift (see
# test_identifier_stability.py for the golden-identifier guard).


class _RAISES:
    """Sentinel: validation must reject this value."""


# (annotation, input, expected-output-or-_RAISES)
_COERCION_CORPUS = [
    (int, 1, 1),
    (int, True, 1),
    (int, 1.0, 1),
    (int, 1.5, _RAISES),
    (int, "5", 5),
    (int, "x", _RAISES),
    (float, 1.0, 1.0),
    (float, 1, 1.0),
    (float, True, 1.0),
    (float, "1.5", 1.5),
    (float, "x", _RAISES),
    (bool, True, True),
    (bool, 1, True),
    (bool, "false", False),
    (bool, "true", True),
    (bool, [], _RAISES),
    (bool, 0, False),
    (str, "a", "a"),
    (str, 1, _RAISES),
    (str, b"x", _RAISES),
    (Path, "p", Path("p")),
    (Path, Path("p"), Path("p")),
    (
        Path,
        {"type": "path", "value": "q"},
        _RAISES,
    ),  # serialized form: handled by the deserializer, not validation
    (Path, 5, _RAISES),
    (list[int], [1, 2], [1, 2]),
    (list[int], [1.0], [1]),
    (list[int], ["x"], _RAISES),
    (list[int], (1, 2), _RAISES),
    (set[int], {1, 2}, {1, 2}),
    (set[int], [1], _RAISES),
    (dict[str, int], {"a": 1}, {"a": 1}),
    (dict[str, int], {"a": "x"}, _RAISES),
    (dict[str, int], [("a", 1)], _RAISES),
]


@pytest.mark.parametrize("annotation,value,expected", _COERCION_CORPUS)
def test_coercion(annotation, value, expected):
    tp = Type.fromType(annotation)
    if expected is _RAISES:
        with pytest.raises((ValueError, TypeError)):
            tp.validate(value)
    else:
        result = tp.validate(value)
        assert result == expected
        assert type(result) is type(expected)


# --- Deeply nested generics ------------------------------------------------
#
# Arbitrarily nested standard generics validate (and coerce) recursively, the
# main reason for moving validation onto pydantic-core. Each Type node composes
# its children's schemas, so nesting depth is unbounded.

# (annotation, valid-input, valid-output)
_NESTED_VALID = [
    (list[dict[str, int]], [{"a": 1}, {"b": 2}], [{"a": 1}, {"b": 2}]),
    (dict[str, list[int]], {"a": [1, 2]}, {"a": [1, 2]}),
    (list[list[int]], [[1], [2, 3]], [[1], [2, 3]]),
    (dict[str, tuple[int, str]], {"a": (1, "x")}, {"a": (1, "x")}),
    (list[int | None], [1, None, 2], [1, None, 2]),
    (dict[str, dict[str, list[int]]], {"a": {"b": [1]}}, {"a": {"b": [1]}}),
    # coercion propagates to the leaves of a nested structure
    (list[dict[str, int]], [{"a": 1.0}], [{"a": 1}]),
]

# (annotation, invalid-input)
_NESTED_INVALID = [
    (list[dict[str, int]], [{"a": "x"}]),  # leaf not an int
    (dict[str, list[int]], {"a": [1, "x"]}),  # leaf not an int
    (list[list[int]], [[1], ["x"]]),  # leaf not an int
    (dict[str, tuple[int, str]], {"a": (1, 2)}),  # tuple slot 1 not a str
    (dict[str, tuple[int, str]], {"a": (1,)}),  # tuple arity
    (list[dict[str, int]], [{"a": 1}, "not-a-dict"]),  # element not a dict
]


@pytest.mark.parametrize("annotation,value,expected", _NESTED_VALID)
def test_nested_generic_valid(annotation, value, expected):
    assert Type.fromType(annotation).validate(value) == expected


@pytest.mark.parametrize("annotation,value", _NESTED_INVALID)
def test_nested_generic_invalid(annotation, value):
    with pytest.raises((ValueError, TypeError)):
        Type.fromType(annotation).validate(value)


def test_nested_generic_error_reports_location_path():
    """The error pinpoints the failing location within the nested structure."""
    tp = Type.fromType(list[dict[str, int]])
    with pytest.raises(ValidationError) as excinfo:
        tp.validate([{"a": 1}, {"b": "x"}])
    locs = [err["loc"] for err in excinfo.value.errors()]
    # second list element (index 1), key "b"
    assert any(loc[:2] == (1, "b") for loc in locs)


# --- Deeply nested generics containing Config objects ----------------------


class _Leaf(Config):
    value: Param[int]


class NestedConfigConfig(Config):
    items: Param[dict[str, list[_Leaf]]]


def test_nested_generic_config_valid():
    expect_validate(
        NestedConfigConfig.C(items={"g": [_Leaf.C(value=1), _Leaf.C(value=2)]})
    )


def test_nested_generic_config_missing_param_caught():
    """A missing required param on a Config buried in nested generics is caught."""
    expect_notvalidate(NestedConfigConfig.C(items={"g": [_Leaf.C(value=1), _Leaf.C()]}))


def test_nested_generic_config_wrong_type_caught():
    """A non-Config where a Config is expected, deep in a nested structure."""
    with pytest.raises((ValueError, TypeError)):
        NestedConfigConfig.C(items={"g": [_Leaf.C(value=1), 42]})
