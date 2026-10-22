"""The legacy typing forms (List, Dict, Optional, Union, ...) behave like the
builtin ones (list, dict, X | None, X | Y).

The rest of the code base uses the builtin forms (ruff UP rules): this file is
exempt from them so that user code written with the legacy forms stays covered.
"""

from typing import Dict, List, Optional, Set, Tuple, Union

import pytest

from experimaestro import Config, Param
from experimaestro.core.types import Type, _is_type_arg_compatible

pytestmark = [pytest.mark.types]


EQUIVALENT_TYPES = [
    (List[int], list[int]),
    (Set[str], set[str]),
    (Dict[str, int], dict[str, int]),
    (Tuple[int, str], tuple[int, str]),
    (Tuple[int, ...], tuple[int, ...]),
    (Optional[int], int | None),
    (Union[int, str], int | str),
    (List[Optional[int]], list[int | None]),
    (Dict[str, List[int]], dict[str, list[int]]),
]


@pytest.mark.parametrize("legacy,builtin", EQUIVALENT_TYPES, ids=str)
def test_legacy_type_same_xpm_type(legacy, builtin):
    assert Type.fromType(legacy).name() == Type.fromType(builtin).name()


@pytest.mark.parametrize("legacy,builtin", EQUIVALENT_TYPES, ids=str)
def test_legacy_type_arg_compatible(legacy, builtin):
    assert _is_type_arg_compatible(legacy, builtin)
    assert _is_type_arg_compatible(builtin, legacy)


@pytest.mark.parametrize("union", [Union[int, str], int | str], ids=str)
def test_union_type_arg(union):
    assert _is_type_arg_compatible(int, union)
    assert _is_type_arg_compatible(str, union)
    assert not _is_type_arg_compatible(float, union)


class LegacyParams(Config):
    __xpmid__ = "test.legacy_typing.params"

    values: Param[List[int]]
    mapping: Param[Dict[str, float]]
    name: Param[Optional[str]]
    either: Param[Union[int, str]]


class BuiltinParams(Config):
    __xpmid__ = "test.legacy_typing.params"

    values: Param[list[int]]
    mapping: Param[dict[str, float]]
    name: Param[str | None]
    either: Param[int | str]


@pytest.mark.parametrize("name", ["x", None])
@pytest.mark.parametrize("either", [1, "a"])
def test_legacy_params_same_identifier(name, either):
    kwargs = dict(values=[1, 2], mapping={"a": 1.0}, name=name, either=either)
    legacy = LegacyParams.C(**kwargs)
    builtin = BuiltinParams.C(**kwargs)
    assert legacy.__xpm__.identifier.all == builtin.__xpm__.identifier.all


def test_legacy_params_validation():
    with pytest.raises(Exception):
        LegacyParams.C(
            values=["not an int"], mapping={}, name=None, either=1
        ).__xpm__.validate()
