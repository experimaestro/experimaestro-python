# --- Task and types definitions

import logging
from experimaestro import Config, Param
from typing import Union

import pytest
from experimaestro.core.objects import ConfigMixin


def test_multiple_inheritance():
    class A(Config):
        pass

    class B(Config):
        pass

    class B1(B):
        pass

    class C1(B1, A):
        pass

    class C2(A, B1):
        pass

    for C in (C1, C2):
        logging.info("Testing %s", C)
        ctype = C.__getxpmtype__()
        assert issubclass(C, A)
        assert issubclass(C, B)
        assert issubclass(C, B1)

        assert ctype.value_type == C.__getxpmtype__().value_type

        assert issubclass(C.__getxpmtype__().value_type, B1.__getxpmtype__().value_type)
        assert issubclass(C.__getxpmtype__().value_type, B.__getxpmtype__().value_type)
        assert issubclass(C.__getxpmtype__().value_type, A.__getxpmtype__().value_type)
        assert not issubclass(C.__getxpmtype__().value_type, ConfigMixin)


def test_missing_hierarchy():
    class A(Config):
        pass

    class A1(A):
        pass

    class B(A1):
        pass

    B.__getxpmtype__()

    assert issubclass(B, A)
    assert issubclass(B, A1)


def test_types_union():
    class A(Config):
        x: Param[Union[int, str]]

    A.C(x=1)
    A.C(x="hello")
    with pytest.raises(ValueError):
        A.C(x=[])
