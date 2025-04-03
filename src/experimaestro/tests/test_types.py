# --- Task and types definitions

import logging
from experimaestro import Config, Param
from typing import Union

import pytest
from experimaestro.core.objects import TypeConfig


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
        ctype = C.__xpmtype__
        assert issubclass(C, A)
        assert issubclass(C, B)
        assert issubclass(C, B1)

        assert ctype.objecttype == C.__xpmtype__.objecttype

        assert issubclass(C.__xpmtype__.objecttype, B1.__xpmtype__.basetype)
        assert issubclass(C.__xpmtype__.objecttype, B.__xpmtype__.basetype)
        assert issubclass(C.__xpmtype__.objecttype, A.__xpmtype__.basetype)
        assert not issubclass(C.__xpmtype__.objecttype, TypeConfig)


def test_missing_hierarchy():
    class A(Config):
        pass

    class A1(A):
        pass

    class B(A1):
        pass

    B.__xpmtype__

    assert issubclass(B, A)
    assert issubclass(B, A1)


def test_types_union():
    class A(Config):
        x: Param[Union[int, str]]

    A(x=1)
    A(x="hello")
    with pytest.raises(ValueError):
        A(x=[])
