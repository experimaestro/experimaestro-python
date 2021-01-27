# --- Task and types definitions

import logging
from experimaestro import Config, config
from experimaestro.core.objects import TypeConfig

from .utils import TemporaryExperiment
from experimaestro.scheduler import JobState


def test_multiple_inheritance():
    @config()
    class A:
        pass

    @config()
    class B:
        pass

    @config()
    class B1(B):
        pass

    @config()
    class C1(B1, A):
        pass

    @config()
    class C2(A, B1):
        pass

    for C in (C1, C2):
        logging.info("Testing %s", C)
        ctype = C.xpmtype()
        assert issubclass(C, A)
        assert issubclass(C, B)
        assert issubclass(C, B1)

        assert ctype.objecttype == C.xpmtype().objecttype

        assert issubclass(C.xpmtype().objecttype, B1.xpmtype().basetype)
        assert issubclass(C.xpmtype().objecttype, B.xpmtype().basetype)
        assert issubclass(C.xpmtype().objecttype, A.xpmtype().basetype)
        assert not issubclass(C.xpmtype().objecttype, TypeConfig)


def test_missing_hierarchy():
    @config()
    class A:
        pass

    class A1(A):
        pass

    @config()
    class B(A1):
        pass

    B.xpmtype()

    assert issubclass(B, A)
    assert issubclass(B, A1)
