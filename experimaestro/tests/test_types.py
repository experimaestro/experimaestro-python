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
    @config()
    class A:
        pass

    class A1(A):
        pass

    @config()
    class B(A1):
        pass

    B.__xpmtype__

    assert issubclass(B, A)
    assert issubclass(B, A1)
