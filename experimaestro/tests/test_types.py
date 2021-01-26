# --- Task and types definitions

import logging
from experimaestro import Config, config

from .utils import TemporaryExperiment
from experimaestro.scheduler import JobState

from .definitions_types import *


def test_simple():
    with TemporaryExperiment("simple") as xp:
        assert IntegerTask(value=5).submit().__xpm__.job.wait() == JobState.DONE
        assert FloatTask(value=5.1).submit().__xpm__.job.wait() == JobState.DONE


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

        assert ctype.objecttype == C.Object

        assert issubclass(C.Object, B1.Object)
        assert issubclass(C.Object, B.Object)
        assert issubclass(C.Object, A.Object)
        assert not issubclass(C.Object, Config)


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
    assert issubclass(B.Object, A1.Object)
    assert issubclass(B.Object, A.Object)
