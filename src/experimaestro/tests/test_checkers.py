from experimaestro import Config
from experimaestro.core.arguments import Annotated
from experimaestro.checkers import Choices
import pytest


def test_choices():
    """Test choices"""

    class TestChoices(Config):
        a: Annotated[str, Choices(["a", "b"])]

    TestChoices(a="a").__xpm__.validate()

    with pytest.raises((ValueError, KeyError)):
        TestChoices(a="c").__xpm__.validate()
