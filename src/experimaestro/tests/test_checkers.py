from experimaestro import config
from experimaestro.core.arguments import Annotated
from experimaestro.checkers import Choices
import pytest


def test_choices():
    """Test choices"""

    @config()
    class TestChoices:
        a: Annotated[str, Choices(["a", "b"])]

    TestChoices(a="a").__xpm__.validate()

    with pytest.raises((ValueError, KeyError)):
        TestChoices(a="c").__xpm__.validate()
