import pytest

from experimaestro import Config
from experimaestro.core.arguments import Annotated
from experimaestro.checkers import Choices

# Mark all tests in this module as config tests
pytestmark = pytest.mark.config


def test_choices():
    """Test choices"""

    class TestChoices(Config):
        a: Annotated[str, Choices(["a", "b"])]

    TestChoices.C(a="a").__xpm__.validate()

    with pytest.raises((ValueError, KeyError)):
        TestChoices.C(a="c").__xpm__.validate()
