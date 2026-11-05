"""The former names of experiment specifications still work, with a warning"""

import pytest

from experimaestro.experiments import ExperimentSpecification


def test_configuration_decorator_and_base_are_deprecated():
    from experimaestro.experiments import ConfigurationBase, configuration

    with pytest.warns(DeprecationWarning, match="ConfigurationBase"):

        class Legacy(ConfigurationBase):
            value: int = 1

    with pytest.warns(DeprecationWarning, match="specification"):
        Legacy = configuration()(Legacy)

    with pytest.warns(DeprecationWarning, match="specification"):

        @configuration
        class LegacyNoParens(ExperimentSpecification):
            other: str = "a"

    spec = Legacy(id="xp", value=3)
    assert isinstance(spec, ExperimentSpecification)
    assert spec.value == 3
    assert LegacyNoParens(id="xp").other == "a"
