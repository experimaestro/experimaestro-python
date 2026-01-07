"""Tests for partial (partial identifier computation)"""

from experimaestro import (
    Config,
    Task,
    Param,
    field,
    partial,
    param_group,
    ParameterGroup,
    Partial,
)


# Define parameter groups at module level
iter_group = param_group("iter")
model_group = param_group("model")


class TestPartialBasic:
    """Test basic partial functionality"""

    def test_param_group_creation(self):
        """Test creating parameter groups"""
        group = param_group("test")
        assert isinstance(group, ParameterGroup)
        assert group.name == "test"

    def test_param_group_hashable(self):
        """Test that parameter groups are hashable"""
        group1 = param_group("test")
        group2 = param_group("test")
        # Same name means equal (frozen dataclass uses value equality)
        assert group1 == group2
        # Both should be hashable and deduplicated
        s = {group1, group2}
        assert len(s) == 1

        # Different names should be different
        group3 = param_group("other")
        assert group1 != group3
        s2 = {group1, group3}
        assert len(s2) == 2

    def test_partials_creation(self):
        """Test creating partial"""
        sp = partial(exclude_groups=[iter_group])
        assert isinstance(sp, Partial)
        assert iter_group in sp.exclude_groups

    def test_partials_is_excluded(self):
        """Test is_excluded method"""
        sp = partial(exclude_groups=[iter_group])
        assert sp.is_excluded({iter_group}) is True
        assert sp.is_excluded({model_group}) is False
        assert sp.is_excluded(set()) is False

    def test_partials_include_overrides_exclude(self):
        """Test that include_groups overrides exclude_groups"""
        sp = partial(
            exclude_groups=[iter_group, model_group], include_groups=[iter_group]
        )
        # iter_group is in both, but include wins
        assert sp.is_excluded({iter_group}) is False
        assert sp.is_excluded({model_group}) is True

    def test_partials_exclude_all(self):
        """Test exclude_all option"""
        sp = partial(exclude_all=True, include_groups=[model_group])
        assert sp.is_excluded({iter_group}) is True
        assert sp.is_excluded({model_group}) is False
        assert sp.is_excluded(set()) is True

    def test_partials_exclude_no_group(self):
        """Test exclude_no_group option"""
        sp = partial(exclude_no_group=True)
        assert sp.is_excluded(set()) is True
        assert sp.is_excluded({iter_group}) is False


class TestPartialIdentifiers:
    """Test partial identifier computation"""

    def test_field_groups(self):
        """Test that field groups are correctly stored in Argument"""

        class MyConfig(Config):
            x: Param[int] = field(groups=[iter_group])
            y: Param[float]

        xpmtype = MyConfig.__getxpmtype__()
        xpmtype.__initialize__()

        assert iter_group in xpmtype.arguments["x"].groups
        assert len(xpmtype.arguments["y"].groups) == 0

    def test_partials_collected_in_objecttype(self):
        """Test that partial are collected in ObjectType"""

        class MyTask(Task):
            checkpoints = partial(exclude_groups=[iter_group])
            x: Param[int]

        xpmtype = MyTask.__getxpmtype__()
        xpmtype.__initialize__()

        assert "checkpoints" in xpmtype._partials
        assert xpmtype._partials["checkpoints"].name == "checkpoints"

    def test_partial_identifier_same_when_excluded_differs(self):
        """Test that partial identifiers are the same when only excluded params differ"""

        class MyTask(Task):
            checkpoints = partial(exclude_groups=[iter_group])
            max_iter: Param[int] = field(groups=[iter_group])
            learning_rate: Param[float]

        c1 = MyTask.C(max_iter=100, learning_rate=0.1)
        c2 = MyTask.C(max_iter=200, learning_rate=0.1)

        # Regular identifiers should differ
        assert c1.__xpm__.identifier != c2.__xpm__.identifier

        # Partial identifiers should be the same
        pid1 = c1.__xpm__.get_partial_identifier(MyTask.checkpoints)
        pid2 = c2.__xpm__.get_partial_identifier(MyTask.checkpoints)
        assert pid1 == pid2

    def test_partial_identifier_differs_when_included_differs(self):
        """Test that partial identifiers differ when included params differ"""

        class MyTask(Task):
            checkpoints = partial(exclude_groups=[iter_group])
            max_iter: Param[int] = field(groups=[iter_group])
            learning_rate: Param[float]

        c1 = MyTask.C(max_iter=100, learning_rate=0.1)
        c2 = MyTask.C(max_iter=100, learning_rate=0.2)

        # Partial identifiers should differ (learning_rate is not excluded)
        pid1 = c1.__xpm__.get_partial_identifier(MyTask.checkpoints)
        pid2 = c2.__xpm__.get_partial_identifier(MyTask.checkpoints)
        assert pid1 != pid2

    def test_partial_identifier_with_multiple_groups(self):
        """Test partial identifiers with parameters in multiple groups"""

        class MyTask(Task):
            checkpoints = partial(exclude_groups=[iter_group])
            # This parameter is in both groups
            x: Param[int] = field(groups=[iter_group, model_group])
            y: Param[float]

        c1 = MyTask.C(x=1, y=0.1)
        c2 = MyTask.C(x=2, y=0.1)

        # Partial identifiers should be the same (x is in iter_group which is excluded)
        pid1 = c1.__xpm__.get_partial_identifier(MyTask.checkpoints)
        pid2 = c2.__xpm__.get_partial_identifier(MyTask.checkpoints)
        assert pid1 == pid2
