"""Tests for DynamicLauncher and launcher priority."""

import pytest
from collections import Counter

from experimaestro.launchers import DynamicLauncher
from experimaestro.launchers.direct import DirectLauncher
from experimaestro.connectors.local import LocalConnector
from experimaestro.scriptbuilder import PythonScriptBuilder


@pytest.fixture
def connector():
    """Provides a local connector for tests."""
    return LocalConnector.instance()


class TestLauncherPriority:
    """Tests for the priority property on Launcher."""

    def test_default_priority(self, connector):
        """Test that default priority is 0."""
        launcher = DirectLauncher(connector)
        assert launcher.priority == 0

    def test_custom_priority(self, connector):
        """Test setting custom priority."""
        launcher = DirectLauncher(connector, priority=5)
        assert launcher.priority == 5

    def test_negative_priority(self, connector):
        """Test that negative priorities are allowed."""
        launcher = DirectLauncher(connector, priority=-10)
        assert launcher.priority == -10

    def test_float_priority(self, connector):
        """Test that float priorities work."""
        launcher = DirectLauncher(connector, priority=3.5)
        assert launcher.priority == 3.5


class TestDynamicLauncherInit:
    """Tests for DynamicLauncher initialization."""

    def test_empty_launchers_raises(self):
        """Test that empty launcher list raises ValueError."""
        with pytest.raises(ValueError, match="requires at least one launcher"):
            DynamicLauncher([])

    def test_single_launcher(self, connector):
        """Test initialization with single launcher."""
        launcher = DirectLauncher(connector)
        dynamic = DynamicLauncher([launcher])
        assert len(dynamic._launchers) == 1

    def test_multiple_launchers(self, connector):
        """Test initialization with multiple launchers."""
        launchers = [DirectLauncher(connector, priority=i) for i in range(5)]
        dynamic = DynamicLauncher(launchers)
        assert len(dynamic._launchers) == 5

    def test_connector_from_first_launcher(self, connector):
        """Test that connector defaults to first launcher's connector."""
        launcher1 = DirectLauncher(connector)
        launcher2 = DirectLauncher(connector)
        dynamic = DynamicLauncher([launcher1, launcher2])
        assert dynamic.connector is connector

    def test_custom_connector(self, connector):
        """Test that custom connector can be provided."""
        launcher = DirectLauncher(connector)
        custom_connector = LocalConnector()
        dynamic = DynamicLauncher([launcher], connector=custom_connector)
        assert dynamic.connector is custom_connector

    def test_sample_mode(self, connector):
        """Test sample parameter initialization."""
        launcher = DirectLauncher(connector, priority=1)
        dynamic_no_sample = DynamicLauncher([launcher], sample=False)
        dynamic_sample = DynamicLauncher([launcher], sample=True)
        assert dynamic_no_sample._sample is False
        assert dynamic_sample._sample is True

    def test_dynamic_launcher_priority(self, connector):
        """Test that DynamicLauncher itself can have a priority."""
        launcher = DirectLauncher(connector)
        dynamic = DynamicLauncher([launcher], priority=100)
        assert dynamic.priority == 100


class TestLauncherSorting:
    """Tests for launcher sorting by priority."""

    def test_launchers_sorted_by_priority(self, connector):
        """Test that launchers property returns sorted list."""
        launcher1 = DirectLauncher(connector, priority=1)
        launcher2 = DirectLauncher(connector, priority=10)
        launcher3 = DirectLauncher(connector, priority=5)

        dynamic = DynamicLauncher([launcher1, launcher2, launcher3])
        sorted_priorities = [lnch.priority for lnch in dynamic.launchers]
        assert sorted_priorities == [10, 5, 1]

    def test_original_list_unchanged(self, connector):
        """Test that original list is not modified."""
        launcher1 = DirectLauncher(connector, priority=1)
        launcher2 = DirectLauncher(connector, priority=10)

        dynamic = DynamicLauncher([launcher1, launcher2])
        # Access sorted list
        _ = dynamic.launchers
        # Original order should be preserved
        assert dynamic._launchers[0] is launcher1
        assert dynamic._launchers[1] is launcher2


class TestSelectionWithoutSampling:
    """Tests for launcher selection without sampling mode."""

    def test_selects_highest_priority(self, connector):
        """Test that highest priority launcher is selected."""
        low = DirectLauncher(connector, priority=1)
        high = DirectLauncher(connector, priority=10)

        dynamic = DynamicLauncher([low, high], sample=False)

        # Run multiple times - should always select high priority
        for _ in range(10):
            selected = dynamic.select_launcher()
            assert selected is high

    def test_uniform_selection_among_ties(self, connector):
        """Test uniform random selection when priorities tie."""
        launcher1 = DirectLauncher(connector, priority=10)
        launcher2 = DirectLauncher(connector, priority=10)
        launcher3 = DirectLauncher(connector, priority=1)

        dynamic = DynamicLauncher([launcher1, launcher2, launcher3], sample=False)

        # Run many times and check distribution
        selections = Counter()
        for _ in range(1000):
            selected = dynamic.select_launcher()
            selections[id(selected)] += 1

        # Should only select from the two tied launchers
        assert id(launcher3) not in selections
        # Both tied launchers should be selected (with some tolerance)
        assert selections[id(launcher1)] > 300
        assert selections[id(launcher2)] > 300


class TestSelectionWithSampling:
    """Tests for launcher selection with sampling mode."""

    def test_samples_proportionally(self, connector):
        """Test that sampling is proportional to priority."""
        high = DirectLauncher(connector, priority=9)
        low = DirectLauncher(connector, priority=1)

        dynamic = DynamicLauncher([high, low], sample=True)

        selections = Counter()
        for _ in range(1000):
            selected = dynamic.select_launcher()
            selections[id(selected)] += 1

        # High priority (9) should be selected ~90% of the time
        high_ratio = selections[id(high)] / 1000
        assert 0.85 < high_ratio < 0.95

    def test_negative_priority_raises(self, connector):
        """Test that negative priority raises error in sample mode."""
        neg = DirectLauncher(connector, priority=-1)
        pos = DirectLauncher(connector, priority=5)

        dynamic = DynamicLauncher([neg, pos], sample=True)

        with pytest.raises(ValueError, match="must be positive"):
            dynamic.select_launcher()

    def test_zero_priority_raises(self, connector):
        """Test that zero priority raises error in sample mode."""
        zero = DirectLauncher(connector, priority=0)
        pos = DirectLauncher(connector, priority=5)

        dynamic = DynamicLauncher([zero, pos], sample=True)

        with pytest.raises(ValueError, match="must be positive"):
            dynamic.select_launcher()


class TestMethodDelegation:
    """Tests for method delegation to selected launcher."""

    def test_scriptbuilder_delegation(self, connector):
        """Test that scriptbuilder is delegated to selected launcher."""
        launcher = DirectLauncher(connector, priority=10)
        dynamic = DynamicLauncher([launcher])

        sb = dynamic.scriptbuilder()
        assert isinstance(sb, PythonScriptBuilder)

    def test_processbuilder_delegation(self, connector):
        """Test that processbuilder is delegated to selected launcher."""
        launcher = DirectLauncher(connector, priority=10)
        dynamic = DynamicLauncher([launcher])

        # First call scriptbuilder to select launcher
        dynamic.scriptbuilder()
        pb = dynamic.processbuilder()
        # Should return connector's process builder
        assert pb is not None

    def test_launcher_info_code_delegation(self, connector):
        """Test that launcher_info_code is delegated to selected launcher."""
        launcher = DirectLauncher(connector, priority=10)
        dynamic = DynamicLauncher([launcher])

        # First call scriptbuilder to select launcher
        dynamic.scriptbuilder()
        info_code = dynamic.launcher_info_code()
        # DirectLauncher returns empty string
        assert info_code == ""

    def test_selected_launcher_persists(self, connector):
        """Test that selected launcher persists across calls."""
        launcher1 = DirectLauncher(connector, priority=10)
        launcher2 = DirectLauncher(connector, priority=10)

        dynamic = DynamicLauncher([launcher1, launcher2])

        # Select via scriptbuilder
        dynamic.scriptbuilder()
        selected = dynamic._selected_launcher

        # Subsequent calls should use same launcher
        dynamic.processbuilder()
        assert dynamic._selected_launcher is selected

        dynamic.launcher_info_code()
        assert dynamic._selected_launcher is selected


class TestAddRemoveLaunchers:
    """Tests for adding and removing launchers."""

    def test_add_launcher(self, connector):
        """Test adding a launcher."""
        launcher1 = DirectLauncher(connector, priority=1)
        dynamic = DynamicLauncher([launcher1])

        launcher2 = DirectLauncher(connector, priority=10)
        dynamic.add_launcher(launcher2)

        assert len(dynamic._launchers) == 2
        assert launcher2 in dynamic._launchers

    def test_remove_launcher(self, connector):
        """Test removing a launcher."""
        launcher1 = DirectLauncher(connector, priority=1)
        launcher2 = DirectLauncher(connector, priority=10)
        dynamic = DynamicLauncher([launcher1, launcher2])

        dynamic.remove_launcher(launcher1)

        assert len(dynamic._launchers) == 1
        assert launcher1 not in dynamic._launchers

    def test_remove_nonexistent_raises(self, connector):
        """Test that removing non-existent launcher raises error."""
        launcher1 = DirectLauncher(connector, priority=1)
        launcher2 = DirectLauncher(connector, priority=10)
        dynamic = DynamicLauncher([launcher1])

        with pytest.raises(ValueError):
            dynamic.remove_launcher(launcher2)


class TestUpdateMethod:
    """Tests for the update method."""

    def test_update_called_on_select(self, connector):
        """Test that update is called when selecting launcher."""
        launcher = DirectLauncher(connector, priority=10)
        dynamic = DynamicLauncher([launcher])

        update_called = []

        def mock_update():
            update_called.append(True)

        dynamic.update = mock_update

        dynamic.select_launcher()
        assert len(update_called) == 1

    def test_custom_update_modifies_priorities(self, connector):
        """Test that custom update can modify priorities."""
        launcher1 = DirectLauncher(connector, priority=1)
        launcher2 = DirectLauncher(connector, priority=10)

        class CustomDynamicLauncher(DynamicLauncher):
            def update(self):
                # Swap priorities
                for launcher in self._launchers:
                    if launcher.priority == 1:
                        launcher.priority = 100
                    elif launcher.priority == 10:
                        launcher.priority = 0

        dynamic = CustomDynamicLauncher([launcher1, launcher2])

        # After update, launcher1 should have highest priority
        selected = dynamic.select_launcher()
        assert selected is launcher1


class TestStringRepresentation:
    """Tests for string representation."""

    def test_str_few_launchers(self, connector):
        """Test string representation with few launchers."""
        launcher1 = DirectLauncher(connector)
        launcher2 = DirectLauncher(connector)
        dynamic = DynamicLauncher([launcher1, launcher2])

        s = str(dynamic)
        assert "DynamicLauncher" in s
        assert "sample=False" in s

    def test_str_many_launchers(self, connector):
        """Test string representation with many launchers shows count."""
        launchers = [DirectLauncher(connector) for _ in range(10)]
        dynamic = DynamicLauncher(launchers)

        s = str(dynamic)
        assert "10 total" in s

    def test_str_with_sampling(self, connector):
        """Test string representation shows sample mode."""
        launcher = DirectLauncher(connector, priority=1)
        dynamic = DynamicLauncher([launcher], sample=True)

        s = str(dynamic)
        assert "sample=True" in s
