"""Tests for workspace trigger matching (issue #119)"""

from pathlib import Path
from experimaestro.settings import WorkspaceSettings, find_workspace, Settings
from unittest.mock import patch


def test_workspace_trigger_exact_match():
    """Test exact match trigger"""
    workspaces = [
        WorkspaceSettings(
            id="neuralir",
            path=Path("/tmp/test1"),
            triggers=["my-awesome-experiment"],
        ),
        WorkspaceSettings(
            id="default",
            path=Path("/tmp/test2"),
        ),
    ]

    settings = Settings(workspaces=workspaces)

    with patch("experimaestro.settings.get_settings", return_value=settings):
        ws = find_workspace(experiment_id="my-awesome-experiment")
        assert ws.id == "neuralir"


def test_workspace_trigger_glob_match():
    """Test glob pattern trigger"""
    workspaces = [
        WorkspaceSettings(
            id="neuralir",
            path=Path("/tmp/test1"),
            triggers=["base_id-*"],
        ),
        WorkspaceSettings(
            id="default",
            path=Path("/tmp/test2"),
        ),
    ]

    settings = Settings(workspaces=workspaces)

    with patch("experimaestro.settings.get_settings", return_value=settings):
        ws = find_workspace(experiment_id="base_id-123")
        assert ws.id == "neuralir"

        ws = find_workspace(experiment_id="base_id-test")
        assert ws.id == "neuralir"


def test_workspace_trigger_multiple_patterns():
    """Test multiple trigger patterns"""
    workspaces = [
        WorkspaceSettings(
            id="neuralir",
            path=Path("/tmp/test1"),
            triggers=["base_id-*", "my-awesome-experiment", "test-*"],
        ),
        WorkspaceSettings(
            id="default",
            path=Path("/tmp/test2"),
        ),
    ]

    settings = Settings(workspaces=workspaces)

    with patch("experimaestro.settings.get_settings", return_value=settings):
        ws = find_workspace(experiment_id="base_id-123")
        assert ws.id == "neuralir"

        ws = find_workspace(experiment_id="my-awesome-experiment")
        assert ws.id == "neuralir"

        ws = find_workspace(experiment_id="test-foo")
        assert ws.id == "neuralir"


def test_workspace_trigger_no_match_uses_default():
    """Test that default workspace (first in list) is used when no trigger matches"""
    workspaces = [
        WorkspaceSettings(
            id="default",
            path=Path("/tmp/test1"),
        ),
        WorkspaceSettings(
            id="neuralir",
            path=Path("/tmp/test2"),
            triggers=["base_id-*"],
        ),
    ]

    settings = Settings(workspaces=workspaces)

    with patch("experimaestro.settings.get_settings", return_value=settings):
        ws = find_workspace(experiment_id="other-experiment")
        assert ws.id == "default"  # First workspace is the default


def test_workspace_trigger_first_match_wins():
    """Test that first matching workspace is selected"""
    workspaces = [
        WorkspaceSettings(
            id="first",
            path=Path("/tmp/test1"),
            triggers=["test-*"],
        ),
        WorkspaceSettings(
            id="second",
            path=Path("/tmp/test2"),
            triggers=["test-*"],
        ),
    ]

    settings = Settings(workspaces=workspaces)

    with patch("experimaestro.settings.get_settings", return_value=settings):
        ws = find_workspace(experiment_id="test-experiment")
        assert ws.id == "first"


def test_workspace_explicit_takes_precedence():
    """Test that explicit workspace parameter overrides triggers"""
    workspaces = [
        WorkspaceSettings(
            id="neuralir",
            path=Path("/tmp/test1"),
            triggers=["base_id-*"],
        ),
        WorkspaceSettings(
            id="other",
            path=Path("/tmp/test2"),
        ),
    ]

    settings = Settings(workspaces=workspaces)

    with patch("experimaestro.settings.get_settings", return_value=settings):
        # Even though experiment_id matches neuralir trigger, explicit workspace wins
        ws = find_workspace(workspace="other", experiment_id="base_id-123")
        assert ws.id == "other"


def test_workspace_no_triggers_backward_compatible():
    """Test that workspaces without triggers still work (backward compatibility)"""
    workspaces = [
        WorkspaceSettings(
            id="default",
            path=Path("/tmp/test1"),
        ),
    ]

    settings = Settings(workspaces=workspaces)

    with patch("experimaestro.settings.get_settings", return_value=settings):
        ws = find_workspace(experiment_id="any-experiment")
        assert ws.id == "default"
