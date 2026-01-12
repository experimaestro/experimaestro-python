"""Tests for environment capture utilities"""

import json
import pytest

from experimaestro.utils.git import get_git_info
from experimaestro.utils.environment import (
    get_environment_info,
    get_editable_packages_git_info,
    get_current_environment,
    load_environment_info,
    ExperimentEnvironment,
)


class TestGetGitInfo:
    """Tests for get_git_info function"""

    def test_returns_dict_in_git_repo(self, tmp_path):
        """Test that get_git_info returns a dict when in a git repo"""
        # Use the current working directory which should be a git repo
        git_info = get_git_info()

        assert git_info is not None
        assert isinstance(git_info, dict)
        assert "commit" in git_info
        assert "commit_short" in git_info
        assert "branch" in git_info
        assert "dirty" in git_info
        assert "message" in git_info
        assert "author" in git_info
        assert "date" in git_info

    def test_commit_format(self):
        """Test that commit hashes have correct format"""
        git_info = get_git_info()
        if git_info is None:
            pytest.skip("Not in a git repository")

        # Full commit should be 40 hex characters
        assert len(git_info["commit"]) == 40
        assert all(c in "0123456789abcdef" for c in git_info["commit"])

        # Short commit should be 7-12 characters (git uses more if needed for uniqueness)
        assert 7 <= len(git_info["commit_short"]) <= 12

    def test_returns_none_for_non_git_dir(self, tmp_path):
        """Test that get_git_info returns None for non-git directories"""
        git_info = get_git_info(tmp_path)
        assert git_info is None

    def test_dirty_flag(self):
        """Test that dirty flag is a boolean"""
        git_info = get_git_info()
        if git_info is None:
            pytest.skip("Not in a git repository")

        assert isinstance(git_info["dirty"], bool)


class TestGetEnvironmentInfo:
    """Tests for get_environment_info function"""

    def test_returns_dict_with_required_keys(self):
        """Test that get_environment_info returns dict with required keys"""
        env_info = get_environment_info()

        assert isinstance(env_info, dict)
        assert "python_version" in env_info
        assert "packages" in env_info
        assert "editable_packages" in env_info

    def test_python_version_format(self):
        """Test that python_version has correct format"""
        env_info = get_environment_info()
        version = env_info["python_version"]

        # Should be in format X.Y.Z
        parts = version.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    def test_packages_is_dict(self):
        """Test that packages is a dict of name -> version"""
        env_info = get_environment_info()
        packages = env_info["packages"]

        assert isinstance(packages, dict)
        assert len(packages) > 0  # Should have at least some packages

        # Check that all values are strings (versions)
        for name, version in packages.items():
            assert isinstance(name, str)
            assert isinstance(version, str)

    def test_experimaestro_is_editable(self):
        """Test that experimaestro itself is detected as editable"""
        env_info = get_environment_info()
        editable = env_info["editable_packages"]

        # When running tests, experimaestro should be installed in editable mode
        assert "experimaestro" in editable
        assert "version" in editable["experimaestro"]
        assert "path" in editable["experimaestro"]
        assert "git" in editable["experimaestro"]

    def test_editable_package_has_git_info(self):
        """Test that editable packages include git info"""
        env_info = get_environment_info()
        editable = env_info["editable_packages"]

        # experimaestro should have git info since it's in a git repo
        if "experimaestro" in editable:
            git_info = editable["experimaestro"]["git"]
            if git_info is not None:  # May be None if not in git repo
                assert "commit" in git_info
                assert "dirty" in git_info


class TestGetEditablePackagesGitInfo:
    """Tests for get_editable_packages_git_info function"""

    def test_returns_dict(self):
        """Test that function returns a dict"""
        result = get_editable_packages_git_info()
        assert isinstance(result, dict)

    def test_contains_experimaestro(self):
        """Test that experimaestro is in the result"""
        result = get_editable_packages_git_info()
        assert "experimaestro" in result


class TestSaveAndLoadEnvironmentInfo:
    """Tests for get_current_environment and load_environment_info functions"""

    def test_save_creates_file(self, tmp_path):
        """Test that get_current_environment + save creates a JSON file"""
        path = tmp_path / "environment.json"

        env = get_current_environment()
        env.save(path)

        assert path.exists()
        assert isinstance(env, ExperimentEnvironment)

    def test_save_writes_valid_json(self, tmp_path):
        """Test that saved file contains valid JSON"""
        path = tmp_path / "environment.json"

        env = get_current_environment()
        env.save(path)

        content = json.loads(path.read_text())
        assert "python_version" in content
        assert "packages" in content
        assert "editable_packages" in content

    def test_load_reads_saved_data(self, tmp_path):
        """Test that load_environment_info reads back saved data"""
        path = tmp_path / "environment.json"

        saved = get_current_environment()
        saved.save(path)
        loaded = load_environment_info(path)

        assert loaded.python_version == saved.python_version
        assert loaded.packages == saved.packages
        assert loaded.editable_packages == saved.editable_packages

    def test_load_returns_none_for_missing_file(self, tmp_path):
        """Test that load returns None for non-existent file"""
        path = tmp_path / "nonexistent.json"

        result = load_environment_info(path)

        assert result is None

    def test_load_returns_empty_for_invalid_json(self, tmp_path):
        """Test that load returns empty ExperimentEnvironment for invalid JSON"""
        path = tmp_path / "invalid.json"
        path.write_text("not valid json{")

        result = load_environment_info(path)

        # Returns empty ExperimentEnvironment (graceful degradation)
        assert result is not None
        assert result.python_version is None
        assert result.packages == {}
        assert result.editable_packages == {}


class TestExperimentEnvironmentSaving:
    """Integration tests for environment saving in experiments"""

    def test_experiment_saves_environment_info(self, xpmdirectory):
        """Test that experiment saves environment.json on start"""
        from experimaestro import experiment

        # Just enter the experiment context, no need to run any tasks
        with experiment(xpmdirectory, "test-env-save") as xp:
            pass  # environment.json should be saved on __enter__

        env_path = xp.workdir / "environment.json"
        assert env_path.exists()

        env_info = json.loads(env_path.read_text())
        assert "python_version" in env_info
        assert "packages" in env_info
        assert "editable_packages" in env_info
