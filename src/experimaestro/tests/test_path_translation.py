
import pytest
from pathlib import Path
from experimaestro.scheduler.state_provider import StateProvider
from experimaestro.scheduler.remote.client import SSHStateProviderClient
from experimaestro.scheduler.interfaces import BaseJob, BaseExperiment

class MockJob(BaseJob):
    def __init__(self, path):
        self.path = path

class MockExperiment(BaseExperiment):
    def __init__(self, workdir):
        self.workdir = workdir

class MockStateProvider(StateProvider):
    def clean_job(self, *args, **kwargs): pass
    def close(self, *args, **kwargs): pass
    def get_all_jobs(self, *args, **kwargs): return []
    def get_current_run(self, *args, **kwargs): return None
    def get_dependencies_map(self, *args, **kwargs): return {}
    def get_experiment(self, *args, **kwargs): return None
    def get_experiment_job_info(self, *args, **kwargs): return None
    def get_experiment_runs(self, *args, **kwargs): return []
    def get_experiments(self, *args, **kwargs): return []
    def get_job(self, *args, **kwargs): return None
    def get_jobs(self, *args, **kwargs): return []
    def get_services(self, *args, **kwargs): return []
    def get_tags_map(self, *args, **kwargs): return {}
    def kill_job(self, *args, **kwargs): pass

def test_state_provider_translate_path():
    provider = MockStateProvider()
    path = Path("/some/local/path")
    assert provider.translate_path(path) == str(path)
    
    job = MockJob(path)
    assert provider.get_display_path(job) == str(path)

def test_ssh_state_provider_client_translate_path():
    client = SSHStateProviderClient(
        host="remote-host",
        remote_workspace="/remote/workspace"
    )
    # Set local_cache_dir which is normally set during connect/init
    client.local_cache_dir = Path("/tmp/local/cache")
    
    # Path in local cache should be translated
    local_path = Path("/tmp/local/cache/experiments/exp1/run1")
    expected_remote = "/remote/workspace/experiments/exp1/run1"
    assert client.translate_path(local_path) == expected_remote
    
    # Path NOT in local cache should be returned as-is
    other_path = Path("/some/other/path")
    assert client.translate_path(other_path) == str(other_path)
    
    # Test get_display_path (which now uses translate_path)
    job = MockJob(local_path)
    assert client.get_display_path(job) == expected_remote
    
    # Test for experiments (direct use of translate_path as in experiments.py)
    exp = MockExperiment(local_path)
    assert client.translate_path(exp.workdir) == expected_remote
