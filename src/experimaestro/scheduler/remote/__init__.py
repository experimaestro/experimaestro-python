"""Remote monitoring support for experimaestro

This package provides SSH-based remote monitoring capabilities for experiments.

Main components:
- SSHStateProviderServer: JSON-RPC server that wraps WorkspaceStateProvider
- SSHStateProviderClient: Client that connects via SSH and implements StateProvider interface
- RemoteFileSynchronizer: Rsync-based file synchronization

Usage:
    # On remote host (run via SSH):
    from experimaestro.scheduler.remote.server import SSHStateProviderServer
    server = SSHStateProviderServer(workspace_path)
    server.start()

    # On local host:
    from experimaestro.scheduler.remote.client import SSHStateProviderClient
    client = SSHStateProviderClient(host="server", remote_workspace="/path")
    client.connect()
    experiments = client.get_experiments()
"""

from experimaestro.scheduler.remote.server import SSHStateProviderServer
from experimaestro.scheduler.remote.client import SSHStateProviderClient
from experimaestro.scheduler.remote.sync import RemoteFileSynchronizer

__all__ = [
    "SSHStateProviderServer",
    "SSHStateProviderClient",
    "RemoteFileSynchronizer",
]
