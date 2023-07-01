import sys
import click

from .configuration import SlurmConfiguration, fill_nodes_configuration


@click.group()
def cli():
    pass


@click.option("--no-hosts", is_flag=True, help="Disable hosts")
@cli.command()
def convert(no_hosts):
    """Convert the ouptut of 'scontrol show node' into a YAML form compatible
    with launchers.yaml"""
    import yaml
    from experimaestro.launcherfinder import LauncherRegistry

    configuration = SlurmConfiguration(id="", partitions={})
    fill_nodes_configuration(sys.stdin, configuration)

    if no_hosts:
        for pid, partition in configuration.partitions.items():
            for node in partition.nodes:
                node.hosts = []
        configuration.use_hosts = False

    yaml.dump(configuration, sys.stdout, Dumper=LauncherRegistry.instance().Dumper)
