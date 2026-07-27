import click
from termcolor import cprint
from experimaestro.settings import get_settings
from dataclasses import asdict
import yaml

@click.group()
def workspaces():
    """Manage workspaces defined in settings.yaml"""
    pass

@workspaces.command(name="list")
@click.option("-v", "--verbose", is_flag=True, help="Display all parameters for each workspace")
def list_workspaces(verbose):
    """List workspaces configured in settings.yaml"""
    settings = get_settings()

    if not settings.workspaces:
        cprint("No workspaces configured in settings.yaml", "yellow")
        return

    cprint("Experimaestro workspaces:", "white", attrs=["bold"])
    for ws in settings.workspaces:
        tags = []
        if ws.is_remote:
            tags.append(f"remote: {ws.ssh.host}")

        tags_str = f" [{', '.join(tags)}]" if tags else ""
        cprint(f"- {ws.id}: {ws.path}{tags_str}", "cyan")

        if verbose:
            # Print a structured representation of all workspace parameters
            ws_dict = asdict(ws)
            # Remove redundant ID and path from the verbose dump as they are already printed
            ws_dict.pop("id", None)
            ws_dict.pop("path", None)
            # path is a Path object, which might not serialize cleanly with simple yaml dump in older PyYAML, but asdict() might keep it as Path.
            # We convert Path to str just to be safe.
            def sanitize(d):
                for k, v in list(d.items()):
                    if hasattr(v, '__fspath__'):
                        d[k] = str(v)
                    elif isinstance(v, dict):
                        sanitize(v)
                    elif isinstance(v, list):
                        for i, item in enumerate(v):
                            if hasattr(item, '__fspath__'):
                                v[i] = str(item)
                            elif isinstance(item, dict):
                                sanitize(item)
            sanitize(ws_dict)
            cprint(yaml.dump(ws_dict, sort_keys=False, default_flow_style=False), "white")
