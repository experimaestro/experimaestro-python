"""Simplified CLI commands for managing and viewing progress files"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import click
from termcolor import colored

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from experimaestro.progress import ProgressEntry, ProgressFileReader
from experimaestro.settings import find_workspace
from . import cli


@click.option("--workspace", default="", help="Experimaestro workspace")
@click.option("--workdir", type=Path, default=None)
@cli.group()
@click.pass_context
def progress(
    ctx,
    workdir: Optional[Path],
    workspace: Optional[str],
):
    """Progress tracking commands"""
    ctx.obj.workspace = find_workspace(workdir=workdir, workspace=workspace)


def format_timestamp(timestamp: float) -> str:
    """Format timestamp for display"""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


@click.argument("jobid", type=str)
@progress.command()
@click.pass_context
def show(ctx, jobid: str):
    """Show current progress state (default command)

    JOBID format: task_name/task_hash
    """
    try:
        task_name, task_hash = jobid.split("/")
    except ValueError:
        raise click.ClickException("JOBID must be in format task_name/task_hash")

    workspace = ctx.obj.workspace
    task_path = workspace.path / "jobs" / task_name / task_hash

    if not task_path.exists():
        raise click.ClickException(f"Job directory not found: {task_path}")

    reader = ProgressFileReader(task_path)
    current_progress = reader.get_current_progress()

    if not current_progress:
        click.echo("No progress information available")
        return

    # Filter out EOJ markers
    current_progress = {k: v for k, v in current_progress.items() if k != -1}

    if not current_progress:
        click.echo("No progress information available")
        return

    click.echo(f"Progress for job {jobid}")
    click.echo("=" * 80)

    # Show simple text-based progress for each level
    for level in sorted(current_progress.keys()):
        entry = current_progress[level]
        indent = "  " * level
        progress_pct = f"{entry.progress * 100:5.1f}%"
        desc = entry.desc or f"Level {level}"
        timestamp = format_timestamp(entry.timestamp)

        color = "green" if entry.progress >= 1.0 else "yellow"
        click.echo(colored(f"{indent}L{level}: {progress_pct} - {desc}", color))
        click.echo(colored(f"{indent}     Last updated: {timestamp}", "cyan"))


def create_progress_bar(
    level: int,
    desc: str,
    progress: float = 0.0,
) -> tqdm:
    """Create a properly aligned progress bar like dashboard style"""
    if level > 0:
        indent = "   " * (level - 1) + "└─ "
    else:
        indent = ""
    label = f"{indent}L{level}"

    colors = ["blue", "yellow", "magenta", "cyan", "white"]
    bar_color = colors[level % len(colors)]

    unit = desc[:50] if desc else f"Level {level}"
    ncols = 100
    wbar = 50

    return tqdm(
        total=100,
        desc=label,
        position=level,
        leave=True,
        bar_format=f"{{desc}}: {{percentage:3.0f}}%|{{bar:{wbar - len(indent)}}}| {{unit}}",  # noqa: F541
        ncols=ncols,  # Adjust width based on level
        unit=unit,
        colour=bar_color,
        initial=progress * 100,
    )


def _update_progress_display(
    reader: ProgressFileReader, progress_bars: Dict[int, tqdm]
) -> bool:
    """Update the tqdm progress bars in dashboard style"""
    current_state: Dict[int, ProgressEntry] = {
        k: v for k, v in reader.get_current_state().items() if k != -1
    }

    if not current_state:
        click.echo("No progress information available yet...")
        return False

    # Update existing bars and create new ones
    for _level, entry in current_state.items():
        progress_val = entry.progress * 100
        desc = entry.desc or f"Level {entry.level}"

        if entry.level not in progress_bars:
            progress_bars[entry.level] = create_progress_bar(
                entry.level, desc, progress_val
            )

        bar = progress_bars[entry.level]
        bar.unit = desc[:50]
        bar.n = progress_val

        bar.refresh()

    # Remove bars for levels that no longer exist
    levels_to_remove = set(progress_bars.keys()) - set(current_state.keys())
    for level in levels_to_remove:
        progress_bars[level].close()
        del progress_bars[level]

    return True


@click.argument("jobid", type=str)
@click.option("--refresh-rate", "-r", default=0.5, help="Refresh rate in seconds")
@progress.command()
@click.pass_context
def live(ctx, jobid: str, refresh_rate: float):
    """Show live progress with tqdm-style bars

    JOBID format: task_name/task_hash
    """
    if not TQDM_AVAILABLE:
        click.echo("tqdm is not available. Install with: pip install tqdm")
        click.echo("Falling back to basic display...")
        ctx.invoke(show, jobid=jobid)
        return

    try:
        task_name, task_hash = jobid.split("/")
    except ValueError:
        raise click.ClickException("JOBID must be in format task_name/task_hash")

    workspace = ctx.obj.workspace
    task_path = workspace.path / "jobs" / task_name / task_hash

    if not task_path.exists():
        raise click.ClickException(f"Job directory not found: {task_path}")

    reader = ProgressFileReader(task_path)
    progress_bars: Dict[int, tqdm] = {}

    def cleanup_bars():
        """Clean up all progress bars"""
        for bar in progress_bars.values():
            bar.close()
        progress_bars.clear()

    click.echo(f"Live progress for job {jobid}")
    click.echo("Press Ctrl+C to stop")
    click.echo("=" * 80)

    try:
        if not _update_progress_display(reader, progress_bars):
            click.echo("No progress information available yet...")

        while True:
            time.sleep(refresh_rate)

            if not _update_progress_display(reader, progress_bars):
                # Check if job is complete
                if reader.is_done():
                    click.echo("\nJob completed!")
                    break

            # Check if all progress bars are at 100%
            if progress_bars and all(bar.n >= 100 for bar in progress_bars.values()):
                cleanup_bars()
                click.echo("\nAll progress completed!")
                break

    except KeyboardInterrupt:
        click.echo("\nStopped monitoring progress")
    finally:
        cleanup_bars()


@progress.command(name="list")
@click.pass_context
def list_jobs(ctx):
    """List all jobs with progress information"""
    ws = ctx.obj.workspace
    jobs_path = ws.path / "jobs"

    if not jobs_path.exists():
        click.echo("No jobs directory found")
        return

    for task_dir in jobs_path.iterdir():
        if not task_dir.is_dir():
            continue

        for job_dir in task_dir.iterdir():
            if not job_dir.is_dir():
                continue

            progress_dir = job_dir / ".experimaestro"
            if not progress_dir.exists():
                continue

            # Check if there are progress files
            progress_files = list(progress_dir.glob("progress-*.jsonl"))
            if not progress_files:
                continue

            job_id = f"{task_dir.name}/{job_dir.name}"
            reader = ProgressFileReader(job_dir)
            current_state = reader.get_current_state()

            # if current_progress:
            if current_state:
                # Get overall progress (level 0)
                level_0 = current_state.get(0)
                if level_0:
                    color = "green" if level_0.progress >= 1.0 else "yellow"
                    desc = f"{level_0.desc}" if level_0.desc else ""
                    progress_pct = f"{level_0.progress * 100:5.1f}%"
                    click.echo(colored(f"{job_id:50} - {progress_pct} - {desc}", color))

                else:
                    click.echo(f"{job_id:50} No level 0 progress")
            else:
                click.echo(f"{job_id:50} No progress data")
