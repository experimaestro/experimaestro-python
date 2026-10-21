from dataclasses import dataclass
from pathlib import Path
import platform
import sys
from typing import TYPE_CHECKING

import click
import jinja2

from experimaestro.scriptbuilder import PythonScriptBuilder
from . import Launcher

if TYPE_CHECKING:
    from rich.console import Console


class DirectLauncher(Launcher):
    """Launcher that runs tasks directly as local processes.

    This is the default launcher that executes tasks on the local machine
    without any job scheduler. Tasks are run as Python subprocesses.

    :param connector: The connector to use (defaults to LocalConnector)
    """

    def scriptbuilder(self):
        return PythonScriptBuilder()

    def launcher_info_code(self) -> str:
        """Returns empty string as local launcher has no time limits."""
        return ""

    def __str__(self):
        return f"DirectLauncher({self.connector})"

    @staticmethod
    def get_cli():
        """Returns the CLI group for direct launcher commands."""
        return direct_cli


# =============================================================================
# System detection utilities
# =============================================================================


@dataclass
class GPUInfo:
    """Information about a GPU."""

    name: str
    memory_bytes: int
    is_mps: bool = False  # Apple Metal Performance Shaders


@dataclass
class SystemInfo:
    """Detected system resources."""

    total_memory_bytes: int
    cpu_count: int
    gpus: list[GPUInfo]
    is_apple_silicon: bool

    @property
    def total_memory_gib(self) -> float:
        return self.total_memory_bytes / (1024**3)

    @property
    def has_mps(self) -> bool:
        return self.is_apple_silicon and any(g.is_mps for g in self.gpus)

    def format_memory(self, bytes_val: int) -> str:
        """Format bytes as human-readable string."""
        if bytes_val >= 1024**3:
            return f"{bytes_val / (1024**3):.1f} GiB"
        return f"{bytes_val / (1024**2):.0f} MiB"


def detect_system_info() -> SystemInfo:
    """Detect system resources (RAM, CPUs, GPUs)."""
    import psutil

    total_memory = psutil.virtual_memory().total
    cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1

    # Detect Apple Silicon
    is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"

    gpus: list[GPUInfo] = []

    # Try to detect NVIDIA GPUs via pynvml
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(GPUInfo(name=name, memory_bytes=mem_info.total))
        pynvml.nvmlShutdown()
    except Exception:
        pass

    # Check for Apple Silicon MPS
    if is_apple_silicon:
        try:
            import torch

            if torch.backends.mps.is_available():
                # MPS uses unified memory - report system memory as GPU memory
                gpus.append(
                    GPUInfo(
                        name="Apple MPS (unified memory)",
                        memory_bytes=total_memory,
                        is_mps=True,
                    )
                )
        except Exception:
            # Even without torch, Apple Silicon has MPS capability
            gpus.append(
                GPUInfo(
                    name="Apple MPS (unified memory)",
                    memory_bytes=total_memory,
                    is_mps=True,
                )
            )

    return SystemInfo(
        total_memory_bytes=total_memory,
        cpu_count=cpu_count,
        gpus=gpus,
        is_apple_silicon=is_apple_silicon,
    )


# =============================================================================
# Configuration data structures
# =============================================================================

# Default token unit size: 256 MiB
DEFAULT_TOKEN_UNIT_MIB = 256


@dataclass
class LauncherConfig:
    """Configuration for launcher generation."""

    # Memory token settings
    use_memory_tokens: bool = False
    memory_token_unit_mib: int = DEFAULT_TOKEN_UNIT_MIB
    total_memory_tokens: int = 0
    reserved_memory_gib: float = 2.0  # Reserve for OS

    # GPU settings
    use_gpu_tokens: bool = False
    gpu_mode: str = "exclusive"  # "exclusive" or "memory"
    gpu_token_unit_mib: int = 1024  # 1 GiB for GPU memory tokens
    total_gpu_tokens: int = 0
    gpu_count: int = 0

    # Apple Silicon MPS handling
    is_mps: bool = False
    mps_shared_memory: bool = True  # GPU memory comes from system RAM


# =============================================================================
# CLI commands
# =============================================================================


@click.group("direct")
def direct_cli():
    """Direct launcher commands"""
    pass


def _check_output_file(
    output: Path,
    force: bool,
    yes: bool,
    non_interactive: bool,
    console: "Console",
) -> bool:
    """Check if output file exists and handle overwrite confirmation."""
    from rich.prompt import Confirm

    if output.exists() and not (force or yes):
        console.print(f"\n[yellow]File already exists:[/yellow] {output}")
        if non_interactive:
            raise click.ClickException(
                f"File already exists: {output}. Pass --force (-f) or --yes (-y) to overwrite."
            )
        if not Confirm.ask("Overwrite?", default=False):
            console.print("[yellow]Aborted.[/yellow]")
            return False
    return True


def _display_system_info(console: "Console", sys_info: SystemInfo) -> None:
    """Display detected system resources table."""
    from rich.table import Table

    table = Table(title="System Resources", border_style="blue")
    table.add_column("Resource", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total RAM", sys_info.format_memory(sys_info.total_memory_bytes))
    table.add_row("CPU cores", str(sys_info.cpu_count))

    if sys_info.gpus:
        for i, gpu in enumerate(sys_info.gpus):
            mem_str = sys_info.format_memory(gpu.memory_bytes)
            if gpu.is_mps:
                mem_str += " [yellow](unified)[/yellow]"
            table.add_row(f"GPU {i}", f"{gpu.name} ({mem_str})")
    else:
        table.add_row("GPUs", "[dim]None detected[/dim]")

    if sys_info.is_apple_silicon:
        table.add_row("Platform", "[yellow]Apple Silicon (MPS)[/yellow]")

    console.print()
    console.print(table)
    console.print()


def _configure_memory_tokens(
    console: "Console",
    config: LauncherConfig,
    sys_info: SystemInfo,
    non_interactive: bool,
    memory_tokens: bool | None,
    reserved_memory: float | None,
    memory_token_unit: int | None,
) -> None:
    """Prompt and configure memory tokens on LauncherConfig."""
    from rich.panel import Panel
    from rich.prompt import Confirm, FloatPrompt, IntPrompt

    if not non_interactive:
        console.print(
            Panel(
                "[bold]Memory Tokens[/bold]\n\n"
                "Memory tokens prevent over-committing system RAM.\n"
                "Each token represents a unit of memory (default: 256 MiB).\n"
                "Tasks request tokens based on their memory needs.\n\n"
                "[dim]Example: 1 GiB requirement = 4 tokens (with 256 MiB unit)[/dim]",
                border_style="blue",
            )
        )

    if memory_tokens is not None:
        config.use_memory_tokens = memory_tokens
    elif non_interactive:
        config.use_memory_tokens = True
    else:
        config.use_memory_tokens = Confirm.ask("\nEnable memory tokens?", default=True)

    if config.use_memory_tokens:
        # Reserved memory for OS
        if reserved_memory is not None:
            config.reserved_memory_gib = reserved_memory
        elif non_interactive:
            config.reserved_memory_gib = 2.0
        else:
            config.reserved_memory_gib = FloatPrompt.ask(
                "  Reserved memory for OS (GiB)",
                default=2.0,
            )

        # Token unit size
        if memory_token_unit is not None:
            config.memory_token_unit_mib = memory_token_unit
        elif non_interactive:
            config.memory_token_unit_mib = DEFAULT_TOKEN_UNIT_MIB
        else:
            config.memory_token_unit_mib = IntPrompt.ask(
                "  Token unit size (MiB)",
                default=DEFAULT_TOKEN_UNIT_MIB,
            )

        # Calculate available tokens
        available_bytes = sys_info.total_memory_bytes - int(
            config.reserved_memory_gib * (1024**3)
        )
        config.total_memory_tokens = max(
            1, available_bytes // (config.memory_token_unit_mib * (1024**2))
        )

        console.print(
            f"  [green]→ {config.total_memory_tokens} memory tokens available[/green]"
        )


def _configure_gpu_tokens(
    console: "Console",
    config: LauncherConfig,
    sys_info: SystemInfo,
    non_interactive: bool,
    gpu_tokens: bool | None,
    gpu_mode: str | None,
    gpu_token_unit: int | None,
) -> None:
    """Prompt and configure GPU tokens on LauncherConfig."""
    from rich.panel import Panel
    from rich.prompt import Confirm, IntPrompt, Prompt

    if not sys_info.gpus:
        return

    if not non_interactive:
        console.print()
        console.print(
            Panel(
                "[bold]GPU Tokens[/bold]\n\n"
                "GPU tokens control concurrent GPU task execution.\n\n"
                "[cyan]Exclusive mode:[/cyan] 1 token = 1 GPU (simple, recommended)\n"
                "[cyan]Memory mode:[/cyan] Tokens based on GPU memory units\n"
                + (
                    "\n[yellow]Note: Apple MPS uses unified memory - GPU memory\n"
                    "requests will reduce available CPU memory tokens.[/yellow]"
                    if sys_info.has_mps
                    else ""
                ),
                border_style="blue",
            )
        )

    if gpu_tokens is not None:
        config.use_gpu_tokens = gpu_tokens
    elif non_interactive:
        config.use_gpu_tokens = True
    else:
        config.use_gpu_tokens = Confirm.ask("\nEnable GPU tokens?", default=True)

    if not config.use_gpu_tokens:
        return

    config.gpu_count = len(sys_info.gpus)

    # GPU mode selection
    if sys_info.has_mps:
        console.print(
            "  [yellow]MPS detected: using memory-based tokens "
            "(shared with CPU memory)[/yellow]"
        )
        config.gpu_mode = "memory"
        config.mps_shared_memory = True
    elif gpu_mode is not None:
        config.gpu_mode = gpu_mode
    elif non_interactive:
        config.gpu_mode = "exclusive"
    else:
        config.gpu_mode = Prompt.ask(
            "  GPU token mode",
            choices=["exclusive", "memory"],
            default="exclusive",
        )

    if config.gpu_mode == "exclusive":
        config.total_gpu_tokens = config.gpu_count
        console.print(
            f"  [green]→ {config.total_gpu_tokens} GPU tokens (1 per GPU)[/green]"
        )
    else:
        if gpu_token_unit is not None:
            config.gpu_token_unit_mib = gpu_token_unit
        elif non_interactive:
            config.gpu_token_unit_mib = 1024
        else:
            config.gpu_token_unit_mib = IntPrompt.ask(
                "  GPU token unit size (MiB)",
                default=1024,
            )
        # Calculate GPU tokens from first GPU (or total for MPS)
        if sys_info.has_mps:
            # For MPS, GPU memory is same as system memory
            # Don't double-count - just note that GPU uses memory tokens
            console.print("  [yellow]→ MPS: GPU tasks will use memory tokens[/yellow]")
            config.total_gpu_tokens = 0  # Use memory tokens instead
        else:
            total_gpu_mem = sum(g.memory_bytes for g in sys_info.gpus)
            config.total_gpu_tokens = max(
                1, total_gpu_mem // (config.gpu_token_unit_mib * (1024**2))
            )
            console.print(
                f"  [green]→ {config.total_gpu_tokens} GPU memory tokens[/green]"
            )


def _display_summary(console: "Console", output: Path, config: LauncherConfig) -> None:
    """Display configuration summary table."""
    from rich.table import Table

    console.print()
    summary = Table(title="Configuration Summary", border_style="green")
    summary.add_column("Setting", style="cyan")
    summary.add_column("Value", style="green")

    summary.add_row("Output file", str(output))

    if config.use_memory_tokens:
        summary.add_row(
            "Memory tokens",
            f"{config.total_memory_tokens} ({config.memory_token_unit_mib} MiB each)",
        )
    else:
        summary.add_row("Memory tokens", "[dim]Disabled[/dim]")

    if config.use_gpu_tokens and config.total_gpu_tokens > 0:
        if config.gpu_mode == "exclusive":
            summary.add_row("GPU tokens", f"{config.total_gpu_tokens} (exclusive)")
        else:
            summary.add_row(
                "GPU tokens",
                f"{config.total_gpu_tokens} ({config.gpu_token_unit_mib} MiB each)",
            )
    elif config.use_gpu_tokens and config.is_mps:
        summary.add_row("GPU tokens", "[yellow]Using memory tokens (MPS)[/yellow]")
    else:
        summary.add_row("GPU tokens", "[dim]Disabled[/dim]")

    console.print(summary)
    console.print()


def _write_outputs(
    output: Path,
    config: LauncherConfig,
    sys_info: SystemInfo,
    force: bool,
    yes: bool,
    non_interactive: bool,
    console: "Console",
) -> None:
    """Prompt confirmation (if interactive) and write output files."""
    from rich.prompt import Confirm

    # Confirm generation
    if not non_interactive and not Confirm.ask("Generate configuration?", default=True):
        console.print("[yellow]Aborted.[/yellow]")
        return

    # Generate files
    content = _generate_launchers_py(config, sys_info)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    console.print(f"\n[green]✓[/green] Generated [bold]{output}[/bold]")

    # Generate tokens.yaml
    if config.use_memory_tokens or (
        config.use_gpu_tokens and config.total_gpu_tokens > 0
    ):
        tokens_yaml_path = output.parent / "tokens.yaml"
        write_tokens = force or yes or non_interactive or not tokens_yaml_path.exists()
        if not write_tokens:
            write_tokens = Confirm.ask(f"Generate {tokens_yaml_path}?", default=True)

        if write_tokens:
            tokens_content = _generate_tokens_yaml(config)
            tokens_yaml_path.write_text(tokens_content)
            console.print(f"[green]✓[/green] Generated [bold]{tokens_yaml_path}[/bold]")

    console.print(
        "\n[bold green]Done![/bold green] Your launcher configuration is ready.\n"
    )


@direct_cli.command("generate")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for launchers.py (default: ~/.config/experimaestro/launchers.py)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing file without prompting",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Non-interactive mode: accept defaults without prompting",
)
@click.option(
    "--memory-tokens/--no-memory-tokens",
    default=None,
    help="Enable or disable CPU memory tokens (default: enabled)",
)
@click.option(
    "--reserved-memory",
    type=float,
    default=None,
    help="Reserved memory for OS in GiB (default: 2.0)",
)
@click.option(
    "--memory-token-unit",
    type=int,
    default=None,
    help=f"Memory token unit size in MiB (default: {DEFAULT_TOKEN_UNIT_MIB})",
)
@click.option(
    "--gpu-tokens/--no-gpu-tokens",
    default=None,
    help="Enable or disable GPU tokens when GPUs are detected (default: enabled)",
)
@click.option(
    "--gpu-mode",
    type=click.Choice(["exclusive", "memory"]),
    default=None,
    help="GPU token mode: 'exclusive' (1 token/GPU) or 'memory' (default: exclusive for CUDA, memory for MPS)",
)
@click.option(
    "--gpu-token-unit",
    type=int,
    default=None,
    help="GPU memory token unit size in MiB when using memory mode (default: 1024)",
)
def generate(
    output: Path | None,
    force: bool,
    yes: bool,
    memory_tokens: bool | None,
    reserved_memory: float | None,
    memory_token_unit: int | None,
    gpu_tokens: bool | None,
    gpu_mode: str | None,
    gpu_token_unit: int | None,
):
    """Generate a launchers.py file for direct (local) execution.

    This command detects system resources and configures memory-based tokens
    for CPU and GPU resource management. Use `-y / --yes` for non-interactive
    generation with sensible defaults.

    Token system:
      - Memory tokens: 1 token = configurable unit (default 256 MiB)
      - Tasks request tokens based on their memory requirements
      - Example: 1 GiB requirement = 4 tokens (with 256 MiB unit)

    Apple Silicon (MPS):
      - GPU uses unified memory (shared with CPU)
      - GPU memory requests reduce available CPU memory tokens

    Examples:
        experimaestro launchers direct generate -y
        experimaestro launchers direct generate
        experimaestro launchers direct generate -o ./launchers.py -y
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    non_interactive = yes or not sys.stdin.isatty()

    # Default output path
    if output is None:
        output = Path("~/.config/experimaestro/launchers.py").expanduser()
    else:
        output = output.expanduser()

    # Header
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Direct Launcher Configuration[/bold cyan]\n\n"
            "This wizard configures resource tokens for local task execution.\n"
            "Tokens ensure tasks don't exceed available memory.",
            border_style="cyan",
        )
    )

    # Check if file exists
    if not _check_output_file(output, force, yes, non_interactive, console):
        return

    # Detect system resources
    console.print("\n[cyan]Detecting system resources...[/cyan]")
    sys_info = detect_system_info()
    _display_system_info(console, sys_info)

    # Initialize config
    config = LauncherConfig()
    config.is_mps = sys_info.has_mps

    # Configure memory tokens
    _configure_memory_tokens(
        console,
        config,
        sys_info,
        non_interactive,
        memory_tokens,
        reserved_memory,
        memory_token_unit,
    )

    # Configure GPU tokens
    _configure_gpu_tokens(
        console,
        config,
        sys_info,
        non_interactive,
        gpu_tokens,
        gpu_mode,
        gpu_token_unit,
    )

    # Display configuration summary
    _display_summary(console, output, config)

    # Write generated files
    _write_outputs(
        output,
        config,
        sys_info,
        force,
        yes,
        non_interactive,
        console,
    )


# Template environment for code generation
_template_env = jinja2.Environment(
    loader=jinja2.PackageLoader("experimaestro.launchers", "templates"),
    keep_trailing_newline=True,
)


def _generate_launchers_py(config: LauncherConfig, sys_info: SystemInfo) -> str:
    """Generate launchers.py with memory-based token system using Jinja2 template."""
    available_mem = int(
        sys_info.total_memory_bytes - config.reserved_memory_gib * (1024**3)
    )

    template = _template_env.get_template("launchers.py.j2")
    return template.render(
        # Config values
        use_memory_tokens=config.use_memory_tokens,
        use_gpu_tokens=config.use_gpu_tokens,
        total_memory_tokens=config.total_memory_tokens,
        total_gpu_tokens=config.total_gpu_tokens,
        memory_token_unit_mib=config.memory_token_unit_mib,
        gpu_token_unit_mib=config.gpu_token_unit_mib,
        gpu_mode=config.gpu_mode,
        is_mps=config.is_mps,
        mps_shared_memory=config.mps_shared_memory,
        # System info
        available_memory=available_mem,
        cpu_count=sys_info.cpu_count,
        gpus=sys_info.gpus,
        has_cuda_gpus=bool(sys_info.gpus) and not config.is_mps,
    )


def _generate_tokens_yaml(config: LauncherConfig) -> str:
    """Generate tokens.yaml for memory-based token system using Jinja2 template."""
    template = _template_env.get_template("tokens.yaml.j2")
    return template.render(
        use_memory_tokens=config.use_memory_tokens,
        use_gpu_tokens=config.use_gpu_tokens,
        total_memory_tokens=config.total_memory_tokens,
        total_gpu_tokens=config.total_gpu_tokens,
        memory_token_unit_mib=config.memory_token_unit_mib,
        gpu_token_unit_mib=config.gpu_token_unit_mib,
        gpu_mode=config.gpu_mode,
    )
