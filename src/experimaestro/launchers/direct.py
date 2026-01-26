from dataclasses import dataclass
from pathlib import Path
import platform
import sys

import click
import jinja2

from experimaestro.scriptbuilder import PythonScriptBuilder
from . import Launcher


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
def generate(output: Path | None, force: bool):
    """Generate a launchers.py file for direct (local) execution.

    This command detects system resources and interactively configures
    memory-based tokens for CPU and GPU resource management.

    Token system:
      - Memory tokens: 1 token = configurable unit (default 256 MiB)
      - Tasks request tokens based on their memory requirements
      - Example: 1 GiB requirement = 4 tokens (with 256 MiB unit)

    Apple Silicon (MPS):
      - GPU uses unified memory (shared with CPU)
      - GPU memory requests reduce available CPU memory tokens

    Examples:
        experimaestro launchers direct generate
        experimaestro launchers direct generate -o ./launchers.py
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
    from rich.table import Table

    console = Console()

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
    if output.exists() and not force:
        console.print(f"\n[yellow]File already exists:[/yellow] {output}")
        if not Confirm.ask("Overwrite?", default=False):
            console.print("[yellow]Aborted.[/yellow]")
            return

    # Detect system resources
    console.print("\n[cyan]Detecting system resources...[/cyan]")
    sys_info = detect_system_info()

    # Display system info
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

    # Initialize config
    config = LauncherConfig()
    config.is_mps = sys_info.has_mps

    # === Memory tokens configuration ===
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

    config.use_memory_tokens = Confirm.ask("\nEnable memory tokens?", default=True)

    if config.use_memory_tokens:
        # Reserved memory for OS
        config.reserved_memory_gib = FloatPrompt.ask(
            "  Reserved memory for OS (GiB)",
            default=2.0,
        )

        # Token unit size
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

    # === GPU tokens configuration ===
    if sys_info.gpus:
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

        config.use_gpu_tokens = Confirm.ask("\nEnable GPU tokens?", default=True)

        if config.use_gpu_tokens:
            config.gpu_count = len(sys_info.gpus)

            # GPU mode selection
            if sys_info.has_mps:
                console.print(
                    "  [yellow]MPS detected: using memory-based tokens "
                    "(shared with CPU memory)[/yellow]"
                )
                config.gpu_mode = "memory"
                config.mps_shared_memory = True
            else:
                config.gpu_mode = Prompt.ask(
                    "  GPU token mode",
                    choices=["exclusive", "memory"],
                    default="exclusive",
                )

            if config.gpu_mode == "exclusive":
                config.total_gpu_tokens = config.gpu_count
                console.print(
                    f"  [green]→ {config.total_gpu_tokens} GPU tokens "
                    f"(1 per GPU)[/green]"
                )
            else:
                config.gpu_token_unit_mib = IntPrompt.ask(
                    "  GPU token unit size (MiB)",
                    default=1024,
                )
                # Calculate GPU tokens from first GPU (or total for MPS)
                if sys_info.has_mps:
                    # For MPS, GPU memory is same as system memory
                    # Don't double-count - just note that GPU uses memory tokens
                    console.print(
                        "  [yellow]→ MPS: GPU tasks will use memory tokens[/yellow]"
                    )
                    config.total_gpu_tokens = 0  # Use memory tokens instead
                else:
                    total_gpu_mem = sum(g.memory_bytes for g in sys_info.gpus)
                    config.total_gpu_tokens = max(
                        1, total_gpu_mem // (config.gpu_token_unit_mib * (1024**2))
                    )
                    console.print(
                        f"  [green]→ {config.total_gpu_tokens} GPU memory tokens[/green]"
                    )

    # === Summary ===
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

    # Confirm generation
    if not Confirm.ask("Generate configuration?", default=True):
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
        write_tokens = force or not tokens_yaml_path.exists()
        if not write_tokens:
            write_tokens = Confirm.ask(f"Generate {tokens_yaml_path}?", default=True)

        if write_tokens:
            tokens_content = _generate_tokens_yaml(config)
            tokens_yaml_path.write_text(tokens_content)
            console.print(f"[green]✓[/green] Generated [bold]{tokens_yaml_path}[/bold]")

    console.print(
        "\n[bold green]Done![/bold green] Your launcher configuration is ready.\n"
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
