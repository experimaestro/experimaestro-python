"""Textual TUI for SLURM launcher configuration."""

from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
    Tree,
)
from textual_fspicker import FileOpen, FileSave, Filters

from .cli import (
    SlurmCommandCache,
    detect_cluster_info,
    get_default_cache_file,
)
from .models import (
    SlurmConfig,
    ConfiguredPartition,
)


def field_widget(
    value: str | None,
    is_override: bool = False,
    default: str | None = None,
    is_inferred: bool = False,
) -> Text:
    """Create a styled Text widget for a field value."""
    if is_inferred and value:
        return Text(value, style="green")
    elif is_override and value:
        return Text(value, style="cyan")
    elif value:
        return Text(value)
    elif default:
        return Text(default)
    else:
        return Text("-", style="dim")


def list_field_widget(
    items: list[str],
    max_display: int = 2,
    is_inferred: bool = False,
    is_override: bool = False,
) -> Text:
    """Create a styled Text widget for a list field.

    Args:
        items: List of items to display
        max_display: Maximum items to show before truncating
        is_inferred: If True, use green style to indicate inferred from features
        is_override: If True, use cyan style to indicate user override
    """
    if not items:
        return Text("-", style="dim")
    display = ", ".join(items[:max_display])
    if len(items) > max_display:
        display += f" (+{len(items) - max_display})"
    # Green for inferred, cyan for explicit override, default for cluster/none
    if is_inferred:
        style = "green"
    elif is_override:
        style = "cyan"
    else:
        style = ""
    return Text(display, style=style)


class InputScreen(Screen):
    """Screen for text input."""

    CSS_PATH = "slurm_config.tcss"

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        title: str,
        label: str,
        initial_value: str = "",
        callback_key: str = "",
    ):
        super().__init__()
        self.title = title
        self.label = label
        self.initial_value = initial_value
        self.callback_key = callback_key

    def compose(self) -> ComposeResult:
        with Container(id="input-dialog"):
            yield Label(f"[bold]{self.title}[/bold]")
            yield Label(self.label)
            yield Input(value=self.initial_value, id="input-field")
            with Horizontal(classes="button-row"):
                yield Button("OK", id="ok-button", variant="primary")
                yield Button("Cancel", id="cancel-button")
                yield Button("Clear", id="clear-button", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok-button":
            input_field = self.query_one("#input-field", Input)
            self.dismiss((self.callback_key, input_field.value))
        elif event.button.id == "clear-button":
            self.dismiss((self.callback_key, ""))
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss((self.callback_key, event.value))


class SelectionScreen(Screen):
    """Screen for multi-selection from a list."""

    CSS_PATH = "slurm_config.tcss"

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("space", "toggle", "Toggle"),
    ]

    def __init__(
        self,
        title: str,
        items: list[str],
        selected: list[str],
        callback_key: str = "",
    ):
        super().__init__()
        self.title = title
        self.items = items
        self.selected = set(selected)
        self.callback_key = callback_key

    def compose(self) -> ComposeResult:
        with Container(id="selection-dialog"):
            yield Label(f"[bold]{self.title}[/bold]")
            yield Label("Space to toggle, Enter to confirm")
            table = DataTable(id="selection-table", cursor_type="row")
            table.add_columns("✓", "Item")
            for item in sorted(self.items):
                checked = "✓" if item in self.selected else ""
                table.add_row(checked, item, key=item)
            yield table
            with Horizontal(classes="button-row"):
                yield Button("OK", id="ok-button", variant="primary")
                yield Button("Cancel", id="cancel-button")
                yield Button("Clear All", id="clear-button", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok-button":
            self.dismiss((self.callback_key, list(self.selected)))
        elif event.button.id == "clear-button":
            self.selected.clear()
            self._refresh_table()
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_toggle(self) -> None:
        self._toggle_current()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._toggle_current()

    def _toggle_current(self) -> None:
        table = self.query_one("#selection-table", DataTable)
        if table.cursor_row is None:
            return
        row_key = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key
        item = str(row_key.value) if row_key.value else None
        if item:
            if item in self.selected:
                self.selected.discard(item)
            else:
                self.selected.add(item)
            checked = "✓" if item in self.selected else ""
            col_key = list(table.columns.keys())[0]
            table.update_cell(item, col_key, checked)

    def _refresh_table(self) -> None:
        table = self.query_one("#selection-table", DataTable)
        col_key = list(table.columns.keys())[0]
        for item in self.items:
            checked = "✓" if item in self.selected else ""
            table.update_cell(item, col_key, checked)


class PartitionsTab(TabPane):
    """Tab for configuring partitions."""

    BINDINGS = [
        Binding("t", "edit_gpu_type", "GPU Type"),
        Binding("m", "edit_gpu_mem", "GPU Mem"),
        Binding("p", "edit_priority", "Priority"),
        Binding("o", "edit_qos", "QoS"),
        Binding("a", "edit_accounts", "Accounts"),
        Binding("e", "toggle_enabled", "Toggle"),
        Binding("delete", "reset_field", "Reset"),
        Binding("backspace", "reset_field", "Reset", show=False),
    ]

    def __init__(
        self,
        config: SlurmConfig,
        new_partitions: set[str] | None = None,
    ):
        super().__init__("Partitions", id="partitions-tab")
        self.config = config
        self.new_partitions = new_partitions or set()
        # Column keys - set in compose()
        self.col_keys: dict[str, any] = {}

    def compose(self) -> ComposeResult:
        help_text = (
            "Configure partition settings. "
            "[cyan]Cyan = override[/cyan], [green]Green = inferred from features[/green]."
        )
        if self.new_partitions:
            help_text += " [yellow]★ = new[/yellow]"
        yield Static(help_text, classes="help-text")

        table = DataTable(id="partitions-table", cursor_type="cell")

        # Build column list dynamically based on available data
        col_names = [
            "",
            "Partition",
            "Features",
            "Cores",
            "Memory",
            "GPUs",
            "GPU Type",
            "GPU Mem",
            "Priority",
        ]
        self.has_qos = bool(self.config.qos)
        self.has_accounts = bool(self.config.accounts)
        if self.has_qos:
            col_names.append("QoS")
        if self.has_accounts:
            col_names.append("Accounts")
        col_names.append("Enabled")

        keys = table.add_columns(*col_names)
        self.col_keys = dict(zip(col_names, keys))

        for name, partition in sorted(self.config.partitions.items()):
            if not partition._cluster.available:
                continue

            new_marker = (
                Text("★", style="yellow") if name in self.new_partitions else ""
            )
            gpu_mem = (
                f"{partition.effective_gpu_memory_gb} GB"
                if partition.effective_gpu_memory_gb > 0
                else None
            )
            features_str = ", ".join(partition.features[:2])
            if len(partition.features) > 2:
                features_str += f" (+{len(partition.features) - 2})"

            row_data = [
                new_marker,
                name,
                Text(features_str, style="dim")
                if features_str
                else Text("-", style="dim"),
                field_widget(
                    str(partition.effective_cores),
                    partition.cores.is_overridden,
                    str(partition.cpus_per_node),
                    is_inferred=partition.cores_is_inferred,
                ),
                field_widget(
                    f"{partition.effective_memory_gb} GB",
                    partition.memory_gb.is_overridden,
                    f"{partition.memory_mb // 1024} GB",
                    is_inferred=partition.memory_is_inferred,
                ),
                str(partition.gpus_per_node) if partition.gpus_per_node > 0 else "-",
                field_widget(
                    partition.effective_gpu_type,
                    partition.gpu_type.is_overridden,
                    partition.gpu_type.cluster_value,
                    is_inferred=partition.gpu_type_is_inferred,
                ),
                field_widget(
                    gpu_mem,
                    is_override=partition.gpu_memory_gb.is_overridden,
                    is_inferred=partition.gpu_memory_is_inferred,
                ),
                field_widget(
                    str(partition.priority.value), partition.priority.is_overridden
                ),
            ]
            if self.has_qos:
                row_data.append(
                    list_field_widget(
                        partition.effective_qos,
                        is_inferred=partition.qos_is_inferred,
                        is_override=partition.allowed_qos.is_overridden,
                    )
                )
            if self.has_accounts:
                row_data.append(
                    list_field_widget(
                        partition.effective_accounts,
                        is_inferred=partition.accounts_is_inferred,
                        is_override=partition.allowed_accounts.is_overridden,
                    )
                )
            row_data.append("✓" if partition.enabled.value else "✗")

            table.add_row(*row_data, key=name)

        yield table

    def refresh_table(self) -> None:
        """Refresh all cells in the partitions table."""
        table = self.table
        for name, partition in self.config.partitions.items():
            if not partition._cluster.available:
                continue
            # Update Cores column
            table.update_cell(
                name,
                self.col_keys["Cores"],
                field_widget(
                    str(partition.effective_cores),
                    partition.cores.is_overridden,
                    str(partition.cpus_per_node),
                    is_inferred=partition.cores_is_inferred,
                ),
            )
            # Update Memory column
            table.update_cell(
                name,
                self.col_keys["Memory"],
                field_widget(
                    f"{partition.effective_memory_gb} GB",
                    partition.memory_gb.is_overridden,
                    f"{partition.memory_mb // 1024} GB",
                    is_inferred=partition.memory_is_inferred,
                ),
            )
            # Update GPU Type column
            table.update_cell(
                name,
                self.col_keys["GPU Type"],
                field_widget(
                    partition.effective_gpu_type,
                    partition.gpu_type.is_overridden,
                    partition.gpu_type.cluster_value,
                    is_inferred=partition.gpu_type_is_inferred,
                ),
            )
            # Update GPU Mem column
            gpu_mem = (
                f"{partition.effective_gpu_memory_gb} GB"
                if partition.effective_gpu_memory_gb > 0
                else None
            )
            table.update_cell(
                name,
                self.col_keys["GPU Mem"],
                field_widget(
                    gpu_mem,
                    is_override=partition.gpu_memory_gb.is_overridden,
                    is_inferred=partition.gpu_memory_is_inferred,
                ),
            )
            # Update QoS column (if present)
            if self.has_qos:
                table.update_cell(
                    name,
                    self.col_keys["QoS"],
                    list_field_widget(
                        partition.effective_qos,
                        is_inferred=partition.qos_is_inferred,
                        is_override=partition.allowed_qos.is_overridden,
                    ),
                )
            # Update Accounts column (if present)
            if self.has_accounts:
                table.update_cell(
                    name,
                    self.col_keys["Accounts"],
                    list_field_widget(
                        partition.effective_accounts,
                        is_inferred=partition.accounts_is_inferred,
                        is_override=partition.allowed_accounts.is_overridden,
                    ),
                )

    @on(DataTable.CellSelected)
    def on_cell_selected(self, event: DataTable.CellSelected) -> None:
        """Handle Enter key on cell."""
        col_key = event.cell_key.column_key
        # Map column keys to editor actions
        editors = {
            self.col_keys["GPU Type"]: self.action_edit_gpu_type,
            self.col_keys["GPU Mem"]: self.action_edit_gpu_mem,
            self.col_keys["Priority"]: self.action_edit_priority,
            self.col_keys["Enabled"]: self.action_toggle_enabled,
        }
        # Add QoS/Accounts only if columns exist
        if self.has_qos:
            editors[self.col_keys["QoS"]] = self.action_edit_qos
        if self.has_accounts:
            editors[self.col_keys["Accounts"]] = self.action_edit_accounts
        if col_key in editors:
            editors[col_key]()

    @on(DataTable.CellHighlighted)
    def on_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        """Show full cell value in status bar when highlighted."""
        col_key = event.cell_key.column_key
        row_key = event.cell_key.row_key
        name = str(row_key.value) if row_key.value else None
        if not name:
            self.app.set_status("")
            return

        partition = self.config.partitions.get(name)
        if not partition:
            self.app.set_status("")
            return

        # Show full values for columns that may be truncated
        full_value = None
        if col_key == self.col_keys.get("Features"):
            full_value = ", ".join(partition.features) if partition.features else None
        elif col_key == self.col_keys.get("QoS"):
            full_value = (
                ", ".join(partition.effective_qos) if partition.effective_qos else None
            )
        elif col_key == self.col_keys.get("Accounts"):
            full_value = (
                ", ".join(partition.effective_accounts)
                if partition.effective_accounts
                else None
            )

        if full_value:
            self.app.set_status(f"{name}: {full_value}")
        else:
            self.app.set_status("")

    @property
    def table(self) -> DataTable:
        return self.query_one("#partitions-table", DataTable)

    def _get_selected_partition(
        self,
    ) -> tuple[str, ConfiguredPartition] | tuple[None, None]:
        """Get the currently selected partition name and info."""
        table = self.table
        if table.cursor_row is None or table.row_count == 0:
            self.app.set_status("No partition selected")
            return None, None

        row_key = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key
        name = str(row_key.value) if row_key.value else None
        if not name:
            self.app.set_status("No partition selected")
            return None, None

        partition = self.config.partitions.get(name)
        if not partition:
            self.app.set_status(f"Partition not found: {name}")
            return None, None

        return name, partition

    def action_edit_gpu_type(self) -> None:
        """Edit GPU type for selected partition."""
        name, partition = self._get_selected_partition()
        if not name:
            return

        table = self.table

        def handle_result(result: tuple[str, str] | None) -> None:
            if result is None:
                return
            callback_key, value = result
            if callback_key != name:
                return

            if value:
                partition.gpu_type.set(value)
            else:
                partition.gpu_type.reset()
            self.app.set_status(f"{'Set' if value else 'Cleared'} GPU type for {name}")
            self.app._auto_save()

            display = field_widget(
                partition.gpu_type.value,
                partition.gpu_type.is_overridden,
                partition.gpu_type.cluster_value,
            )
            table.update_cell(name, self.col_keys["GPU Type"], display)

        self.app.push_screen(
            InputScreen(
                title=f"GPU Type for '{name}'",
                label="Enter GPU type (e.g., v100, a100, h100):",
                initial_value=partition.gpu_type.override_value or "",
                callback_key=name,
            ),
            handle_result,
        )

    def action_edit_gpu_mem(self) -> None:
        """Edit GPU memory for selected partition."""
        name, partition = self._get_selected_partition()
        if not name:
            return

        table = self.table
        current = (
            str(partition.gpu_memory_gb.value)
            if partition.gpu_memory_gb.value > 0
            else ""
        )

        def handle_result(result: tuple[str, str] | None) -> None:
            if result is None:
                return
            callback_key, value = result
            if callback_key != f"gpu_mem_{name}":
                return

            try:
                gpu_mem = int(value) if value else 0
                if gpu_mem > 0:
                    partition.gpu_memory_gb.set(gpu_mem)
                else:
                    partition.gpu_memory_gb.reset()
                self.app.set_status(
                    f"{'Set' if gpu_mem else 'Cleared'} GPU memory for {name}"
                )
                self.app._auto_save()

                display = field_widget(
                    f"{gpu_mem} GB" if gpu_mem > 0 else None,
                    is_override=partition.gpu_memory_gb.is_overridden,
                )
                table.update_cell(name, self.col_keys["GPU Mem"], display)
            except ValueError:
                self.app.set_status(f"Invalid GPU memory value: {value}")

        self.app.push_screen(
            InputScreen(
                title=f"GPU Memory for '{name}'",
                label="Enter GPU memory in GB (e.g., 32, 80):",
                initial_value=current,
                callback_key=f"gpu_mem_{name}",
            ),
            handle_result,
        )

    def action_edit_priority(self) -> None:
        """Edit priority for selected partition."""
        name, partition = self._get_selected_partition()
        if not name:
            return

        table = self.table

        def handle_result(result: tuple[str, str] | None) -> None:
            if result is None:
                return
            callback_key, value = result
            if callback_key != f"priority_{name}":
                return

            try:
                priority = int(value) if value else 10
                if priority != 10:
                    partition.priority.set(priority)
                else:
                    partition.priority.reset()
                self.app.set_status(f"Set priority for {name}: {priority}")
                self.app._auto_save()

                display = field_widget(
                    str(partition.priority.value), partition.priority.is_overridden
                )
                table.update_cell(name, self.col_keys["Priority"], display)
            except ValueError:
                self.app.set_status(f"Invalid priority value: {value}")

        self.app.push_screen(
            InputScreen(
                title=f"Priority for '{name}'",
                label="Enter priority (higher = preferred, default: 10):",
                initial_value=str(partition.priority.value),
                callback_key=f"priority_{name}",
            ),
            handle_result,
        )

    def action_edit_qos(self) -> None:
        """Edit allowed QoS for selected partition."""
        name, partition = self._get_selected_partition()
        if not name:
            return

        all_qos = list(self.config.qos.keys())
        if not all_qos:
            self.app.set_status("No QoS available")
            return

        table = self.table

        def handle_result(result: tuple[str, list[str]] | None) -> None:
            if result is None:
                return
            callback_key, selected = result
            if callback_key != f"qos_{name}":
                return

            if selected:
                partition.allowed_qos.set(selected)
            else:
                partition.allowed_qos.reset()
            self.app.set_status(f"{'Set' if selected else 'Cleared'} QoS for {name}")
            self.app._auto_save()

            table.update_cell(
                name,
                self.col_keys["QoS"],
                list_field_widget(
                    partition.effective_qos,
                    is_inferred=partition.qos_is_inferred,
                    is_override=partition.allowed_qos.is_overridden,
                ),
            )

        self.app.push_screen(
            SelectionScreen(
                title=f"Allowed QoS for '{name}'",
                items=all_qos,
                selected=partition.effective_qos,
                callback_key=f"qos_{name}",
            ),
            handle_result,
        )

    def action_edit_accounts(self) -> None:
        """Edit allowed accounts for selected partition."""
        name, partition = self._get_selected_partition()
        if not name:
            return

        all_accounts = list(set(acc.account for acc in self.config.accounts))
        if not all_accounts:
            self.app.set_status("No accounts available")
            return

        table = self.table

        def handle_result(result: tuple[str, list[str]] | None) -> None:
            if result is None:
                return
            callback_key, selected = result
            if callback_key != f"accounts_{name}":
                return

            if selected:
                partition.allowed_accounts.set(selected)
            else:
                partition.allowed_accounts.reset()
            self.app.set_status(
                f"{'Set' if selected else 'Cleared'} accounts for {name}"
            )
            self.app._auto_save()

            table.update_cell(
                name,
                self.col_keys["Accounts"],
                list_field_widget(
                    partition.effective_accounts,
                    is_inferred=partition.accounts_is_inferred,
                    is_override=partition.allowed_accounts.is_overridden,
                ),
            )

        self.app.push_screen(
            SelectionScreen(
                title=f"Allowed accounts for '{name}'",
                items=all_accounts,
                selected=partition.effective_accounts,
                callback_key=f"accounts_{name}",
            ),
            handle_result,
        )

    def action_toggle_enabled(self) -> None:
        """Toggle enabled state for selected partition."""
        name, partition = self._get_selected_partition()
        if not name:
            return

        new_state = not partition.enabled.value
        partition.enabled.set(new_state)
        status = "enabled" if new_state else "disabled"
        self.app.set_status(f"Partition {name} {status}")
        self.app._auto_save()

        self.table.update_cell(
            name, self.col_keys["Enabled"], "✓" if new_state else "✗"
        )

    def action_reset_field(self) -> None:
        """Reset the current field to default."""
        table = self.table
        if table.cursor_row is None or table.row_count == 0:
            return

        cell_key = table.coordinate_to_cell_key((table.cursor_row, table.cursor_column))
        col_key = cell_key.column_key

        name = cell_key.row_key.value
        if not name:
            return

        partition = self.config.partitions.get(name)
        if not partition:
            return

        # Map column keys to reset actions
        col_name = None
        if col_key == self.col_keys["GPU Type"]:
            col_name = "GPU Type"
            partition.gpu_type.reset()
            display = field_widget(
                partition.effective_gpu_type,
                partition.gpu_type.is_overridden,
                partition.gpu_type.cluster_value,
                is_inferred=partition.gpu_type_is_inferred,
            )
            table.update_cell(name, col_key, display)
        elif col_key == self.col_keys["GPU Mem"]:
            col_name = "GPU Mem"
            partition.gpu_memory_gb.reset()
            gpu_mem = (
                f"{partition.effective_gpu_memory_gb} GB"
                if partition.effective_gpu_memory_gb > 0
                else None
            )
            display = field_widget(
                gpu_mem,
                is_override=partition.gpu_memory_gb.is_overridden,
                is_inferred=partition.gpu_memory_is_inferred,
            )
            table.update_cell(name, col_key, display)
        elif col_key == self.col_keys["Priority"]:
            col_name = "Priority"
            partition.priority.reset()
            table.update_cell(
                name,
                col_key,
                field_widget(
                    str(partition.priority.value), partition.priority.is_overridden
                ),
            )
        elif col_key == self.col_keys["QoS"]:
            col_name = "QoS"
            partition.allowed_qos.reset()
            display = list_field_widget(
                partition.effective_qos,
                is_inferred=partition.qos_is_inferred,
                is_override=partition.allowed_qos.is_overridden,
            )
            table.update_cell(name, col_key, display)
        elif col_key == self.col_keys["Accounts"]:
            col_name = "Accounts"
            partition.allowed_accounts.reset()
            display = list_field_widget(
                partition.effective_accounts,
                is_inferred=partition.accounts_is_inferred,
                is_override=partition.allowed_accounts.is_overridden,
            )
            table.update_cell(name, col_key, display)
        elif col_key == self.col_keys["Enabled"]:
            col_name = "Enabled"
            partition.enabled.reset()
            table.update_cell(name, col_key, "✓")
        else:
            return

        self.app.set_status(f"Reset {col_name} for {name}")
        self.app._auto_save()


class FeatureMappingTab(TabPane):
    """Tab for mapping features to cores, memory, GPU types, QoS, and accounts."""

    BINDINGS = [
        Binding("c", "edit_cores", "Cores"),
        Binding("m", "edit_memory", "Memory"),
        Binding("t", "edit_gpu_type", "GPU Type"),
        Binding("g", "edit_gpu_count", "GPUs"),
        Binding("v", "edit_gpu_mem", "GPU Mem"),
        Binding("o", "edit_qos", "QoS"),
        Binding("a", "edit_accounts", "Accounts"),
        Binding("delete", "reset_field", "Reset"),
        Binding("backspace", "reset_field", "Reset", show=False),
    ]

    def __init__(self, config: SlurmConfig):
        super().__init__("Features", id="features-tab")
        self.config = config
        # Column keys - set in compose()
        self.col_keys: dict[str, any] = {}

    def compose(self) -> ComposeResult:
        yield Static(
            "Map node features to cores, memory, GPU types/memory/count, QoS, and accounts. "
            "[cyan]Cyan = user configured[/cyan]. Press Enter to edit cell.",
            classes="help-text",
        )

        table = DataTable(id="features-table", cursor_type="cell")

        # Build column list dynamically based on available data
        col_names = ["Feature", "Cores", "Memory", "GPU Type", "GPUs", "GPU Mem"]
        self.has_qos = bool(self.config.qos)
        self.has_accounts = bool(self.config.accounts)
        if self.has_qos:
            col_names.append("QoS")
        if self.has_accounts:
            col_names.append("Accounts")

        keys = table.add_columns(*col_names)
        self.col_keys = dict(zip(col_names, keys))

        for feature_name in sorted(self.config.all_features):
            feature = self.config.features.get(feature_name)
            cores = (
                str(feature.cores.value)
                if feature and feature.cores.value > 0
                else None
            )
            memory = (
                f"{feature.memory_gb.value} GB"
                if feature and feature.memory_gb.value > 0
                else None
            )
            gpu_count = (
                str(feature.gpu_count.value)
                if feature and feature.gpu_count.value > 0
                else None
            )
            gpu_mem = (
                f"{feature.gpu_memory_gb.value} GB"
                if feature and feature.gpu_memory_gb.value > 0
                else None
            )

            row_data = [
                feature_name,
                field_widget(
                    cores, is_override=feature.cores.is_overridden if feature else False
                ),
                field_widget(
                    memory,
                    is_override=feature.memory_gb.is_overridden if feature else False,
                ),
                field_widget(
                    feature.gpu_type.value if feature else None,
                    is_override=feature.gpu_type.is_overridden if feature else False,
                ),
                field_widget(
                    gpu_count,
                    is_override=feature.gpu_count.is_overridden if feature else False,
                ),
                field_widget(
                    gpu_mem,
                    is_override=feature.gpu_memory_gb.is_overridden
                    if feature
                    else False,
                ),
            ]
            if self.has_qos:
                row_data.append(
                    list_field_widget(
                        feature.allowed_qos.value if feature else [],
                        is_override=feature.allowed_qos.is_overridden
                        if feature
                        else False,
                    )
                )
            if self.has_accounts:
                row_data.append(
                    list_field_widget(
                        feature.allowed_accounts.value if feature else [],
                        is_override=feature.allowed_accounts.is_overridden
                        if feature
                        else False,
                    )
                )

            table.add_row(*row_data, key=feature_name)

        yield table

    @on(DataTable.CellSelected)
    def on_cell_selected(self, event: DataTable.CellSelected) -> None:
        """Handle Enter key on cell."""
        col_key = event.cell_key.column_key
        editors = {
            self.col_keys["Cores"]: self.action_edit_cores,
            self.col_keys["Memory"]: self.action_edit_memory,
            self.col_keys["GPU Type"]: self.action_edit_gpu_type,
            self.col_keys["GPUs"]: self.action_edit_gpu_count,
            self.col_keys["GPU Mem"]: self.action_edit_gpu_mem,
        }
        if self.has_qos:
            editors[self.col_keys["QoS"]] = self.action_edit_qos
        if self.has_accounts:
            editors[self.col_keys["Accounts"]] = self.action_edit_accounts
        if col_key in editors:
            editors[col_key]()

    @on(DataTable.CellHighlighted)
    def on_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        """Show full cell value in status bar when highlighted."""
        col_key = event.cell_key.column_key
        row_key = event.cell_key.row_key
        feature_name = str(row_key.value) if row_key.value else None
        if not feature_name:
            self.app.set_status("")
            return

        feature = self.config.features.get(feature_name)

        # Show full values for columns that may be truncated
        full_value = None
        if feature:
            if self.has_qos and col_key == self.col_keys.get("QoS"):
                full_value = (
                    ", ".join(feature.allowed_qos.value)
                    if feature.allowed_qos.value
                    else None
                )
            elif self.has_accounts and col_key == self.col_keys.get("Accounts"):
                full_value = (
                    ", ".join(feature.allowed_accounts.value)
                    if feature.allowed_accounts.value
                    else None
                )

        if full_value:
            self.app.set_status(f"{feature_name}: {full_value}")
        else:
            self.app.set_status("")

    @property
    def table(self) -> DataTable:
        return self.query_one("#features-table", DataTable)

    def _get_selected_feature(self) -> str | None:
        """Get the currently selected feature name."""
        table = self.table
        if table.cursor_row is None or table.row_count == 0:
            self.app.set_status("No feature selected")
            return None

        row_key = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key
        feature = str(row_key.value) if row_key.value else None
        if not feature:
            self.app.set_status("No feature selected")
            return None
        return feature

    def action_edit_cores(self) -> None:
        """Edit cores for selected feature."""
        feature_name = self._get_selected_feature()
        if not feature_name:
            return

        feature = self.config.features.get(feature_name)
        current = (
            str(feature.cores.value) if feature and feature.cores.value > 0 else ""
        )
        table = self.table

        def handle_result(result: tuple[str, str] | None) -> None:
            if result is None:
                return
            callback_key, value = result
            if callback_key != f"cores_{feature_name}":
                return

            try:
                cores = int(value) if value else 0
                feature = self.config.get_feature(feature_name)
                if cores > 0:
                    feature.cores.set(cores)
                else:
                    feature.cores.reset()
                self.app.set_status(
                    f"{'Set' if cores else 'Cleared'} cores for {feature_name}"
                )

                if not feature.has_any_config():
                    del self.config.features[feature_name]
                self.app._auto_save()

                display = field_widget(
                    str(cores) if cores > 0 else None,
                    is_override=feature.cores.is_overridden
                    if feature_name in self.config.features
                    else False,
                )
                table.update_cell(feature_name, self.col_keys["Cores"], display)
            except ValueError:
                self.app.set_status(f"Invalid cores value: {value}")

        self.app.push_screen(
            InputScreen(
                title=f"Cores for '{feature_name}'",
                label="Enter number of CPU cores:",
                initial_value=current,
                callback_key=f"cores_{feature_name}",
            ),
            handle_result,
        )

    def action_edit_memory(self) -> None:
        """Edit memory for selected feature."""
        feature_name = self._get_selected_feature()
        if not feature_name:
            return

        feature = self.config.features.get(feature_name)
        current = (
            str(feature.memory_gb.value)
            if feature and feature.memory_gb.value > 0
            else ""
        )
        table = self.table

        def handle_result(result: tuple[str, str] | None) -> None:
            if result is None:
                return
            callback_key, value = result
            if callback_key != f"memory_{feature_name}":
                return

            try:
                memory = int(value) if value else 0
                feature = self.config.get_feature(feature_name)
                if memory > 0:
                    feature.memory_gb.set(memory)
                else:
                    feature.memory_gb.reset()
                self.app.set_status(
                    f"{'Set' if memory else 'Cleared'} memory for {feature_name}"
                )

                if not feature.has_any_config():
                    del self.config.features[feature_name]
                self.app._auto_save()

                display = field_widget(
                    f"{memory} GB" if memory > 0 else None,
                    is_override=feature.memory_gb.is_overridden
                    if feature_name in self.config.features
                    else False,
                )
                table.update_cell(feature_name, self.col_keys["Memory"], display)
            except ValueError:
                self.app.set_status(f"Invalid memory value: {value}")

        self.app.push_screen(
            InputScreen(
                title=f"Memory for '{feature_name}'",
                label="Enter memory in GB:",
                initial_value=current,
                callback_key=f"memory_{feature_name}",
            ),
            handle_result,
        )

    def action_edit_gpu_type(self) -> None:
        """Edit GPU type for selected feature."""
        feature_name = self._get_selected_feature()
        if not feature_name:
            return

        feature = self.config.features.get(feature_name)
        table = self.table

        def handle_result(result: tuple[str, str] | None) -> None:
            if result is None:
                return
            callback_key, value = result
            if callback_key != feature_name:
                return

            feature = self.config.get_feature(feature_name)
            if value:
                feature.gpu_type.set(value)
            else:
                feature.gpu_type.reset()
            self.app.set_status(
                f"{'Set' if value else 'Cleared'} GPU type for {feature_name}"
            )

            # Remove feature if no config left
            if not feature.has_any_config():
                del self.config.features[feature_name]
            self.app._auto_save()

            display = (
                field_widget(value, is_override=True)
                if value
                else Text("-", style="dim")
            )
            table.update_cell(feature_name, self.col_keys["GPU Type"], display)

        self.app.push_screen(
            InputScreen(
                title=f"GPU Type for '{feature_name}'",
                label="Enter GPU type (e.g., v100, a100, h100):",
                initial_value=feature.gpu_type.value if feature else "",
                callback_key=feature_name,
            ),
            handle_result,
        )

    def action_edit_gpu_count(self) -> None:
        """Edit GPU count for selected feature."""
        feature_name = self._get_selected_feature()
        if not feature_name:
            return

        feature = self.config.features.get(feature_name)
        current = (
            str(feature.gpu_count.value)
            if feature and feature.gpu_count.value > 0
            else ""
        )
        table = self.table

        def handle_result(result: tuple[str, str] | None) -> None:
            if result is None:
                return
            callback_key, value = result
            if callback_key != f"gpu_count_{feature_name}":
                return

            try:
                gpu_count = int(value) if value else 0
                feature = self.config.get_feature(feature_name)
                if gpu_count > 0:
                    feature.gpu_count.set(gpu_count)
                else:
                    feature.gpu_count.reset()
                self.app.set_status(
                    f"{'Set' if gpu_count else 'Cleared'} GPU count for {feature_name}"
                )

                if not feature.has_any_config():
                    del self.config.features[feature_name]
                self.app._auto_save()

                display = field_widget(
                    str(gpu_count) if gpu_count > 0 else None,
                    is_override=feature.gpu_count.is_overridden
                    if feature_name in self.config.features
                    else False,
                )
                table.update_cell(feature_name, self.col_keys["GPUs"], display)
            except ValueError:
                self.app.set_status(f"Invalid GPU count value: {value}")

        self.app.push_screen(
            InputScreen(
                title=f"GPU Count for '{feature_name}'",
                label="Enter number of GPUs:",
                initial_value=current,
                callback_key=f"gpu_count_{feature_name}",
            ),
            handle_result,
        )

    def action_edit_gpu_mem(self) -> None:
        """Edit GPU memory for selected feature."""
        feature_name = self._get_selected_feature()
        if not feature_name:
            return

        feature = self.config.features.get(feature_name)
        current = (
            str(feature.gpu_memory_gb.value)
            if feature and feature.gpu_memory_gb.value > 0
            else ""
        )
        table = self.table

        def handle_result(result: tuple[str, str] | None) -> None:
            if result is None:
                return
            callback_key, value = result
            if callback_key != f"gpu_mem_{feature_name}":
                return

            try:
                gpu_mem = int(value) if value else 0
                feature = self.config.get_feature(feature_name)
                if gpu_mem > 0:
                    feature.gpu_memory_gb.set(gpu_mem)
                else:
                    feature.gpu_memory_gb.reset()
                self.app.set_status(
                    f"{'Set' if gpu_mem else 'Cleared'} GPU memory for {feature_name}"
                )

                if not feature.has_any_config():
                    del self.config.features[feature_name]
                self.app._auto_save()

                display = field_widget(
                    f"{gpu_mem} GB" if gpu_mem > 0 else None,
                    is_override=feature.gpu_memory_gb.is_overridden
                    if feature_name in self.config.features
                    else False,
                )
                table.update_cell(feature_name, self.col_keys["GPU Mem"], display)
            except ValueError:
                self.app.set_status(f"Invalid GPU memory value: {value}")

        self.app.push_screen(
            InputScreen(
                title=f"GPU Memory for '{feature_name}'",
                label="Enter GPU memory in GB (e.g., 40, 80):",
                initial_value=current,
                callback_key=f"gpu_mem_{feature_name}",
            ),
            handle_result,
        )

    def action_edit_qos(self) -> None:
        """Edit allowed QoS for selected feature."""
        feature_name = self._get_selected_feature()
        if not feature_name:
            return

        all_qos = list(self.config.qos.keys())
        if not all_qos:
            self.app.set_status("No QoS available")
            return

        feature = self.config.features.get(feature_name)
        table = self.table

        def handle_result(result: tuple[str, list[str]] | None) -> None:
            if result is None:
                return
            callback_key, selected = result
            if callback_key != f"qos_{feature_name}":
                return

            feature = self.config.get_feature(feature_name)
            if selected:
                feature.allowed_qos.set(selected)
            else:
                feature.allowed_qos.reset()
            self.app.set_status(
                f"{'Set' if selected else 'Cleared'} QoS for {feature_name}"
            )

            if not feature.has_any_config():
                del self.config.features[feature_name]
            self.app._auto_save()

            table.update_cell(
                feature_name,
                self.col_keys["QoS"],
                list_field_widget(selected, is_override=bool(selected)),
            )

        self.app.push_screen(
            SelectionScreen(
                title=f"Allowed QoS for '{feature_name}'",
                items=all_qos,
                selected=feature.allowed_qos.value if feature else [],
                callback_key=f"qos_{feature_name}",
            ),
            handle_result,
        )

    def action_edit_accounts(self) -> None:
        """Edit allowed accounts for selected feature."""
        feature_name = self._get_selected_feature()
        if not feature_name:
            return

        all_accounts = list(set(acc.account for acc in self.config.accounts))
        if not all_accounts:
            self.app.set_status("No accounts available")
            return

        feature = self.config.features.get(feature_name)
        table = self.table

        def handle_result(result: tuple[str, list[str]] | None) -> None:
            if result is None:
                return
            callback_key, selected = result
            if callback_key != f"accounts_{feature_name}":
                return

            feature = self.config.get_feature(feature_name)
            if selected:
                feature.allowed_accounts.set(selected)
            else:
                feature.allowed_accounts.reset()
            self.app.set_status(
                f"{'Set' if selected else 'Cleared'} accounts for {feature_name}"
            )

            if not feature.has_any_config():
                del self.config.features[feature_name]
            self.app._auto_save()

            table.update_cell(
                feature_name,
                self.col_keys["Accounts"],
                list_field_widget(selected, is_override=bool(selected)),
            )

        self.app.push_screen(
            SelectionScreen(
                title=f"Allowed accounts for '{feature_name}'",
                items=all_accounts,
                selected=feature.allowed_accounts.value if feature else [],
                callback_key=f"accounts_{feature_name}",
            ),
            handle_result,
        )

    def action_reset_field(self) -> None:
        """Reset the current field to default."""
        table = self.table
        if table.cursor_row is None or table.row_count == 0:
            return

        cell_key = table.coordinate_to_cell_key((table.cursor_row, table.cursor_column))
        col_key = cell_key.column_key

        feature_name = cell_key.row_key.value
        if not feature_name:
            return

        feature = self.config.features.get(feature_name)
        if not feature:
            return

        # Map column keys to reset actions
        col_name = None
        if col_key == self.col_keys["GPU Type"]:
            col_name = "GPU Type"
            feature.gpu_type.reset()
            table.update_cell(feature_name, col_key, Text("-", style="dim"))
        elif col_key == self.col_keys["GPUs"]:
            col_name = "GPUs"
            feature.gpu_count.reset()
            table.update_cell(feature_name, col_key, Text("-", style="dim"))
        elif col_key == self.col_keys["GPU Mem"]:
            col_name = "GPU Mem"
            feature.gpu_memory_gb.reset()
            table.update_cell(feature_name, col_key, Text("-", style="dim"))
        elif col_key == self.col_keys["QoS"]:
            col_name = "QoS"
            feature.allowed_qos.reset()
            table.update_cell(feature_name, col_key, Text("-", style="dim"))
        elif col_key == self.col_keys["Accounts"]:
            col_name = "Accounts"
            feature.allowed_accounts.reset()
            table.update_cell(feature_name, col_key, Text("-", style="dim"))
        else:
            return

        self.app.set_status(f"Reset {col_name} for {feature_name}")

        if not feature.has_any_config():
            del self.config.features[feature_name]
        self.app._auto_save()


class QoSTab(TabPane):
    """Tab for viewing QoS information (read-only)."""

    def __init__(self, config: SlurmConfig):
        super().__init__("QoS", id="qos-tab")
        self.config = config

    def compose(self) -> ComposeResult:
        yield Static("QoS information from cluster (read-only).", classes="help-text")

        table = DataTable(id="qos-table")
        table.add_columns("QoS", "Max Wall Time", "Priority", "GRES Limits")

        for name, q in sorted(self.config.qos.items()):
            gres_str = ", ".join(q.gres_limits[:2]) if q.gres_limits else "-"
            if len(q.gres_limits) > 2:
                gres_str += f" (+{len(q.gres_limits) - 2})"

            table.add_row(
                name,
                q.max_wall or "unlimited",
                str(q.priority),
                gres_str,
            )

        yield table


class AccountsTab(TabPane):
    """Tab for viewing account associations (read-only)."""

    def __init__(self, config: SlurmConfig):
        super().__init__("Accounts", id="accounts-tab")
        self.config = config

    def compose(self) -> ComposeResult:
        yield Static("Your account associations from cluster.", classes="help-text")

        table = DataTable(id="accounts-table")
        table.add_columns("Account", "Partition", "Available QoS")

        for acc in self.config.accounts:
            partition_str = acc.partition or "(all)"
            qos_str = ", ".join(acc.qos_list[:4]) if acc.qos_list else "(default)"
            if len(acc.qos_list) > 4:
                qos_str += f" (+{len(acc.qos_list) - 4})"

            table.add_row(
                acc.account,
                partition_str,
                qos_str,
            )

        yield table


class SettingsTab(TabPane):
    """Tab for global settings organized in a tree."""

    BINDINGS = [
        Binding("space", "toggle", "Toggle"),
        Binding("enter", "toggle", "Toggle", show=False),
    ]

    def __init__(self, config: SlurmConfig):
        super().__init__("Settings", id="settings-tab")
        self.config = config
        # Map node IDs to setting keys
        self._node_to_setting: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Static(
            "Global settings. Press Space or Enter to toggle. [dim]✓ = enabled[/dim]",
            classes="help-text",
        )

        tree: Tree[str] = Tree("Settings", id="settings-tree")
        tree.root.expand()

        # Group settings by category
        categories: dict[str, list] = {}
        for setting in self.config.get_settings():
            if setting.category not in categories:
                categories[setting.category] = []
            categories[setting.category].append(setting)

        # Build tree structure
        for category, settings in categories.items():
            category_node = tree.root.add(f"[bold]{category}[/bold]", expand=True)
            for setting in settings:
                checked = "✓" if setting.get_value() else "○"
                label = f"[green]{checked}[/green] {setting.label} [dim]- {setting.description}[/dim]"
                node = category_node.add_leaf(label)
                self._node_to_setting[str(node.id)] = setting.key

        yield tree

    @property
    def tree(self) -> Tree:
        return self.query_one("#settings-tree", Tree)

    def action_toggle(self) -> None:
        self._toggle_current()

    @on(Tree.NodeSelected)
    def on_node_selected(self, event: Tree.NodeSelected) -> None:
        self._toggle_current()

    def _toggle_current(self) -> None:
        tree = self.tree
        node = tree.cursor_node
        if node is None:
            return

        # Get setting key from node ID
        node_id = str(node.id)
        setting_key = self._node_to_setting.get(node_id)
        if not setting_key:
            return  # Category node, not a setting

        # Find and toggle the setting
        for setting in self.config.get_settings():
            if setting.key == setting_key:
                new_value = not setting.get_value()
                setting.set_value(new_value)

                # Update node label
                checked = "✓" if new_value else "○"
                node.set_label(
                    f"[green]{checked}[/green] {setting.label} [dim]- {setting.description}[/dim]"
                )

                self.app.set_status(
                    f"{'Enabled' if new_value else 'Disabled'} {setting.label}"
                )
                self.app._auto_save()

                # Refresh partitions table to reflect changes
                try:
                    partitions_tab = self.app.query_one(PartitionsTab)
                    partitions_tab.refresh_table()
                except Exception as e:
                    self.app.set_status(f"Refresh error: {e}")
                break


class SlurmConfigApp(App):
    """Textual app for SLURM launcher configuration."""

    CSS_PATH = "slurm_config.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("g", "generate", "Generate"),
        Binding("s", "save", "Save Config"),
        Binding("r", "reload", "Reload"),
        Binding("x", "export", "Export"),
        Binding("i", "import_config", "Import"),
    ]

    def __init__(
        self,
        config: SlurmConfig,
        config_path: Path,
        new_partitions: set[str] | None = None,
        cache_file: Path | None = None,
        use_cache: bool = False,
    ):
        super().__init__()
        self.config = config
        self.config_path = config_path
        self.new_partitions = new_partitions or set()
        self.cache_file = cache_file or get_default_cache_file()
        self.use_cache = use_cache

    def compose(self) -> ComposeResult:
        yield Static(
            "[bold]SLURM Launcher Configuration[/bold]",
            id="app-title",
        )

        with TabbedContent():
            yield PartitionsTab(self.config, self.new_partitions)
            yield FeatureMappingTab(self.config)
            if self.config.qos:
                yield QoSTab(self.config)
            if self.config.accounts:
                yield AccountsTab(self.config)
            yield SettingsTab(self.config)

        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Set initial status on mount."""
        if self.new_partitions:
            self.set_status(
                f"Cluster: {self.config.cluster_name} "
                f"(+{len(self.new_partitions)} new partitions - save to keep)"
            )
        else:
            self.set_status(f"Cluster: {self.config.cluster_name}")

    @on(TabbedContent.TabActivated)
    def on_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Clear status bar when switching tabs."""
        self.set_status("")

    def set_status(self, message: str) -> None:
        """Update status bar."""
        status = self.query_one("#status-bar", Static)
        status.update(message)

    def action_quit(self) -> None:
        """Quit the application."""
        self._auto_save()
        self.exit()

    def _auto_save(self) -> None:
        """Auto-save user configuration."""
        if not self.config:
            return
        try:
            self.config.save_to_yaml(self.config_path)
        except Exception as e:
            self.set_status(f"Auto-save error: {e}")

    def action_generate(self) -> None:
        """Generate launchers.py."""
        self.set_status("Generate not yet implemented")

    def action_save(self) -> None:
        """Save configuration to YAML."""
        if not self.config:
            self.set_status("No configuration to save")
            return

        try:
            self.config.save_to_yaml(self.config_path)
            self.set_status(f"Saved configuration to {self.config_path}")
        except Exception as e:
            self.set_status(f"Error saving: {e}")

    async def action_reload(self) -> None:
        """Reload cluster information."""
        self.set_status("Reloading...")
        try:
            cache = SlurmCommandCache(
                cache_file=self.cache_file,
                use_cache=self.use_cache,
            )
            cluster_data = detect_cluster_info(cache)
            self.config = SlurmConfig.from_cluster_and_file(
                cluster_data, self.config_path
            )
            self.new_partitions = set(self.config.get_new_partitions(self.config_path))
            self.set_status(f"Reloaded: {self.config.cluster_name}")
            await self.recompose()
        except Exception as e:
            self.set_status(f"Reload error: {e}")

    def action_export(self) -> None:
        """Export configuration to a file for sharing."""
        yaml_filter = Filters(("YAML files", lambda p: p.suffix in (".yaml", ".yml")))

        def handle_result(export_path: Path | None) -> None:
            if export_path is None:
                self.set_status("Export cancelled")
                return

            try:
                # Ensure .yaml extension
                if export_path.suffix not in (".yaml", ".yml"):
                    export_path = export_path.with_suffix(".yaml")
                self.config.save_to_yaml(export_path)
                self.set_status(f"Exported to: {export_path}")
            except Exception as e:
                self.set_status(f"Export error: {e}")

        self.push_screen(
            FileSave(
                location=self.config_path.parent,
                default_file=f"{self.config.cluster_name}_config.yaml",
                filters=yaml_filter,
                title="Export Configuration",
            ),
            handle_result,
        )

    def action_import_config(self) -> None:
        """Import configuration from a file."""
        yaml_filter = Filters(("YAML files", lambda p: p.suffix in (".yaml", ".yml")))

        def handle_result(import_path: Path | None) -> None:
            if import_path is None:
                self.set_status("Import cancelled")
                return

            try:
                self.config.load_from_yaml(import_path)
                self._auto_save()
                self.set_status(f"Imported from: {import_path}")
                # Trigger recompose to refresh all tabs
                self.call_later(self._refresh_after_import)
            except Exception as e:
                self.set_status(f"Import error: {e}")

        self.push_screen(
            FileOpen(
                location=self.config_path.parent,
                filters=yaml_filter,
                title="Import Configuration",
            ),
            handle_result,
        )

    async def _refresh_after_import(self) -> None:
        """Refresh the UI after importing configuration."""
        await self.recompose()


def run_tui(
    cache_file: Path | None = None,
    use_cache: bool = False,
    config_path: Path | None = None,
) -> None:
    """Run the SLURM configuration TUI."""
    from rich.console import Console

    console = Console()

    # Resolve paths
    if config_path is None:
        config_path = Path("~/.config/experimaestro/slurm.yaml").expanduser()
    if cache_file is None:
        cache_file = get_default_cache_file()

    # Load cluster info before starting TUI (easier to debug)
    console.print("[cyan]Loading SLURM cluster information...[/cyan]")
    cache = SlurmCommandCache(cache_file=cache_file, use_cache=use_cache)
    cluster_data = detect_cluster_info(cache)
    console.print(f"[green]Cluster:[/green] {cluster_data.cluster_name}")

    # Create config and load saved settings
    config = SlurmConfig.from_cluster_and_file(cluster_data, config_path)

    # Detect new partitions
    new_partitions = set(config.get_new_partitions(config_path))
    if new_partitions:
        console.print(
            f"[yellow]New partitions detected:[/yellow] {', '.join(new_partitions)}"
        )

    # Start TUI
    app = SlurmConfigApp(
        config=config,
        config_path=config_path,
        new_partitions=new_partitions,
        cache_file=cache_file,
        use_cache=use_cache,
    )
    app.run()
