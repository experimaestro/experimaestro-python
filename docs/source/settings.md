# Experimaestro configuration

This pages describes how experimaestro itself can be configured for your need.

## Settings file

Default settings can be stored in the `$HOME/.config/experimaestro/settings.yaml file`.
All the settings are optional

```yaml

# Experiment server settings
server:
    port: 12345
    token: 2134inmd8132323
    host: 192.168.1.1

# Environment variables for experimental tasks
env:
  JAVA_HOME: /path/to/java

workspaces:
    # First workspace is the default
  - id: neuralir
    path: ~/experiments/xpmir
    # Specific environment for this workspace
    env:
      VARNAME: VALUE
    # Auto-select this workspace when experiment ID matches these patterns
    triggers:
      - "neuralir-*"
      - "my-awesome-experiment"

  - id: other_project
    path: ~/experiments/other
    triggers:
      - "other-*"
```

## Workspace Selection

When running experiments with `experimaestro run-experiment`, the workspace is selected using the following priority:

1. **Explicit `--workspace` flag**: If provided, uses the workspace with that ID
2. **Explicit `--workdir` flag**: If provided, uses that directory directly
3. **Auto-selection via triggers**: If the experiment ID matches a workspace trigger pattern, that workspace is selected
4. **Default workspace**: The first workspace in the settings list is used as fallback

### Workspace Triggers

Workspace triggers allow automatic workspace selection based on experiment ID patterns. This is useful when you have multiple workspaces for different projects and want experiments to automatically run in the correct workspace.

Triggers use glob-style pattern matching:
- `*` matches any characters (e.g., `"base_id-*"` matches `"base_id-123"`, `"base_id-test"`)
- `?` matches a single character
- `[abc]` matches any character in the brackets

**Example:**

```yaml
workspaces:
  - id: neuralir
    path: ~/experiments/xpmir
    triggers:
      - "neuralir-*"          # Matches neuralir-123, neuralir-test, etc.
      - "ir-experiment"       # Exact match
      - "test-ir-*"          # Matches test-ir-1, test-ir-baseline, etc.

  - id: nlp
    path: ~/experiments/nlp
    triggers:
      - "nlp-*"
      - "transformer-*"

  - id: default
    path: ~/experiments/default
    # No triggers - this is the fallback
```

If an experiment's ID matches multiple workspace triggers, the first matching workspace in the list wins.

## Remote Workspaces (SSH)

Workspaces can be configured with SSH settings for remote monitoring. A workspace
with `ssh` settings is a **remote workspace** — it is automatically excluded from
local experiment workspace discovery (trigger matching and default workspace selection).

```yaml
workspaces:
  - id: my-cluster
    path: /remote/path/to/experiments
    ssh:
      host: user@cluster
      shell_init: "source /etc/profile; module load python/3.10"
      uv_offline: true
      python: python3

  - id: local-workspace
    path: ~/experiments/local
    # No ssh — this is a local workspace
```

### SSH Settings

| Setting | Description |
|---------|-------------|
| `host` | SSH host (e.g., `user@cluster` or SSH config alias). Required. |
| `shell_init` | Shell commands to run before the remote experimaestro command (e.g., `source /etc/profile; module load python/3.10`) |
| `uv_offline` | Pass `--offline` to `uv tool run` (use cached packages, no network). Useful on HPC clusters without direct PyPI access. |
| `python` | Python interpreter for `uv tool run --python` |
| `xpm_path` | Path to experimaestro on remote host (bypasses `uv tool run` entirely) |
| `options` | Additional SSH options (e.g., `["-p", "2222"]`) |

### Using Workspace SSH Settings

Once configured, use `--workspace` with `ssh-monitor` to avoid repeating host/options:

```bash
# All settings come from the workspace configuration
experimaestro experiments ssh-monitor --workspace my-cluster

# CLI flags override workspace settings
experimaestro experiments ssh-monitor --workspace my-cluster --uv-offline
```

All SSH settings can be overridden by CLI flags (`--remote-shell-init`, `--uv-offline`,
`--remote-python`, `--remote-xpm`, `-o`).
