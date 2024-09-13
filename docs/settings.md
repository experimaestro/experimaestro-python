# Configuration

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
```
