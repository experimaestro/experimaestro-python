## 0.29.4 (2023-06-26)

### Fix

- better instance()

## 0.29.3 (2023-06-26)

### Fix

- identifiers

## 0.29.2 (2023-06-21)

### Fix

- pre-task are properly handled

## 0.29.1 (2023-06-20)

### Fix

- exception thrown when adding pre-task to a sealed config

## 0.29.0 (2023-06-19)

### Fix

- pre task dependencies are taken into account

## 0.28.0 (2023-06-16)

### BREAKING CHANGE

- - subparam were removed (should be replaced by something more stable)
- serialiazed configurations were removed (too much trouble too)

### Feat

- show dependencies when simulating
- removed config wrapper
- easier path LW task access
- **Lightweight-pre-tasks**: Lightweight pre-tasks allow code to be executed to modify objects

### Fix

- bug in dependency tracking

## 0.27.0 (2023-05-26)

### Feat

- Expose the unwrap function

### Fix

- Removes unnecessary server logs

## 0.26.0 (2023-05-26)

### Fix

- Fix submit hooks (and document them)

## 0.25.0 (2023-05-26)

## 0.24.0 (2023-05-23)

### Feat

- serialized configurations

### Fix

- requirement for fabric
- add gevent-websocket for supporting websockets

### Refactor

- Changed TaskOutput to ConfigWrapper

## 0.23.0 (2023-04-07)

### Feat

- submit hooks to allow e.g. changing the environment variables

## 0.22.0 (2023-04-05)

### Feat

- tags as immutable and hashable dicts

### Fix

- corrected service status update for servers
- improved server

## 0.21.0 (2023-03-28)

### Feat

- When an experiment fails, display the path to stderr
- service proxying

### Fix

- Information message when locking experiment
- Improving slurm support
- Fix test bugs
- better handlign of services

### Refactor

- **server**: switched to flask and socketio for future evolutions of the server

## 0.20.0 (2023-02-18)

### Feat

- improvements for dry-run modes to show completed jobs

### Refactor

- more reliable identifier computation

## 0.19.2 (2023-02-16)

### Fix

- better identifier recomputation

## 0.19.1 (2023-02-15)

### Fix

- fix bugs with generate/dry-run modes

## 0.19.0 (2023-02-14)

### Feat

- allow using the old task identifier computation to fix params.json

## 0.18.0 (2023-02-13)

### BREAKING CHANGE

- New identifiers will be different in all cases - use the deprecated command to recompute identifiers for old experiments
- For any task output which is different than the task itself, the identifier will change

### Feat

- **configuration**: re-use computed sub-configuration identifiers

### Fix

- **server**: fix some display bugs in the UI
- **configuration**: fixed more bugs with identifiers
- **configuration**: fixed bugs with identifiers
- **configuration**: serialize the task to recompute exactly the identifier

### Refactor

- removed jsonstreams dependency

## 0.16.0 (2023-02-08)

### Feat

- **server**: web services for experiment server

## 0.15.1 (2023-02-08)

### Fix

- wrong indent
