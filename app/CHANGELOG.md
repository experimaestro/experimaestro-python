## 0.16.0 (2023-02-08)

### Feat

- **server**: web services for experiment server

## 0.15.1 (2023-02-08)

### Fix

- wrong indent

## 0.15.0 (2023-02-08)

### Feat

- **scheduler**: foundations for experiment services

## 0.14.6 (2023-02-06)

## 0.14.5 (2023-02-06)

## 0.14.4 (2023-02-03)

## 0.14.3 (2023-01-24)

## 0.14.2 (2023-01-21)

## 0.14.1 (2023-01-21)

## 0.14 (2023-01-19)

## 0.13.3 (2022-12-24)

## 0.13.2 (2022-12-24)

## 0.13.1 (2022-12-01)

## 0.13.0 (2022-12-01)

## 0.12.2 (2022-11-30)

## 0.12.0 (2022-11-29)

## 0.11.8 (2022-11-22)

## 0.11.7 (2022-10-24)

## 0.11.6 (2022-10-14)

## 0.11.5 (2022-10-08)

## 0.11.3 (2022-10-06)

## 0.11.2 (2022-10-05)

## 0.11.1 (2022-10-05)

## 0.10.6 (2022-09-29)

## 0.10.5 (2022-05-31)

## 0.10.4 (2022-05-12)

## 0.10.3 (2022-02-10)

## 0.10.2 (2022-02-07)

## 0.10.1 (2022-02-04)

## 0.10.0 (2022-02-04)

## 0.9.12 (2022-01-11)

## 0.9.11 (2021-11-19)

## 0.9.10 (2021-09-23)

## 0.9.9 (2021-07-22)

## 0.9.8 (2021-07-22)

## 0.9.7 (2021-07-19)

## 0.9.6 (2021-07-19)

## 0.9.5 (2021-07-19)

## 0.9.4 (2021-05-26)

## 0.9.3 (2021-05-26)

## 0.9.2 (2021-05-25)

## 0.9.1 (2021-05-24)

## 0.9.0 (2021-05-24)

## 0.8.9 (2021-05-20)

## 0.8.8 (2021-04-05)

## 0.8.7 (2021-04-03)

## 0.8.6 (2021-03-18)

## 0.8.5 (2021-03-18)

## 0.8.4 (2021-03-02)

## 0.8.3 (2021-02-19)

## 0.8.2 (2021-01-29)

## 0.8.1 (2021-01-28)

## 0.8.0 (2021-01-27)

## 0.7.12 (2021-01-12)

## 0.7.11 (2021-01-07)

## 0.7.10 (2021-01-07)

## 0.7.9 (2020-12-15)

## 0.7.8 (2020-12-09)

## 0.7.7 (2020-10-19)

## 0.7.6 (2020-10-02)

## 0.7.5 (2020-09-17)

## 0.7.4 (2020-09-11)

## 0.7.3 (2020-07-07)

## 0.7.2 (2020-05-27)

## 0.7.0 (2020-05-26)

## 0.6.0 (2020-05-21)

## 0.5.9 (2020-01-14)

## 0.5.7 (2020-01-13)

## 0.5.6 (2020-01-10)

## 0.5.5 (2019-12-19)

## 0.5.4 (2019-12-12)

## 0.5.3 (2019-12-12)

## 0.5.2 (2019-12-12)

## 0.5.1 (2019-12-11)

## 0.5.0 (2019-12-11)

## v1.7.0rc4 (2025-04-08)

## v1.7.0rc3 (2025-04-08)

## v1.7.0rc2 (2025-04-08)

## v1.7.0rc1 (2025-04-08)

## v1.7.0rc0 (2025-04-08)

### Feat

- dynamic outputs and field factory

### Fix

- npm audit
- removed temporarly dynamic output
- wheel includes server data files
- launchers.py is a resource

## v1.6.2 (2025-01-24)

### Fix

- python path is now propagated to the tasks

## v1.6.1 (2025-01-22)

### Fix

- **cli.py**: Python path should be modified including when running the experiments

## v1.6.0 (2025-01-16)

### Feat

- python path and module in experiments

### Refactor

- pre-commit commitizen update

## v1.5.14 (2024-11-05)

### Fix

- when fabric is not there, use placeholders

## v1.5.13 (2024-10-23)

### Fix

- js vulnerabilities fix

## v1.5.11 (2024-09-13)

### Fix

- python 3.12 compatibility (but without ssh)
- no dependencies for loaded configurations
- fabric is now optional

## v1.5.10 (2024-09-02)

### Feat

- include jobs not in experiments
- include jobs not in experiments

## v1.5.9 (2024-07-28)

### Fix

- bug in setting workspace env

## v1.5.8 (2024-07-27)

### Fix

- andles better null process
- use a better way to load the XP file

### Refactor

- removed the launchers.yaml support

## v1.5.7 (2024-05-31)

### Feat

- better job interactions

### Fix

- bug when no settings.yaml file
- set init tasks before job creation
- removed extra logging information
- wrong information reported
- wrong test for running/done tasks

### Refactor

- change the cli source dir

## v1.5.6 (2024-05-10)

### Fix

- bug in environment

## v1.5.5 (2024-05-10)

### Fix

- bugs in normal task exit

### Refactor

- change run-experiment parameters
- removed unused environments

## v1.5.4 (2024-03-06)

### Fix

- get back to sys.exit

## v1.5.3 (2024-03-06)

### Fix

- force exit with multiprocessing

## v1.5.2 (2024-03-06)

### Fix

- multiple launchers

## v1.5.1 (2024-02-29)

### Feat

- handles properly mp.spawn
- requirements can now include a disjunction
- environment can be defined in the settings

### Fix

- **slurm**: Check if slurm process is alive before returning it
- really using the workspace directory

## v1.5.0 (2024-02-26)

### Feat

- uses job failure status
- HandledException to shorten error stack trace

### Fix

- Deprecate YAML defined launchers
- added pre/post yaml options
- better error messages
- support python 3.12 pathlib
- **typinutils**: test for optional
- _collect_parameters is in Python 3.11
- **types**: accept generics as Param (but no real validation)
- **ConfigurationBase**: Documentation and new fields to describe an experiment
- process YAML file in the right order
- Configuration must be OmegaConf to be transformed

### Refactor

- Use XPMValue and XPMConfig to distinguish configurations and values

## v1.4.3 (2023-12-21)

### Fix

- set id as MISSING

## v1.4.2 (2023-12-21)

### Fix

- parent should be really optional

## v1.4.1 (2023-12-19)

### Feat

- decorator class method for experiment helpers
- representation of identifier as hex string
- load task from directory

### Fix

- identifier is a method not a property

## v1.4.0 (2023-12-16)

### Feat

- run-experiment cli

## v1.3.6 (2023-12-12)

### Feat

- Saving/Loading from experiments

## v1.3.5 (2023-12-08)

### Feat

- access run mode from experiment

## v1.3.4 (2023-12-07)

### Feat

- Test for falsy documented

## v1.3.3 (2023-12-06)

## v1.3.2 (2023-11-28)

### Feat

- utility function to format time for SLURM

## v1.3.1 (2023-11-27)

### Fix

- bad assert test

## v1.3.0 (2023-11-24)

### Feat

- launchers.py specification file
- Serialization for arbitrary data structures

## v1.2.3 (2023-10-04)

### Fix

- task identifier and reload

## v2.0.0 (2023-10-03)

### Fix

- task identifier and reload

## v1.2.1 (2023-10-03)

### Feat

- workspace settings

## v1.2.0 (2023-10-03)

### Feat

- init tasks

## v1.1.2 (2023-08-28)

### Fix

- undocumented produces more information

## v1.1.1 (2023-10-03)

### Feat

- undocumented in sphinx documentation
- Added decorator (NOOP) for config only methods

## v1.1.0 (2023-10-03)

### Feat

- Generic documentation checker (for use in automated tests)
- improved check documentation command

## v1.0.0 (2023-10-03)

### Feat

- List experiments
- **documentation**: Checks for undocumented configuration objects in a package

## v0.30.0 (2023-10-03)

### Feat

- **configuration**: Add access to pre-task and dependency copying

## v0.29.11 (2023-10-03)

### Fix

- more slurm fixes

## v0.29.10 (2023-10-03)

### Fix

- max duration QoS was ignored

## v0.29.9 (2023-10-03)

### Fix

- more fine-grained SLURM configuration
- blank line after job information

## v0.29.8 (2023-10-03)

### Fix

- **slurm**: Better SLURM launcher finder

## v0.29.7 (2023-10-03)

## v0.29.6 (2023-10-03)

### Fix

- cleaned up the instance() mode

## v0.29.5 (2023-10-03)

### Fix

- cleaned up the instance() mode

## v0.29.4 (2023-10-03)

### Fix

- better instance()

## v0.29.3 (2023-10-03)

### Fix

- identifiers

## v0.29.2 (2023-10-03)

### Fix

- pre-task are properly handled

## v0.29.1 (2023-10-03)

### Fix

- exception thrown when adding pre-task to a sealed config

## v0.29.0 (2023-10-03)

### Fix

- pre task dependencies are taken into account

## v0.28.0 (2023-10-03)

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

## v0.27.0 (2023-10-03)

### Feat

- Expose the unwrap function

### Fix

- Removes unnecessary server logs

## v0.26.0 (2023-10-03)

### Fix

- Fix submit hooks (and document them)

## v0.25.0 (2023-10-03)

## v0.24.0 (2023-10-03)

### Feat

- serialized configurations

### Fix

- requirement for fabric
- add gevent-websocket for supporting websockets

### Refactor

- Changed TaskOutput to ConfigWrapper

## v0.23.0 (2023-10-03)

### Feat

- submit hooks to allow e.g. changing the environment variables

## v0.22.0 (2023-10-03)

### Feat

- tags as immutable and hashable dicts

### Fix

- corrected service status update for servers
- improved server

## v0.21.0 (2023-10-03)

### Feat

- When an experiment fails, display the path to stderr
- service proxying

### Fix

- JS vulnerabilities fix
- Information message when locking experiment
- Improving slurm support
- Fix test bugs
- better handlign of services

### Refactor

- **server**: switched to flask and socketio for future evolutions of the server

## v0.20.0 (2023-10-03)

### Feat

- improvements for dry-run modes to show completed jobs

### Refactor

- more reliable identifier computation

## v0.19.2 (2023-10-03)

### Fix

- better identifier recomputation

## v0.19.1 (2023-10-03)

### Fix

- fix bugs with generate/dry-run modes

## v0.19.0 (2023-10-03)

### Feat

- allow using the old task identifier computation to fix params.json

## v0.18.0 (2023-10-03)

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

## v0.16.0 (2023-10-03)

### Feat

- **server**: web services for experiment server

## v0.15.1 (2023-10-03)

### Fix

- wrong indent

## v0.15.0 (2023-10-03)

### Feat

- **scheduler**: foundations for experiment services

## v0.14.6 (2023-10-03)

## v0.14.5 (2023-10-03)

## v0.14.4 (2023-10-03)

## v0.14.3 (2023-10-03)

## v0.14.2 (2023-10-03)

## v0.14.1 (2023-10-03)

## v0.13.3 (2023-10-03)

## v0.13.2 (2023-10-03)

## v0.13.1 (2023-10-03)

## v0.13.0 (2023-10-03)

## v0.12.2 (2023-10-03)

## v0.12.0 (2023-10-03)

## v0.11.8 (2023-10-03)

## v0.11.7 (2023-10-03)

## v0.11.6 (2023-10-03)

## v0.11.5 (2023-10-03)

## v0.11.3 (2023-10-03)

## v0.11.2 (2023-10-03)

## v0.11.1 (2023-10-03)

## v0.10.6 (2023-10-03)

## v0.10.5 (2023-10-03)

## v0.10.4 (2023-10-03)

## v0.10.3 (2023-10-03)

## v0.10.2 (2023-10-03)

## v0.10.1 (2023-10-03)

## v0.10.0 (2023-10-03)

## v0.9.12 (2023-10-03)

## v0.9.11 (2023-10-03)

## v0.9.10 (2023-10-03)

## v0.9.9 (2023-10-03)

## v0.9.8 (2023-10-03)

## v0.9.7 (2023-10-03)

## v0.9.6 (2023-10-03)

## v0.9.5 (2023-10-03)

## v0.9.4 (2023-10-03)

## v0.9.3 (2023-10-03)

## v0.9.2 (2023-10-03)

## v0.9.1 (2023-10-03)

## v0.9.0 (2023-10-03)

## v0.8.9 (2023-10-03)

## v0.8.8 (2023-10-03)

## v0.8.7 (2023-10-03)

## v0.8.6 (2023-10-03)

## v0.8.5 (2023-10-03)

## v0.8.4 (2023-10-03)

## v0.8.3 (2023-10-03)

## v0.8.2 (2023-10-03)

## v0.8.1 (2023-10-03)

## v0.8.0 (2023-10-03)

## v0.7.12 (2023-10-03)

## v0.7.11 (2023-10-03)

## v0.7.10 (2023-10-03)

## v0.7.9 (2023-10-03)

## v0.7.8 (2023-10-03)

## v0.7.7 (2023-10-03)

## v0.7.6 (2023-10-03)

## v0.7.5 (2023-10-03)

## v0.7.4 (2023-10-03)

## v0.7.3 (2023-10-03)

## v0.7.2 (2023-10-03)

## v0.7.0 (2020-05-26)

## v0.6.0 (2023-10-03)

## v0.5.9 (2023-10-03)

## v0.5.7 (2023-10-03)

## v0.5.6 (2023-10-03)

## v0.5.5 (2023-10-03)

## v0.5.4 (2023-10-03)

## v0.5.3 (2023-10-03)

## v0.5.2 (2023-10-03)

## v0.5.1 (2023-10-03)

## v0.5.0 (2023-10-03)

## v0.3.1 (2019-02-22)

## v0.3.0 (2019-02-22)
