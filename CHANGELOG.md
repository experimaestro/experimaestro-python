- Jupyter support with `jobmonitor` and `serverwidget`

# 0.9.4

- Generated documentation uses relative links (so it can be moved to any directory)

# 0.9.3

- Task outputs objets to wrap task outputs
- Re-submitting a failed job now works

# 0.8.7

- (Enhancement) Slurm support

# 0.8.6

- (Enhancement) Python 3.9 support

# 0.8.5

- (Enhancement) Dict(ionaries) parameters (keys should be simple types)
- (Fix) Nested configurations dependencies
- (Enhancement) Generated path conflicts are handled
- (Fix) Validate before sealing
- (Fix) Generate values when validation is done
- (Fix) Fixed bug with dependencies of task returning configurations

# 0.8.4

- Improvement to documentation
- Use `Task` as base class instead of `@task`

# 0.8.3

- Possible to use `Config` as base class instead of `@config`
- Value checkers annotations
- Constant values are now properly handled

# 0.8.2

- Alternative annotation for default values (to avoid a bug in e.g. Torch)

# 0.8.1

- Fixes for (un)serialization (through pickle \_\_getnewargs_ex\_\_)
- Full type hint support
- Initial tqdm support

# 0.7.12

- Tasks can access their tags at runtime (e.g. to log hyper-parameters with tensorboard)
- Tasks and configurations can be executed without scheduling (debugging)

# 0.7.11

- NPM packages update (security)

# 0.7.10

- Sub-parameters
- Fixes with file-based tokens
- Fixes with duplicate objects

# 0.7.9

- Attribute `__xpm_default_keep__` can be used to avoid using `@configmethod` for configuration only-classes (e.g. datamaestro)

# 0.7.8

- Fixed dependency token deadlock
- Directory-based tokens (with external token watch)
