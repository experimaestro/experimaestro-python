# Walkthrough

This walkthrough is based on the [experimaestro-demo](https://github.com/experimaestro/experimaestro-demo) repository, which provides a complete working example of using experimaestro for hyperparameter search in deep learning.

:::{tip} Running the walkthrough
To follow along, clone the demo repository:
```bash
git clone https://github.com/experimaestro/experimaestro-demo.git
cd experimaestro-demo
```
:::

---

```{include} demo/README.md
:start-after: <!-- doc:start -->
:end-before: <!-- doc:end -->
```

---

## Going further

These features extend the demo and are worth knowing once the basic flow above is comfortable:

- **Long-running / preemptible jobs** — wrap `Learn` as a [`ResumableTask`](experiments/task.md), check `remaining_time()` and raise `GracefulTimeout` to checkpoint cleanly when a SLURM walltime is about to expire.
- **Shared workspace settings** — `imports:` in `settings.yaml` lets multiple project-level YAML files inherit a base workspace block. See [Settings](settings.md).
- **Archiving completed jobs** (beta) — declare auxiliary `folders` with `mode: backup` (or `move`) on a workspace to automatically copy/migrate finished job directories to long-term storage. See [Settings](settings.md).
- **Moving experiments across workspaces** — `experimaestro experiments copy` transfers a finished experiment (and its job directories) between workspaces, useful when promoting a laptop prototype to a cluster run. See [CLI](cli.md).

## Reference

- [Configurations](experiments/config.md) — deep dive into configuration objects
- [Tasks](experiments/task.md) — task definition, ResumableTask, dynamic outputs
- [Launchers](launchers/index.md) — direct, SLURM, OAR launchers and the requirement DSL
- [Settings](settings.md) — workspaces, triggers, imports, folders
- [CLI](cli.md) — command-line interface reference
