"""Actions of an experiment run, shared by the TUI and the web UI

Actions are declared by a run: each run of an experiment has its own (recorded
in its ``status.json``, their configuration in its ``objects.jsonl``). The
interfaces list and execute the actions of the run they show, the current run
by default; executing the action of a former run is allowed, but the
interfaces ask for a confirmation first (:data:`PAST_RUN_WARNING`).
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from experimaestro.scheduler.interfaces import BaseAction, BaseExperiment
    from experimaestro.scheduler.state_provider import StateProvider

#: Shown (with the run id) before executing the action of a former run
PAST_RUN_WARNING = (
    "This action was declared by a former run of the experiment ({run_id}), "
    "not by the current one: it is executed with the configuration of that "
    "run, which may be outdated (e.g. jobs or files that were replaced since)."
)


def is_past_run(
    provider: "StateProvider", experiment_id: str, run_id: str | None
) -> bool:
    """Whether the run is a former run (None: the current run)"""
    return run_id is not None and run_id != provider.get_current_run(experiment_id)


def run_experiment(
    provider: "StateProvider", experiment_id: str, run_id: str | None
) -> "BaseExperiment | None":
    """The experiment run (the current run when None)"""
    if not is_past_run(provider, experiment_id, run_id):
        return provider.get_experiment(experiment_id)
    return next(
        (
            run
            for run in provider.get_experiment_runs(experiment_id)
            if run.run_id == run_id
        ),
        None,
    )


def run_actions(
    provider: "StateProvider", experiment_id: str, run_id: str | None = None
) -> dict[str, "BaseAction"]:
    """Actions declared by a run (the current run when None)"""
    run = run_experiment(provider, experiment_id, run_id)
    return dict(run.actions) if run is not None else {}


def load_action(
    provider: "StateProvider",
    experiment_id: str,
    action_id: str,
    run_id: str | None = None,
) -> Any:
    """The executable action of a run (the current run when None)

    The live action when the experiment runs in this process, otherwise the
    action loaded from the run's serialized objects.

    Raises:
        ValueError: if the run or the action does not exist
    """
    run = run_experiment(provider, experiment_id, run_id)
    if run is None:
        raise ValueError(f"Run {run_id} of experiment {experiment_id} not found")

    action = run.actions.get(action_id)
    if action is not None and hasattr(type(action), "execute"):
        return action

    try:
        xp_info = provider.load_xp_info(experiment_id, run.run_id)
    except NotImplementedError:
        from experimaestro.core.serialization import load_xp_info

        xp_info = load_xp_info(run.workdir)
    if action_id not in xp_info.actions:
        raise ValueError(f"Action '{action_id}' not found")
    return xp_info.actions[action_id]
