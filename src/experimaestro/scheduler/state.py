from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Optional, Type
from experimaestro import Task

from experimaestro.core.context import SerializationContext
from experimaestro.scheduler.jobs import Job, JobDependency
from experimaestro.settings import find_workspace
from experimaestro.core.serialization import from_state_dict, save_definition


@dataclass
class JobInformation:
    id: str
    path: Path
    task: Task
    tags: dict[str, str]
    depends_on: list["JobInformation"] = field(default_factory=list)

    def __post_init__(self):
        self.path = Path(self.path)


class ExperimentState:
    def __init__(self, workdir: Path, name: str, run_id: str = None):
        """Load experiment state from disk.

        Args:
            workdir: Workspace path
            name: Experiment name
            run_id: Run ID to load (default: latest run)
        """
        exp_base = workdir / "experiments" / name
        if run_id is None:
            # Find the latest run directory
            run_dirs = sorted(
                [d for d in exp_base.iterdir() if d.is_dir()],
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
            if not run_dirs:
                raise FileNotFoundError(f"No runs found for experiment {name}")
            run_dir = run_dirs[0]
        else:
            run_dir = exp_base / run_id

        path = run_dir / "state.json"
        with path.open("rt") as fh:
            content = json.load(fh)

        self.states: dict[str, JobInformation] = {
            state_dict["id"]: JobInformation(**state_dict)
            for state_dict in from_state_dict(content, as_instance=False)
        }

        for state in self.states.values():
            state.depends_on = [self.states[key] for key in state.depends_on]

    def get_jobs(self, task_class: Type[Task]) -> list[JobInformation]:
        if task_class is None:
            return list(self.data.values())

        tasks = []
        for job_state in self.states.values():
            if isinstance(job_state.task, task_class):
                tasks.append(job_state)
        return tasks

    @staticmethod
    def save(path: Path, jobs: Iterable[Job]):
        save_definition(
            [
                {
                    "id": job.identifier,
                    "path": str(job.path),
                    "task": job.config,
                    "tags": job.config.__xpm__.tags(),
                    "depends_on": list(
                        dep.origin.identifier
                        for dep in job.dependencies
                        if isinstance(dep, JobDependency)
                    ),
                }
                for job in jobs
            ],
            SerializationContext(),
            path,
        )


def get_experiment(
    name: str, *, workspace: Optional[str] = None, workdir: Optional[Path] = None
) -> ExperimentState:
    ws = find_workspace(workspace=workspace, workdir=workdir)
    return ExperimentState(ws.path, name)
