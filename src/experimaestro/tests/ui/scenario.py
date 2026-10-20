"""The workspace every UI feature test runs against

A single, made-up workspace served by :class:`ScenarioProvider`, an in-memory
:class:`OfflineStateProvider`. Going through the real provider base class means
both interfaces read the state through the same cached queries as in
production, while the scenario can hold states a filesystem fixture cannot
(running jobs, jobs whose process cannot be killed, ...).

Job directories (and their log files) do exist on disk, under a temporary
directory, so that deleting a job, reading its logs or measuring a folder go
through the real code.
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from experimaestro.scheduler.interfaces import (
    ExperimentJobInformation,
    ExperimentStatus,
    JobState,
)
from experimaestro.scheduler.services import WebService
from experimaestro.scheduler.state_status import JobSubmittedEvent
from experimaestro.scheduler.state_provider import (
    MockExperiment,
    MockJob,
    MockService,
    OfflineStateProvider,
)


@dataclass
class JobSpec:
    identifier: str
    task_id: str
    state: str
    tags: dict[str, str] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    #: Submission order (seconds after the run started)
    submitted: int = 0

    @property
    def scriptname(self) -> str:
        return self.task_id.rsplit(".", 1)[-1]


@dataclass
class RunSpec:
    run_id: str
    status: ExperimentStatus
    jobs: list[JobSpec]


@dataclass
class ExperimentSpec:
    experiment_id: str
    #: Runs, oldest first: the last one is the current run
    runs: list[RunSpec]


# --- The scenario -----------------------------------------------------------

#: Job identifiers, named after what they are for readability (real ones are
#: hashes — nothing in the UIs depends on their shape)
LEARN_DONE = "a1learndone"
LEARN_ERROR = "a2learnerror"
EVAL_DONE = "a3evaldone"
OLD_LEARN = "a0oldlearn"
LEARN_RUNNING = "b1learnrunning"
EVAL_WAITING = "b2evalwaiting"
LEARN_STUCK = "b3learnstuck"
ABLATION_DONE = "d1ablationdone"
ABLATION_NOT_RUN = "d2ablationnotrun"
ORPHAN_DONE = "c1orphandone"
STRAY_RUNNING = "c2strayrunning"

#: A running job whose process cannot be found (kill fails)
UNKILLABLE = {LEARN_STUCK}

#: Size of the data file written in the finished orphan job
ORPHAN_DATA_SIZE = 3000

EXPERIMENTS = [
    ExperimentSpec(
        "baseline",
        [
            RunSpec(
                "20260101_100000",
                ExperimentStatus.DONE,
                [JobSpec(OLD_LEARN, "pkg.Learn", "done", {"model": "bm25"})],
            ),
            RunSpec(
                "20260102_090000",
                ExperimentStatus.DONE,
                [
                    JobSpec(LEARN_DONE, "pkg.Learn", "done", {"model": "bm25"}, [], 0),
                    JobSpec(
                        LEARN_ERROR, "pkg.Learn", "error", {"model": "splade"}, [], 1
                    ),
                    JobSpec(
                        EVAL_DONE,
                        "pkg.Evaluate",
                        "done",
                        {"model": "bm25"},
                        [LEARN_DONE],
                        2,
                    ),
                ],
            ),
        ],
    ),
    ExperimentSpec(
        "sweep",
        [
            RunSpec(
                "20260103_120000",
                ExperimentStatus.RUNNING,
                [
                    JobSpec(
                        LEARN_RUNNING, "pkg.Learn", "running", {"model": "bm25"}, [], 0
                    ),
                    JobSpec(
                        EVAL_WAITING,
                        "pkg.Evaluate",
                        "waiting",
                        {"model": "bm25"},
                        [LEARN_RUNNING],
                        1,
                    ),
                    JobSpec(
                        LEARN_STUCK, "pkg.Learn", "running", {"model": "dense"}, [], 2
                    ),
                ],
            )
        ],
    ),
    # Stopped (e.g. interrupted) before all its jobs could run
    ExperimentSpec(
        "ablation",
        [
            RunSpec(
                "20260104_080000",
                ExperimentStatus.FAILED,
                [
                    JobSpec(ABLATION_DONE, "pkg.Learn", "done", {"model": "tiny"}),
                    JobSpec(
                        ABLATION_NOT_RUN,
                        "pkg.Evaluate",
                        "unscheduled",
                        {"model": "tiny"},
                        [ABLATION_DONE],
                        1,
                    ),
                ],
            )
        ],
    ),
]

ORPHANS = [
    JobSpec(ORPHAN_DONE, "pkg.Learn", "done"),
    JobSpec(STRAY_RUNNING, "pkg.Learn", "running"),
]


class DemoService(WebService):
    """A web service the interfaces can recreate from its stored state, and
    start and stop (it serves nothing)"""

    def __init__(self, id: str, title: str):
        super().__init__()
        self.id = id
        self.title = title

    def description(self):
        return self.title

    def _serve(self, running: threading.Event):
        self.url = f"http://localhost:1/{self.id}"
        running.set()
        while not self.should_stop():
            time.sleep(0.01)


#: Services, by experiment
#: (the same service id in two experiments: services are told apart by experiment)
SERVICES = {
    "sweep": [("tensorboard", "TensorBoard")],
    "baseline": [("tensorboard", "TensorBoard (baseline)")],
}


def stdout_text(job_id: str) -> str:
    return f"stdout of {job_id}\nsecond line\n"


def stderr_text(job_id: str) -> str:
    return f"stderr of {job_id}\n"


def _run_start(run_id: str) -> datetime:
    return datetime.strptime(run_id, "%Y%m%d_%H%M%S")


# --- The provider -----------------------------------------------------------


class ScenarioProvider(OfflineStateProvider):
    """In-memory provider serving :data:`EXPERIMENTS` and :data:`ORPHANS`

    Mutating operations (kill, delete) act on the in-memory state and are
    recorded, so that tests can check what each interface asked for.
    """

    def __init__(self, workspace: Path):
        super().__init__()
        self.workspace = workspace

        self.jobs: dict[str, MockJob] = {}
        #: experiment id -> runs (oldest first)
        self.runs: dict[str, list[MockExperiment]] = {}
        #: (experiment id, run id) -> job ids
        self.run_jobs: dict[tuple[str, str], list[str]] = {}
        self.orphans: list[str] = []
        self.services: dict[str, list[MockService]] = {}

        self.killed: list[str] = []
        self.partials_cleanups = 0

        for xp in EXPERIMENTS:
            self.runs[xp.experiment_id] = []
            for run in xp.runs:
                self._add_run(xp.experiment_id, run)
        for spec in ORPHANS:
            self._add_job(spec)
            self.orphans.append(spec.identifier)
        (self.jobs[ORPHAN_DONE].path / "data.bin").write_bytes(b"x" * ORPHAN_DATA_SIZE)

        for xp_id, services in SERVICES.items():
            run_id = self.runs[xp_id][-1].run_id
            self.services[xp_id] = [
                MockService(
                    service_id,
                    description,
                    {"id": service_id, "title": description},
                    service_class=f"{DemoService.__module__}.DemoService",
                    experiment_id=xp_id,
                    run_id=run_id,
                )
                for service_id, description in services
            ]

    def _add_job(self, spec: JobSpec) -> MockJob:
        path = self.workspace / "jobs" / spec.task_id / spec.identifier
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{spec.scriptname}.out").write_text(stdout_text(spec.identifier))
        (path / f"{spec.scriptname}.err").write_text(stderr_text(spec.identifier))
        job = MockJob(
            identifier=spec.identifier,
            task_id=spec.task_id,
            path=path,
            state=spec.state,
            starttime=None,
            endtime=None,
            progress=[],
            updated_at="2026-01-01T00:00:00",
        )
        self.jobs[spec.identifier] = job
        return job

    def _add_run(self, experiment_id: str, run: RunSpec) -> None:
        start = _run_start(run.run_id)
        job_infos = {}
        for spec in run.jobs:
            self._add_job(spec)
            job_infos[spec.identifier] = ExperimentJobInformation(
                job_id=spec.identifier,
                task_id=spec.task_id,
                tags=spec.tags,
                timestamp=start.timestamp() + spec.submitted,
            )
        states = [spec.state for spec in run.jobs]
        workdir = self.workspace / "experiments" / experiment_id / run.run_id
        workdir.mkdir(parents=True, exist_ok=True)
        self.runs[experiment_id].append(
            MockExperiment(
                workdir,
                run.run_id,
                status=run.status,
                hostname="test-host",
                started_at=start,
                ended_at=None if run.status == ExperimentStatus.RUNNING else start,
                job_infos=job_infos,
                dependencies={s.identifier: s.depends_on for s in run.jobs},
                finished_jobs=states.count("done"),
                failed_jobs=states.count("error"),
            )
        )
        self.run_jobs[(experiment_id, run.run_id)] = [s.identifier for s in run.jobs]

    def submit(self, experiment_id: str, spec: JobSpec) -> None:
        """Submit a job to the current run of an experiment (as it runs)"""
        self._add_job(spec)
        run = self.runs[experiment_id][-1]
        self.run_jobs[(experiment_id, run.run_id)].append(spec.identifier)
        self._notify_state_listeners(
            JobSubmittedEvent(
                job_id=spec.identifier,
                task_id=spec.task_id,
                experiment_id=experiment_id,
                run_id=run.run_id,
            )
        )

    def _run(self, experiment_id: str, run_id: str | None) -> MockExperiment | None:
        runs = self.runs.get(experiment_id)
        if not runs:
            return None
        if run_id is None:
            return runs[-1]
        return next((r for r in runs if r.run_id == run_id), None)

    # --- Read hooks (objects are built up front, the entity caches are unused)

    def _create_job(self, full_id, *args, **kwargs):
        raise NotImplementedError

    def _create_experiment(self, experiment_id, run_id, *args, **kwargs):
        raise NotImplementedError

    def _fetch_experiments(self, since, *, refresh=False):
        return [runs[-1] for runs in self.runs.values()]

    def _fetch_experiment(self, experiment_id, *, refresh=False):
        return self._run(experiment_id, None)

    def _fetch_experiment_runs(self, experiment_id, *, refresh=False):
        return list(self.runs.get(experiment_id, []))

    def _fetch_current_run(self, experiment_id, *, refresh=False):
        run = self._run(experiment_id, None)
        return run.run_id if run else None

    def _fetch_jobs(
        self, experiment_id, run_id, task_id, state, tags, since, *, refresh=False
    ):
        run = self._run(experiment_id, run_id)
        if run is None:
            return []
        jobs = [
            self.jobs[job_id]
            for job_id in self.run_jobs[(experiment_id, run.run_id)]
            if job_id in self.jobs
        ]
        if task_id:
            jobs = [j for j in jobs if j.task_id == task_id]
        if state:
            jobs = [j for j in jobs if j.state.name == state]
        if tags:
            run_tags = run.tags
            jobs = [
                j
                for j in jobs
                if all(
                    run_tags.get(j.identifier, {}).get(k) == v for k, v in tags.items()
                )
            ]
        return jobs

    def _fetch_job(self, task_id, job_id, *, refresh=False):
        job = self.jobs.get(job_id)
        return job if job is not None and job.task_id == task_id else None

    def _fetch_all_jobs(self, state, tags, since, *, refresh=False):
        return list(self.jobs.values())

    def _fetch_tags_map(self, experiment_id, run_id, *, refresh=False):
        run = self._run(experiment_id, run_id)
        return run.tags if run else {}

    def _fetch_dependencies_map(self, experiment_id, run_id, *, refresh=False):
        run = self._run(experiment_id, run_id)
        return run.dependencies if run else {}

    def _fetch_experiment_job_info(self, experiment_id, run_id, *, refresh=False):
        run = self._run(experiment_id, run_id)
        return run.job_infos if run else {}

    def _fetch_run_objects(self, experiment_id, run_id):
        # One serialized object per job, referring to its dependencies
        run = self._run(experiment_id, run_id)
        if run is None:
            raise FileNotFoundError(f"No run {run_id} for {experiment_id}")
        oids = {job_id: oid for oid, job_id in enumerate(run.job_infos)}
        return [
            {
                "id": oid,
                "module": "pkg",
                "type": run.job_infos[job_id].task_id.rsplit(".", 1)[-1],
                "identifier": job_id,
                "fields": {
                    "inputs": [
                        {"type": "python", "value": oids[dep]}
                        for dep in run.dependencies.get(job_id, [])
                    ]
                },
            }
            for job_id, oid in oids.items()
        ]

    def _fetch_orphan_jobs(self, *, refresh=False, **kwargs):
        return [self.jobs[job_id] for job_id in self.orphans if job_id in self.jobs]

    def _fetch_services_from_storage(self, experiment_id, run_id):
        if experiment_id is None:
            return [s for services in self.services.values() for s in services]
        return list(self.services.get(experiment_id, []))

    # --- Operations

    def kill_job(self, job, perform=False):
        if not perform:
            return job.state.running()
        if job.identifier in UNKILLABLE:
            raise RuntimeError("No process found for job")
        self.killed.append(job.identifier)
        job.state = JobState.ERROR
        return True

    def clean_job(self, job, perform=False):
        if job.state and not (job.state.finished() or job.state.is_unscheduled()):
            return False
        if perform:
            import shutil

            shutil.rmtree(job.path)
            del self.jobs[job.identifier]
            self.invalidate_caches()
        return True

    def delete_experiment(self, experiment_id, delete_jobs=False, perform=True):
        if experiment_id not in self.runs:
            return False, f"Experiment {experiment_id} not found"
        if perform:
            del self.runs[experiment_id]
            self.invalidate_caches()
        return True, f"Experiment {experiment_id} deleted"

    def cleanup_orphan_partials(self, perform=False):
        if perform:
            self.partials_cleanups += 1
        return []

    def close(self):
        pass
