"""Message classes for TUI inter-widget communication"""

from typing import TYPE_CHECKING, Optional
from textual.message import Message

if TYPE_CHECKING:
    from experimaestro.scheduler.interfaces import JobState


class ExperimentSelected(Message):
    """Message sent when an experiment is selected"""

    def __init__(self, experiment_id: str, run_id: Optional[str] = None) -> None:
        super().__init__()
        self.experiment_id = experiment_id
        self.run_id = run_id


class ExperimentDeselected(Message):
    """Message sent when an experiment is deselected"""

    pass


class JobSelected(Message):
    """Message sent when a job is selected"""

    def __init__(self, job_id: str, experiment_id: str) -> None:
        super().__init__()
        self.job_id = job_id
        self.experiment_id = experiment_id


class JobDeselected(Message):
    """Message sent when returning from job detail view"""

    pass


class ViewJobLogs(Message):
    """Message sent when user wants to view job logs"""

    def __init__(
        self, job_path: str, task_id: str, job_state: Optional["JobState"] = None
    ) -> None:
        super().__init__()
        self.job_path = job_path
        self.task_id = task_id
        self.job_state = job_state


class ViewJobLogsRequest(Message):
    """Message sent when user requests to view logs from jobs table"""

    def __init__(self, job_id: str, experiment_id: str) -> None:
        super().__init__()
        self.job_id = job_id
        self.experiment_id = experiment_id


class DeleteJobRequest(Message):
    """Message sent when user requests to delete a job"""

    def __init__(self, job_id: str, experiment_id: str) -> None:
        super().__init__()
        self.job_id = job_id
        self.experiment_id = experiment_id


class DeleteExperimentRequest(Message):
    """Message sent when user requests to delete an experiment"""

    def __init__(self, experiment_id: str) -> None:
        super().__init__()
        self.experiment_id = experiment_id


class KillJobRequest(Message):
    """Message sent when user requests to kill a running job"""

    def __init__(self, job_id: str, experiment_id: str) -> None:
        super().__init__()
        self.job_id = job_id
        self.experiment_id = experiment_id


class KillExperimentRequest(Message):
    """Message sent when user requests to kill all running jobs in an experiment"""

    def __init__(self, experiment_id: str) -> None:
        super().__init__()
        self.experiment_id = experiment_id


class FilterChanged(Message):
    """Message sent when search filter changes"""

    def __init__(self, filter_fn) -> None:
        super().__init__()
        self.filter_fn = filter_fn


class SearchApplied(Message):
    """Message sent when search filter is applied via Enter"""

    pass


class SizeCalculated(Message):
    """Message sent when a folder size has been calculated"""

    def __init__(self, job_id: str, size: str, size_bytes: int) -> None:
        super().__init__()
        self.job_id = job_id
        self.size = size
        self.size_bytes = size_bytes


class ShowRunsRequest(Message):
    """Message sent when user wants to see experiment runs"""

    def __init__(self, experiment_id: str, current_run_id: Optional[str]) -> None:
        super().__init__()
        self.experiment_id = experiment_id
        self.current_run_id = current_run_id


class RunSelected(Message):
    """Message sent when a run is selected from the runs screen"""

    def __init__(
        self, experiment_id: str, run_id: str, is_current: bool = True
    ) -> None:
        super().__init__()
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.is_current = is_current
