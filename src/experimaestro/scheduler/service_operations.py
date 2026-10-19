"""User operations on services: find, start and stop

Shared by the TUI and the web UI so that they accept and refuse the same
operations and report them with the same messages (see
:mod:`~experimaestro.scheduler.job_operations` for :class:`Outcome`).

Services are identified by their experiment and their id: two experiments can
have services with the same id. Services read from disk (``MockService``) are
recreated as live services on demand (``to_service()``, cached), so they can be
started from any interface, even without a running experiment.
"""

import logging
from typing import TYPE_CHECKING

from experimaestro.scheduler.interfaces import BaseService
from experimaestro.scheduler.job_operations import Outcome
from experimaestro.scheduler.services import ServiceState

if TYPE_CHECKING:
    from experimaestro.scheduler.state_provider import StateProvider

logger = logging.getLogger("xpm.operations")


def find_service(
    provider: "StateProvider", experiment_id: str, service_id: str
) -> BaseService | None:
    """Service of an experiment (current run), if any"""
    return next(
        (s for s in provider.get_services(experiment_id) if s.id == service_id),
        None,
    )


def _state_name(service: BaseService) -> str:
    return service.state.name


def check_start_service(service: BaseService) -> Outcome | None:
    """Refusal to start a service, if any"""
    match _state_name(service):
        case "RUNNING":
            return Outcome("warning", f"Service '{service.id}' is already running")
        case "STARTING":
            return Outcome("warning", f"Service '{service.id}' is already starting")
    return None


def start_service(service: BaseService) -> Outcome:
    """Start a service (blocking until it has a URL)

    Interfaces call :func:`check_start_service` first, then show the service
    as starting (``set_starting``) while this runs. Errors are recorded on the
    service (``set_error``), so that the interfaces show them next to it.
    """
    try:
        live = service.to_service()
        if live is service and service.error:
            # The service could not be recreated (the error is set)
            return Outcome("error", f"Failed to start service: {service.error}")
        # Only web services (and their remote proxies) have a URL to start
        if not hasattr(live, "get_url"):
            return Outcome("warning", f"Service '{service.id}' cannot be started")
        url = live.get_url()
    except Exception as e:
        logger.warning("Service %s could not be started: %s", service.id, e)
        service.set_error(str(e))
        return Outcome("error", f"Failed to start service: {e}")
    return Outcome("info", f"Service started: {url}")


def check_stop_service(service: BaseService) -> Outcome | None:
    """Refusal to stop a service, if any"""
    if _state_name(service) == ServiceState.STOPPED.name:
        return Outcome("warning", f"Service '{service.id}' is not running")
    return None


def stop_service(service: BaseService) -> Outcome:
    """Stop a running service"""
    live = service.to_service()
    if refusal := check_stop_service(live):
        return refusal
    if not hasattr(live, "stop"):
        return Outcome("warning", f"Service '{service.id}' cannot be stopped")
    try:
        live.stop()
    except Exception as e:
        logger.warning("Service %s could not be stopped: %s", service.id, e)
        return Outcome("error", f"Failed to stop service: {e}")
    return Outcome("info", f"Service stopped: {service.id}")
