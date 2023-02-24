from experimaestro.core.context import SerializationContext
from experimaestro.core.objects import ConfigInformation
from experimaestro.utils import logger
import experimaestro.taskglobals as taskglobals
from pathlib import Path
import json


def load_job(job_path: Path, discard_id=True):
    logger.info("Loading configuration %s", job_path.parent)
    params = json.loads(job_path.resolve().read_text())
    taskglobals.Env.instance().wspath = Path(params["workspace"])

    try:
        return params, ConfigInformation.fromParameters(
            params["objects"], False, discard_id=discard_id
        )
    except Exception:
        logger.exception("Error while loading the parameters from %s", job_path)
        return None, None


def fix_deprecated(workpath: Path, fix: bool, cleanup: bool):
    jobspath = workpath / "jobs"
    logger.info("Looking for deprecated jobs in %s", jobspath)

    if cleanup:
        for job_path in jobspath.glob("*/*/params.json"):
            # If link, skip
            if job_path.parent.is_symlink():
                job_path.parent.unlink()
                logger.info("Removing symlink %s", job_path.parent)

    for job_path in jobspath.glob("*/*/params.json"):
        # If link, skip
        if job_path.parent.is_symlink():
            logger.debug("... it is a symlink - skipping")
            continue

        params, job = load_job(job_path)
        if job is None:
            continue

        # Now, computes the old  signature
        name = job_path.parents[1].name
        old_identifier: str = job_path.parent.name
        new_identifier: str = job.__xpm__.identifier.all.hex()

        if new_identifier != old_identifier:
            logger.info(
                "Configuration %s (%s) has a new identifier %s (%s)",
                name,
                old_identifier,
                job.__xpmtype__.identifier,
                new_identifier,
            )

            if fix:
                oldjobpath = jobspath / name / old_identifier
                newjobpath = jobspath / str(job.__xpmtype__.identifier) / new_identifier
                newjobpath.parent.mkdir(exist_ok=True)

                # Remove the old symlink if dandling
                if newjobpath.is_symlink() and not newjobpath.exists():
                    newjobpath.unlink()

                if newjobpath.exists():
                    if newjobpath.resolve() != oldjobpath.resolve():
                        logger.warning(
                            "New job path %s exists and is set to a "
                            "different value (%s) than the computed one (%s)",
                            newjobpath,
                            newjobpath.resolve(),
                            oldjobpath.resolve(),
                        )
                else:
                    logger.info("Fixing %s/%s", name, old_identifier)
                    if cleanup:
                        # Rewrite params.json
                        params["objects"] = job.__xpm__.__get_objects__(
                            [], SerializationContext()
                        )
                        tmppath = job_path.with_suffix(".json.tmp")
                        with tmppath.open("wt") as out:
                            json.dump(params, out)
                        tmppath.replace(job_path)

                        # Rename the folder
                        oldjobpath.rename(newjobpath)
                    else:
                        newjobpath.symlink_to(oldjobpath)
