from experimaestro.core.objects import ConfigInformation, TypeConfig
from experimaestro.utils import logger
import experimaestro.taskglobals as taskglobals
from pathlib import Path
import json


def fix_deprecated(workpath: Path, fix: bool):
    jobspath = workpath / "jobs"
    logger.info("Looking for deprecated jobs in %s", jobspath)
    for job in jobspath.glob("*/*/params.json"):

        logger.info("Looking up at %s", job.parent)

        # If link, skip
        if job.parent.is_symlink():
            logger.debug("... it is a symlink - skipping")
            continue

        # Unserialize
        logger.debug("Loading configuration %s", job.parent)
        params = json.loads(job.resolve().read_text())
        taskglobals.Env.instance().wspath = Path(params["workspace"])

        try:
            object = ConfigInformation.fromParameters(params["objects"], False)
        except:
            logger.exception("Error while loading the parameters from %s", job)
            continue

        # Now, computes the signature
        old_identifier = job.parent.name
        new_identifier = str(object.__xpm__.identifier.all.hex())

        if new_identifier != old_identifier:
            logger.info(
                "Configuration %s (%s) has a new identifier %s (%s)",
                job.parents[1].name,
                old_identifier,
                object.__xpmtype__.identifier,
                new_identifier,
            )

            if fix:
                oldjobpath = jobspath / job.parents[1].name / old_identifier
                newjobpath = (
                    jobspath / str(object.__xpmtype__.identifier) / new_identifier
                )
                newjobpath.parent.mkdir(exist_ok=True)

                if newjobpath.exists():
                    if newjobpath.resolve() != oldjobpath.resolve():
                        logger.warning(
                            "New job path %s exists and is set to a different value (%s) than the computed one (%s)",
                            newjobpath,
                            newjobpath.resolve(),
                            oldjobpath.resolve(),
                        )
                else:
                    logger.info("Fixing %s/%s", job.parents[1].name, old_identifier)
                    newjobpath.symlink_to(oldjobpath)
