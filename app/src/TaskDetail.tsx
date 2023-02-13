import { DateTime } from "luxon";
import { copyToClibpoard } from "./clipboard";
import { createUseStyles } from "react-jss";
import React from "react";
import { Job } from "./reducers";
import { ProgressBar } from "react-bootstrap";
import { useMessages } from "./ui/messages";
import Modal from "react-bootstrap/Modal";

function formatms(t: undefined | string) {
  if (t) {
    const date = DateTime.fromISO(t)
    console.log(t, date)
    return date.toLocaleString(
      DateTime.DATETIME_FULL_WITH_SECONDS
      );
  }
  return "N/A";
}

const useStyles = createUseStyles({
  level: {
    fontWeight: "bold",
  },
  progress_details: {
    maxWidth: "300px",
  },
  details: {
    display: "grid",
    gridTemplateColumns: "1fr 3fr",
    padding: "5px",
    margin: "5px",
  },
  what: {
    fontWeight: "bold"
  }
});

export default ({ job, onHide }: { job: Job, onHide: () => void }) => {
  const classes = useStyles();
  const { success, error } = useMessages();

  console.log("Showing detail of", job);

  return <Modal show={true} size="xl" onHide={onHide}>
    <Modal.Header>
      <Modal.Title>Job details</Modal.Title>
    </Modal.Header>

    <Modal.Body>

    <div className={classes.details}>
      <span className={classes.what}>Status</span>
      <div>{job.status}</div>
      <span className={classes.what}>Path</span>
      <div>
        <span
          className="clipboard"
          onClick={() =>
            copyToClibpoard(job.locator)
            .then(() => success("Job path copied"))
              .catch(() => error("Error when copying job path"))
          }
        >
          {job.locator}
        </span>
      </div>
      <span className={classes.what}>Submitted</span>{" "}
      <div>{formatms(job.submitted)} </div>

      <span className={classes.what}>Start</span> <div>{formatms(job.start)}</div>

      <span className={classes.what}>End</span> <div>{formatms(job.end)}</div>

      <span className={classes.what}>Tags</span>

      <div>
        {job.tags.map((tag) => (
          <span key={tag[0]} className="tag">
            <span className="name">{tag[0]}</span>
            <span className="value">{tag[1]}</span>
          </span>
        ))}
      </div>
      {job.progress && (
        <>
          <span className={classes.what}>Progress</span>
          <div className={classes.progress_details}>
            {job.progress.map((level, ix) => (
              <div key={ix}>
                <div className="level-desc">{level.desc || ""}</div>
                <div>
                  <ProgressBar
                    striped
                    variant="success"
                    now={level.progress * 100}
                    label={`${Math.trunc(level.progress * 1000) / 10}%`}
                  />
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>

      </Modal.Body>
    </Modal>

};
