import { DateTime } from "luxon";
import { copyToClibpoard } from "./clipboard";
import { createUseStyles } from "react-jss";
import React from "react";
import { Job } from "./reducers";
import { ProgressBar } from "react-bootstrap";

function formatms(t: undefined | number) {
  if (t)
    return DateTime.fromMillis(1000 * t).toLocaleString(
      DateTime.DATETIME_FULL_WITH_SECONDS
    );
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
    border: "1px solid black",
    display: "grid",
    gridTemplateColumns: "1fr 3fr",
    padding: "5px",
    margin: "5px",
  },
});
export default ({ job }: { job: Job }) => {
  const classes = useStyles();
  console.log("Showing detail of", job);

  return (
    <div className={classes.details}>
      <span className="what">Status</span>
      <div>{job.status}</div>
      <span className="what">Path</span>
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
      <span className="what">Submitted</span>{" "}
      <div>{formatms(job.submitted)} </div>
      <span className="what">Start</span> <div>{formatms(job.start)}</div>
      <span className="what">End</span> <div>{formatms(job.end)}</div>
      <span className="what">Tags</span>
      <div>
        {job.tags.map((tag) => (
          <span className="tag">
            <span className="name">{tag[0]}</span>
            <span className="value">{tag[1]}</span>
          </span>
        ))}
      </div>
      {job.progress && (
        <>
          <span className="what">Progress</span>
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
  );
};
