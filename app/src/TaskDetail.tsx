import { DateTime } from "luxon";
import { copyToClibpoard } from "./clipboard";
import { createUseStyles } from "react-jss";
import React from "react";
import { Job } from "./reducers";
import { ProgressBar } from "react-bootstrap";
import { useMessages } from "./ui/messages";
import { useAppSelector } from "./store";
import Modal from "react-bootstrap/Modal";

function formatms(t: undefined | string) {
  if (t) {
    const date = DateTime.fromISO(t);
    return date.toLocaleString(DateTime.DATETIME_FULL_WITH_SECONDS);
  }
  return "N/A";
}

function formatDuration(start?: string, end?: string): string {
  if (!start) return "N/A";
  const startDt = DateTime.fromISO(start);
  const endDt = end ? DateTime.fromISO(end) : DateTime.now();
  const secs = endDt.diff(startDt, "seconds").seconds;
  if (isNaN(secs) || secs < 0) return "N/A";
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  return h > 0 ? `${h}h ${m}m ${s}s` : m > 0 ? `${m}m ${s}s` : `${s}s`;
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
    rowGap: "4px",
  },
  what: {
    fontWeight: "bold",
  },
});

export const TaskDetailBody = ({ job }: { job: Job }) => {
  const classes = useStyles();
  const { success, error } = useMessages();
  const allJobs = useAppSelector((state) => state.db.jobs);

  const carbon = job.carbon;
  const proc = job.process;

  const dependencies = (job.dependsOn || []).map((depId) => {
    const depJob = allJobs.byId[depId];
    return {
      jobId: depId,
      taskId: depJob?.taskId || depId.substring(0, 12),
      status: depJob?.status || "unknown",
    };
  });

  return (
    <div className={classes.details}>
          <span className={classes.what}>Status</span>
          <div>
            {job.status}
            {job.failureReason ? ` (${job.failureReason})` : ""}
            {job.schedulerState && job.schedulerState !== job.status
              ? ` — scheduler: ${job.schedulerState}`
              : ""}
          </div>

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

          <span className={classes.what}>Submitted</span>
          <div>{formatms(job.submitted)}</div>

          <span className={classes.what}>Start</span>
          <div>{formatms(job.start)}</div>

          <span className={classes.what}>End</span>
          <div>{formatms(job.end)}</div>

          <span className={classes.what}>Duration</span>
          <div>{formatDuration(job.start, job.end)}</div>

          {(job.exitCode !== null && job.exitCode !== undefined) ||
          (job.retryCount ?? 0) > 0 ? (
            <>
              <span className={classes.what}>Exit / retries</span>
              <div>
                exit {job.exitCode ?? "—"}, retries {job.retryCount ?? 0}
              </div>
            </>
          ) : null}

          {proc && (
            <>
              <span className={classes.what}>Process</span>
              <div>
                {proc.running ? "running" : "not running"}
                {proc.pid != null ? `, pid ${proc.pid}` : ""}
                {proc.type ? ` (${proc.type})` : ""}
                {proc.cpuPercent != null ? `, CPU ${proc.cpuPercent.toFixed(1)}%` : ""}
                {proc.memoryMb != null ? `, ${proc.memoryMb.toFixed(0)} MB` : ""}
                {proc.numThreads != null ? `, ${proc.numThreads} threads` : ""}
              </div>
            </>
          )}

          {carbon && (
            <>
              <span className={classes.what}>Carbon</span>
              <div>
                <b>{(carbon.co2kg ?? 0).toFixed(4)} kg CO₂</b>
                {carbon.energyKwh != null
                  ? `, ${carbon.energyKwh.toFixed(4)} kWh`
                  : ""}
                {(carbon.cpuPowerW ?? 0) +
                  (carbon.gpuPowerW ?? 0) +
                  (carbon.ramPowerW ?? 0) >
                0
                  ? ` — CPU ${carbon.cpuPowerW?.toFixed(0)}W / GPU ${carbon.gpuPowerW?.toFixed(
                      0,
                    )}W / RAM ${carbon.ramPowerW?.toFixed(0)}W`
                  : ""}
                {carbon.region ? ` [${carbon.region}]` : ""}
                {carbon.isFinal ? " (final)" : " (live)"}
              </div>
            </>
          )}

          <span className={classes.what}>Tags</span>
          <div>
            {job.tags.map((tag) => (
              <span key={tag[0]} className="tag">
                <span className="name">{tag[0]}</span>
                <span className="value">{tag[1]}</span>
              </span>
            ))}
          </div>

          {dependencies.length > 0 && (
            <>
              <span className={classes.what}>Dependencies</span>
              <div className="dependencies">
                {dependencies.map((dep) => (
                  <span
                    key={dep.jobId}
                    className={`dependency status-${dep.status}`}
                    title={`${dep.taskId} (${dep.status})`}
                  >
                    {dep.taskId.split(".").pop()}
                  </span>
                ))}
              </div>
            </>
          )}

          {job.progress && job.progress.length > 0 && (
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
  );
};

export default ({ job, onHide }: { job: Job; onHide: () => void }) => {
  return (
    <Modal show={true} size="xl" onHide={onHide}>
      <Modal.Header closeButton>
        <Modal.Title>Job details</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <TaskDetailBody job={job} />
      </Modal.Body>
    </Modal>
  );
};
