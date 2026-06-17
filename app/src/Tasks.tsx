import React, { useState, useMemo } from "react";
import { Button, Modal, Form, Table } from "react-bootstrap";

import TaskJobs from "./TaskJobs";
import TaskDetail from "./TaskDetail";
import LogViewer, { LogTarget } from "./LogViewer";
import client from "./client";
import { Job } from "./reducers";
import { useMessages } from "./ui/messages";
import { useAppSelector } from "./store";

type TagFilter = {
  tag: string;
  value: string;
};

type SortKey = "status" | "task" | "date";

const statusRank = (status: string): number => {
  switch (status) {
    case "running":
      return 5;
    case "error":
    case "waiting":
      return 2;
    case "ready":
      return 1;
    default:
      return 0;
  }
};

export default () => {
  const [tagFilters, setTagFilters] = useState<TagFilter[]>([]);
  const [taskFilter, setTaskFilter] = useState<string>();
  const [showJob, setShowJob] = useState<string>();
  const [confirm, setConfirm] = useState<{ job: Job; kind: "kill" | "delete" }>();
  const [logTarget, setLogTarget] = useState<LogTarget>();
  const [sortKey, setSortKey] = useState<SortKey>("status");
  const [sortAsc, setSortAsc] = useState<boolean>(false);
  const { info } = useMessages();
  const jobs = useAppSelector((state) => state.db.jobs);
  const currentExperiment = useAppSelector((state) => state.db.currentExperiment);

  function updateTagSearch(tag: string) {
    let re = /(\S+):(?:([^"]\S*)|"([^"]+)")\s*/g;
    var match: any[] | null;
    const tagFilters = [];
    while ((match = re.exec(tag)) !== null) {
      tagFilters.push({ tag: match[1], value: match[2] });
    }
    setTagFilters(tagFilters);
  }

  const filteredJobs = useMemo(() => {
    let re_task: null | RegExp;
    try {
      re_task = taskFilter ? new RegExp(taskFilter) : null;
    } catch {
      re_task = null;
    }

    function filter(job: Job) {
      // Filter by experiment if one is selected
      if (currentExperiment !== null) {
        if (!job.experimentIds || !job.experimentIds.includes(currentExperiment)) {
          return false;
        }
      }

      if (re_task && job.taskId.match(re_task) === null) return false;

      mainloop: for (let { tag, value } of tagFilters) {
        for (let tv of job.tags) {
          if (tv[0].search(tag) !== -1 && tv[1].toString().search(value) !== -1)
            continue mainloop;
        }
        return false;
      }

      return true;
    }

    const result = jobs.ids.map((id) => jobs.byId[id]).filter(filter);

    const cmp = (a: Job, b: Job): number => {
      let z = 0;
      switch (sortKey) {
        case "status":
          z = statusRank(a.status) - statusRank(b.status);
          if (z === 0) z = a.jobId.localeCompare(b.jobId);
          break;
        case "task":
          z = a.taskId.localeCompare(b.taskId);
          break;
        case "date":
          z = (a.submitted || "").localeCompare(b.submitted || "");
          break;
      }
      return sortAsc ? z : -z;
    };

    return result.sort(cmp);
  }, [jobs.ids, jobs.byId, tagFilters, taskFilter, currentExperiment, sortKey, sortAsc]);

  function cancelConfirm() {
    setConfirm(undefined);
    info("Action cancelled");
  }

  function runConfirm() {
    if (!confirm) return;
    const { job, kind } = confirm;
    if (kind === "kill") client.job_kill(job.jobId, job.taskId);
    else client.job_delete(job.jobId, job.taskId);
    setConfirm(undefined);
  }

  return (
    <div id="resources">
      <>
        <div className="search">
          <div style={{ display: "flex", alignItems: "flex-end", gap: "0.5rem" }}>
            <Form.Group>
              <Form.Label htmlFor="searchtask">Task</Form.Label>
              <Form.Control
                id="searchtask"
                onChange={(e) => setTaskFilter(e.target.value)}
                placeholder="Filter task"
              />
            </Form.Group>
            <Form.Group>
              <Form.Label htmlFor="searchtags">Tags</Form.Label>
              <Form.Control
                id="searchtags"
                onChange={(e) => updateTagSearch(e.target.value)}
                placeholder="Format tag:value..."
              />
            </Form.Group>
            <Form.Group>
              <Form.Label htmlFor="sortkey">Sort</Form.Label>
              <Form.Select
                id="sortkey"
                size="sm"
                value={sortKey}
                onChange={(e) => setSortKey(e.target.value as SortKey)}
              >
                <option value="status">Status</option>
                <option value="task">Task</option>
                <option value="date">Submitted</option>
              </Form.Select>
            </Form.Group>
            <Button
              size="sm"
              variant="outline-secondary"
              title="Toggle sort direction"
              onClick={() => setSortAsc((v) => !v)}
            >
              <i className={`fas fa-arrow-${sortAsc ? "up" : "down"}`} />
            </Button>
          </div>
        </div>

        {confirm && (
          <Modal show={true} onHide={cancelConfirm}>
            <Modal.Header closeButton>
              <Modal.Title>Are you sure?</Modal.Title>
            </Modal.Header>
            <Modal.Body>
              {confirm.kind === "kill" ? "Kill" : "Delete"} job{" "}
              <b>{confirm.job.taskId}</b>?
              {confirm.kind === "delete" && (
                <div className="text-danger small mt-2">
                  This removes the job directory from disk.
                </div>
              )}
            </Modal.Body>
            <Modal.Footer>
              <Button variant="secondary" onClick={cancelConfirm}>
                Cancel
              </Button>
              <Button
                variant={confirm.kind === "delete" ? "danger" : "primary"}
                onClick={runConfirm}
              >
                {confirm.kind === "kill" ? "Kill" : "Delete"}
              </Button>
            </Modal.Footer>
          </Modal>
        )}

        <Table hover responsive className="align-middle jobs-table">
          <thead>
            <tr>
              <th>Status</th>
              <th>Task</th>
              <th>Tags</th>
              <th>Submitted</th>
              <th>Duration</th>
              <th>CO₂</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {filteredJobs.map((job) => (
              <TaskJobs
                key={job.jobId}
                job={job}
                onKill={() => setConfirm({ job, kind: "kill" })}
                onDelete={() => setConfirm({ job, kind: "delete" })}
                onShow={() => {
                  client.job_details(job.jobId, job.taskId);
                  setShowJob(showJob === job.jobId ? undefined : job.jobId);
                }}
                onLogs={() =>
                  setLogTarget({
                    kind: "job",
                    taskId: job.taskId,
                    jobId: job.jobId,
                    title: job.taskId,
                  })
                }
              />
            ))}
          </tbody>
        </Table>
        {filteredJobs.length === 0 && (
          <p className="text-muted">No jobs match the current filter.</p>
        )}
        {showJob && (
          <TaskDetail onHide={() => setShowJob("")} job={jobs.byId[showJob]} />
        )}
        {logTarget && (
          <LogViewer target={logTarget} onHide={() => setLogTarget(undefined)} />
        )}
      </>
    </div>
  );
};
