import React, { useEffect } from "react";
import { Modal, Tabs, Tab, Button } from "react-bootstrap";

import { Job } from "./reducers";
import { useAppSelector, useNavigate } from "./store";
import client from "./client";
import { TaskDetailBody } from "./TaskDetail";
import { LogPanel } from "./LogViewer";

export type JobTab = "details" | "logs";

const JobModal = ({
  job,
  tab,
  onTabChange,
  onHide,
}: {
  job: Job;
  tab: JobTab;
  onTabChange: (tab: JobTab) => void;
  onHide: () => void;
}) => {
  return (
    <Modal show={true} size="xl" onHide={onHide}>
      <Modal.Header closeButton>
        <Modal.Title>{job.taskId}</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Tabs activeKey={tab} onSelect={(k) => onTabChange((k as JobTab) || "details")}>
          <Tab eventKey="details" title="Details">
            <TaskDetailBody job={job} />
          </Tab>
          <Tab eventKey="logs" title="Logs">
            {tab === "logs" && (
              <LogPanel
                target={{ kind: "job", taskId: job.taskId, jobId: job.jobId }}
              />
            )}
          </Tab>
        </Tabs>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={onHide}>
          Close
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

// The modal of the job named by the route (/.../<task>/<job>[/logs]), shown
// once the job is known
export default () => {
  const navigate = useNavigate();
  const route = useAppSelector((state) => state.db.route);
  const jobs = useAppSelector((state) => state.db.jobs);
  const job = route.job ? jobs.byId[route.job.jobId] : undefined;

  // The details are only sent on request
  useEffect(() => {
    if (job) client.job_details(job.jobId, job.taskId);
  }, [job?.jobId]);

  if (!route.job || !job) return null;
  const tab = route.job.tab;
  return (
    <JobModal
      job={job}
      tab={tab}
      onTabChange={(tab) => navigate({ ...route, job: { ...route.job!, tab } })}
      onHide={() => navigate({ ...route, job: null })}
    />
  );
};
