import React, { useState } from "react";
import { Modal, Tabs, Tab, Button } from "react-bootstrap";

import { Job } from "./reducers";
import { TaskDetailBody } from "./TaskDetail";
import { LogPanel } from "./LogViewer";

export type JobTab = "details" | "logs";

export default ({
  job,
  initialTab = "details",
  onHide,
}: {
  job: Job;
  initialTab?: JobTab;
  onHide: () => void;
}) => {
  const [tab, setTab] = useState<JobTab>(initialTab);

  return (
    <Modal show={true} size="xl" onHide={onHide}>
      <Modal.Header closeButton>
        <Modal.Title>{job.taskId}</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Tabs activeKey={tab} onSelect={(k) => setTab((k as JobTab) || "details")}>
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
