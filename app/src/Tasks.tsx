import React, { useState, useMemo } from "react";
import { Button, Modal, Form } from "react-bootstrap";

import TaskJobs from "./TaskJobs";
import TaskDetail from "./TaskDetail";
import client from "./client";
import { Job } from "./reducers";
import { useMessages } from "./ui/messages";
import { useAppSelector } from "./store";

type TagFilter = {
  tag: string;
  value: string;
};
export default () => {
  const [tagFilters, setTagFilters] = useState<TagFilter[]>([]);
  const [taskFilter, setTaskFilter] = useState<string>();
  const [showJob, setShowJob] = useState<string>();
  const [killJob, setKillJob] = useState<Job>();
  const { info } = useMessages();
  const jobs = useAppSelector((state) => state.db.jobs);

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

    return jobs.ids.map((id) => jobs.byId[id]).filter(filter);
  }, [jobs.ids, jobs.byId, tagFilters, taskFilter]);

  function cancelKill() {
    setKillJob(undefined);
    info("Action cancelled");
  }

  function kill() {
    if (killJob) {
      client.send(
        { type: "kill", payload: killJob.jobId },
        "cannot kill job " + killJob.jobId
      );
      setKillJob(undefined);
    }
  }

  return (
    <div id="resources">
      <>
        <div className="search">
          <div style={{ display: "flex" }}>
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
          </div>
        </div>

        {killJob && (
          <Modal show={true}>
            <Modal.Header>Are you sure?</Modal.Header>
            <Modal.Body>
              Are you sure to kill job <b>{killJob.taskId}</b>?
            </Modal.Body>
            <Modal.Footer>
              <Button onClick={cancelKill}>Cancel</Button>
              <Button onClick={kill}>OK</Button>
            </Modal.Footer>
          </Modal>
        )}

        {/* {showJob == jobId && <TaskDetail job={job}/>} */}

        {filteredJobs.map((job) => (
          <TaskJobs
            key={job.jobId}
            job={job}
            onKill={() => setKillJob(job)}
            onShow={() => {
              console.log("Showing", job.jobId, showJob);
              client.send({
                type: "details",
                payload: job.jobId
              })
              setShowJob(showJob === job.jobId ? undefined : job.jobId);
            }}
          />
        ))}
        {showJob && <TaskDetail onHide={() => setShowJob("")} job={jobs.byId[showJob]} />}
      </>
    </div>
  );
};
