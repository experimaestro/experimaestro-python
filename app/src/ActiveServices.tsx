import React, { useState } from "react";
import { Badge, Button } from "react-bootstrap";

import { useAppSelector, useNavigate } from "./store";
import client from "./client";
import LogViewer, { LogTarget } from "./LogViewer";
import { linkProps, route } from "./routes";

// States in which a service is shown on every page
const ACTIVE_STATES: { [state: string]: string } = {
  STARTING: "info",
  RUNNING: "success",
};

/** The services starting or running (of all the experiments), on every page */
export const ActiveServices = () => {
  const services = useAppSelector((state) => state.db.services);
  const navigate = useNavigate();
  const [logTarget, setLogTarget] = useState<LogTarget>();

  const active = services.ids
    .map((key) => services.byId[key])
    .filter((svc) => svc.state in ACTIVE_STATES);
  if (active.length === 0 && !logTarget) return null;

  return (
    <div className="active-services">
      <span className="text-muted small me-2">Services</span>
      {active.map((svc) => {
        const { id, experimentId } = svc;
        return (
          <span
            key={svc.key}
            className="active-service"
            data-service-id={id}
            data-experiment-id={experimentId}
          >
            <Badge bg={ACTIVE_STATES[svc.state]} className="me-1">
              {svc.state.toLowerCase()}
            </Badge>
            <a
              className="text-muted"
              title="Services of the experiment"
              {...linkProps(route("services", experimentId), navigate)}
            >
              {experimentId}
            </a>
            {" / "}
            {svc.state === "RUNNING" ? (
              <a
                href={`/services/${encodeURIComponent(experimentId)}/${encodeURIComponent(id)}/`}
                target="_blank"
                rel="noreferrer"
                title={svc.description || "Open the service"}
              >
                {id}
              </a>
            ) : (
              <span title={svc.description}>{id}</span>
            )}
            {svc.syncStatus && (
              <span className="text-muted ms-2" title="Synchronization of its files">
                {svc.syncStatus}
              </span>
            )}
            <Button
              size="sm"
              variant="link"
              className="p-0 ms-2"
              title="Logs"
              onClick={() =>
                setLogTarget({ kind: "service", experimentId, serviceId: id, title: id })
              }
            >
              <i className="fas fa-file-lines" />
            </Button>
            <Button
              size="sm"
              variant="link"
              className="p-0 ms-2 text-danger"
              title="Stop"
              onClick={() => client.service_stop(experimentId, id)}
            >
              <i className="fas fa-stop" />
            </Button>
          </span>
        );
      })}
      {logTarget && (
        <LogViewer target={logTarget} onHide={() => setLogTarget(undefined)} />
      )}
    </div>
  );
};

export default ActiveServices;
