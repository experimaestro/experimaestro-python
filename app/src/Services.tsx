import React, { useState } from "react";
import { Table, Badge, Button } from "react-bootstrap";

import { useAppSelector } from "./store";
import client from "./client";
import LogViewer, { LogTarget } from "./LogViewer";
import { DocLink } from "./ui/DocLink";

const STATE_COLOR: { [key: string]: string } = {
  STOPPED: "secondary",
  STARTING: "info",
  RUNNING: "success",
  STOPPING: "warning",
  ERROR: "danger",
};

const Services = () => {
  const services = useAppSelector((state) => state.db.services);
  const currentExperiment = useAppSelector((state) => state.db.currentExperiment);
  const [logTarget, setLogTarget] = useState<LogTarget>();
  // Services of the selected experiment (all of them by default)
  const keys = services.ids.filter(
    (key) =>
      currentExperiment === null || services.byId[key].experimentId === currentExperiment,
  );

  return (
    <div className="mt-3">
      <h5 className="mb-3">
        Services
        <DocLink page="services.html" />
      </h5>
      {keys.length === 0 ? (
        <p className="text-muted">No services.</p>
      ) : (
        <Table hover responsive className="align-middle">
          <thead>
            <tr>
              <th>State</th>
              <th>Service</th>
              <th>Description</th>
              <th>Sync</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {keys.map((key) => {
              const svc = services.byId[key];
              const { id, experimentId } = svc;
              const running = svc.state === "RUNNING";
              // A starting service can be stopped (e.g. a long synchronization)
              const active = running || svc.state === "STARTING";
              return (
                <tr key={key} data-service-id={id} data-experiment-id={experimentId}>
                  <td>
                    <Badge bg={STATE_COLOR[svc.state] || "secondary"}>{svc.state}</Badge>
                  </td>
                  <td>
                    <code>{id}</code>
                  </td>
                  <td className="small">
                    {svc.description}
                    {svc.error && <div className="text-danger small">{svc.error}</div>}
                  </td>
                  <td className="text-muted small">{svc.syncStatus || "—"}</td>
                  <td className="text-nowrap">
                    {active ? (
                      <>
                        {running && (
                          <Button
                            size="sm"
                            variant="outline-primary"
                            className="me-1"
                            href={`/services/${encodeURIComponent(
                              experimentId,
                            )}/${encodeURIComponent(id)}/`}
                            target="_blank"
                          >
                            Open
                          </Button>
                        )}
                        <Button
                          size="sm"
                          variant="outline-danger"
                          className="me-1"
                          onClick={() => client.service_stop(experimentId, id)}
                        >
                          Stop
                        </Button>
                      </>
                    ) : (
                      <Button
                        size="sm"
                        variant="outline-success"
                        className="me-1"
                        onClick={() => client.service_start(experimentId, id)}
                      >
                        Start
                      </Button>
                    )}
                    <Button
                      size="sm"
                      variant="outline-secondary"
                      onClick={() =>
                        setLogTarget({
                          kind: "service",
                          experimentId,
                          serviceId: id,
                          title: id,
                        })
                      }
                    >
                      Logs
                    </Button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </Table>
      )}
      {logTarget && (
        <LogViewer target={logTarget} onHide={() => setLogTarget(undefined)} />
      )}
    </div>
  );
};

export default Services;
