import React, { useState } from "react";
import { Table, Badge, Button } from "react-bootstrap";

import { useAppSelector } from "./store";
import client from "./client";
import LogViewer, { LogTarget } from "./LogViewer";

const STATE_COLOR: { [key: string]: string } = {
  STOPPED: "secondary",
  STARTING: "info",
  RUNNING: "success",
  STOPPING: "warning",
  ERROR: "danger",
};

const Services = () => {
  const services = useAppSelector((state) => state.db.services);
  const [logTarget, setLogTarget] = useState<LogTarget>();

  return (
    <div className="mt-3">
      <h5 className="mb-3">Services</h5>
      {services.ids.length === 0 ? (
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
            {services.ids.map((id) => {
              const svc = services.byId[id];
              const running = svc.state === "RUNNING";
              return (
                <tr key={id}>
                  <td>
                    <Badge bg={STATE_COLOR[svc.state] || "secondary"}>
                      {svc.state}
                    </Badge>
                  </td>
                  <td>
                    <code>{id}</code>
                  </td>
                  <td className="small">
                    {svc.description}
                    {svc.error && (
                      <div className="text-danger small">{svc.error}</div>
                    )}
                  </td>
                  <td className="text-muted small">{svc.syncStatus || "—"}</td>
                  <td className="text-nowrap">
                    {running ? (
                      <>
                        <Button
                          size="sm"
                          variant="outline-primary"
                          className="me-1"
                          href={`/services/${id}`}
                          target="_blank"
                        >
                          Open
                        </Button>
                        <Button
                          size="sm"
                          variant="outline-danger"
                          className="me-1"
                          onClick={() => client.service_stop(id)}
                        >
                          Stop
                        </Button>
                      </>
                    ) : (
                      <Button
                        size="sm"
                        variant="outline-success"
                        className="me-1"
                        onClick={() => client.service_start(id)}
                      >
                        Start
                      </Button>
                    )}
                    <Button
                      size="sm"
                      variant="outline-secondary"
                      onClick={() =>
                        setLogTarget({ kind: "service", serviceId: id, title: id })
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
