import React from "react";
import { Alert } from "react-bootstrap";

import { useAppSelector } from "./store";

/** The connection of the server to a remote (SSH) workspace, when it is not
 * up (on every page) */
export const ProviderConnection = () => {
  const connection = useAppSelector((state) => state.db.providerConnection);
  if (connection === null || connection.state === "connected") return null;

  const host = connection.host ? ` to ${connection.host}` : "";
  const reconnecting = connection.state === "reconnecting";
  return (
    <Alert
      variant={reconnecting ? "warning" : "danger"}
      className="provider-connection mb-0 py-1 rounded-0 small"
      data-state={connection.state}
    >
      {reconnecting ? (
        <>
          <i className="fas fa-rotate fa-spin me-2" />
          Connection{host} lost, reconnecting
          {connection.attempt > 0 &&
            ` (attempt ${connection.attempt}/${connection.maxRetries})`}
        </>
      ) : (
        <>
          <i className="fas fa-plug-circle-xmark me-2" />
          Disconnected{connection.host ? ` from ${connection.host}` : ""}
          {connection.attempt > 0 && ": restart the monitor to reconnect"}
        </>
      )}
      {connection.error && <span className="text-muted ms-2">{connection.error}</span>}
    </Alert>
  );
};

export default ProviderConnection;
