import React, { useEffect } from "react";
import NavDropdown from "react-bootstrap/NavDropdown";
import Button from "react-bootstrap/Button";

import { useAppSelector } from "./store";
import client from "./client";

const SEVERITY_ICON: { [key: string]: string } = {
  info: "ℹ️",
  warning: "⚠️",
  error: "⛔",
};

const Warnings: React.FC = () => {
  const warnings = useAppSelector((s) => s.db.warnings);

  // Request the current unresolved warnings on mount
  useEffect(() => {
    client.request_warnings();
  }, []);

  if (warnings.keys.length === 0) return null;

  const title = (
    <span>
      <i className="fas fa-triangle-exclamation" style={{ color: "#d9822b" }} />{" "}
      Warnings{" "}
      <span className="badge bg-warning text-dark">{warnings.keys.length}</span>
    </span>
  );

  return (
    <NavDropdown title={title} id="warnings-nav-dropdown">
      {warnings.keys.map((key) => {
        const w = warnings.byKey[key];
        return (
          <div key={key} className="px-3 py-2" style={{ maxWidth: 420 }}>
            <div>
              {SEVERITY_ICON[w.severity] || "⚠️"}{" "}
              <span style={{ whiteSpace: "normal" }}>{w.description}</span>
            </div>
            <div className="mt-1">
              {Object.entries(w.actions).map(([actionKey, label]) => (
                <Button
                  key={actionKey}
                  size="sm"
                  variant="outline-secondary"
                  className="me-1"
                  onClick={() =>
                    client.warning_action(
                      w.warningKey,
                      actionKey,
                      w.experimentId,
                      w.runId,
                    )
                  }
                >
                  {label}
                </Button>
              ))}
            </div>
          </div>
        );
      })}
    </NavDropdown>
  );
};

export default Warnings;
