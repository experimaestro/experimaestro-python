import React, { useEffect, useState } from "react";
import { Table, Modal, Button, Form } from "react-bootstrap";

import { useAppSelector } from "./store";
import { useMessages } from "./ui/messages";
import client from "./client";

type PromptField = {
  key: string;
  label: string;
  kind: "choice" | "checkbox" | "text";
  choices?: string[];
  default?: string | boolean;
};

type PromptState = {
  actionId: string;
  experimentId: string;
  field: PromptField;
  inputs: { [key: string]: string };
};

const Actions: React.FC = () => {
  const actionsState = useAppSelector((s) => s.db.actions);
  const currentExperiment = useAppSelector((s) => s.db.currentExperiment);
  const { success, error, info } = useMessages();

  const [prompt, setPrompt] = useState<PromptState | null>(null);
  const [value, setValue] = useState<string>("");

  useEffect(() => {
    client.request_actions(currentExperiment || undefined);
  }, [currentExperiment]);

  // Register the action round-trip callbacks
  useEffect(() => {
    client.onActionPrompt = (payload: PromptState) => {
      setPrompt(payload);
      const d = payload.field.default;
      setValue(d === undefined || d === null ? "" : String(d));
    };
    client.onActionResult = (payload: any) => {
      setPrompt(null);
      if (payload.ok) success(`Action ${payload.actionId} completed`);
      else error(`Action ${payload.actionId} failed: ${payload.error}`);
    };
    return () => {
      client.onActionPrompt = null;
      client.onActionResult = null;
    };
  }, [success, error]);

  const execute = (actionId: string, experimentId: string) => {
    info(`Running action ${actionId}…`);
    client.action_execute(actionId, experimentId, {});
  };

  const submitPrompt = () => {
    if (!prompt) return;
    const inputs = { ...prompt.inputs, [prompt.field.key]: value };
    client.action_execute(prompt.actionId, prompt.experimentId, inputs);
  };

  return (
    <div className="mt-3">
      <h5 className="mb-3">Actions</h5>
      {actionsState.ids.length === 0 ? (
        <p className="text-muted">No actions declared by this experiment.</p>
      ) : (
        <Table hover responsive className="align-middle">
          <thead>
            <tr>
              <th>Action</th>
              <th>Description</th>
              <th>Class</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {actionsState.ids.map((id) => {
              const a = actionsState.byId[id];
              return (
                <tr key={id}>
                  <td>
                    <code>{a.actionId}</code>
                  </td>
                  <td className="small">{a.description}</td>
                  <td className="text-muted small">{a.actionClass}</td>
                  <td className="text-nowrap">
                    <Button
                      size="sm"
                      variant="outline-primary"
                      onClick={() => execute(a.actionId, a.experimentId)}
                    >
                      Run
                    </Button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </Table>
      )}

      {prompt && (
        <Modal show={true} onHide={() => setPrompt(null)}>
          <Modal.Header closeButton>
            <Modal.Title>Action input — {prompt.actionId}</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <Form.Label>{prompt.field.label}</Form.Label>
            {prompt.field.kind === "choice" && (
              <Form.Select value={value} onChange={(e) => setValue(e.target.value)}>
                {(prompt.field.choices || []).map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </Form.Select>
            )}
            {prompt.field.kind === "checkbox" && (
              <Form.Check
                type="checkbox"
                label={prompt.field.label}
                checked={value === "true"}
                onChange={(e) => setValue(e.target.checked ? "true" : "false")}
              />
            )}
            {prompt.field.kind === "text" && (
              <Form.Control value={value} onChange={(e) => setValue(e.target.value)} />
            )}
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setPrompt(null)}>
              Cancel
            </Button>
            <Button variant="primary" onClick={submitPrompt}>
              Continue
            </Button>
          </Modal.Footer>
        </Modal>
      )}
    </div>
  );
};

export default Actions;
