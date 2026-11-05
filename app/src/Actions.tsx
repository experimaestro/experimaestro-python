import React, { useEffect, useState } from "react";
import { Table, Modal, Button, Form } from "react-bootstrap";

import { useAppSelector } from "./store";
import { actions as storeActions, ActionInfo } from "./reducers";
import { useDispatch } from "react-redux";
import { useMessages } from "./ui/messages";
import client from "./client";
import { DocLink } from "./ui/DocLink";
import { copyToClibpoard } from "./clipboard";

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
  runId: string;
  field: PromptField;
  inputs: { [key: string]: string };
};

const Actions: React.FC = () => {
  const actionsState = useAppSelector((s) => s.db.actions);
  const currentExperiment = useAppSelector((s) => s.db.currentExperiment);
  const currentRun = useAppSelector((s) => s.db.currentRun);
  const connectionStatus = useAppSelector((s) => s.db.connectionStatus);
  const dispatch = useDispatch();
  const { success, error, info } = useMessages();

  const [prompt, setPrompt] = useState<PromptState | null>(null);
  const [value, setValue] = useState<string>("");
  // Action of a former run waiting for the confirmation of its execution
  const [confirm, setConfirm] = useState<ActionInfo | null>(null);

  // The actions of the run shown
  useEffect(() => {
    if (connectionStatus !== "connected") return;
    dispatch(storeActions.clearActions());
    client.request_actions(currentExperiment, currentRun);
  }, [connectionStatus, currentExperiment, currentRun]);

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

  const execute = (a: ActionInfo) => {
    info(`Running action ${a.actionId}…`);
    client.action_execute(a.actionId, a.experimentId, a.runId, {});
  };

  // The action of a former run is executed once confirmed
  const run = (a: ActionInfo) => (a.pastRun ? setConfirm(a) : execute(a));

  const submitPrompt = () => {
    if (!prompt) return;
    const inputs = { ...prompt.inputs, [prompt.field.key]: value };
    client.action_execute(prompt.actionId, prompt.experimentId, prompt.runId, inputs);
  };

  return (
    <div className="mt-3">
      <h5 className="mb-3">
        Actions
        <DocLink page="experiments/actions.html" />
      </h5>
      {actionsState.ids.length === 0 ? (
        <p className="text-muted">No actions declared by this experiment run.</p>
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
                <tr
                  key={id}
                  data-experiment-id={a.experimentId}
                  data-action-id={a.actionId}
                >
                  <td>
                    {/* Identifiers are hashes: the start is enough, as for jobs */}
                    <code
                      className="clipboard"
                      title={a.actionId}
                      onClick={() =>
                        copyToClibpoard(a.actionId)
                          .then(() => success("Action id copied"))
                          .catch((e) => error("Error when copying: " + e))
                      }
                    >
                      {a.actionId.substring(0, 12)}
                    </code>
                  </td>
                  <td className="small">{a.description}</td>
                  <td className="text-muted small">{a.actionClass}</td>
                  <td className="text-nowrap">
                    <Button
                      size="sm"
                      variant="outline-primary"
                      onClick={() => run(a)}
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

      {confirm && (
        <Modal show={true} onHide={() => setConfirm(null)}>
          <Modal.Header closeButton>
            <Modal.Title>Run an action of a former run?</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            Run action <code>{confirm.actionId.substring(0, 12)}</code> (
            {confirm.description}) of run <b>{confirm.runId}</b>?
            <div className="text-danger small mt-2 action-past-run-warning">
              <i className="fas fa-triangle-exclamation me-1" />
              {confirm.pastRunWarning}
            </div>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setConfirm(null)}>
              Cancel
            </Button>
            <Button
              variant="warning"
              onClick={() => {
                const a = confirm;
                setConfirm(null);
                execute(a);
              }}
            >
              Run
            </Button>
          </Modal.Footer>
        </Modal>
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
