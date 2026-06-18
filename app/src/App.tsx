import React, { useState } from "react";
import { useDispatch } from "react-redux";

import Container from "react-bootstrap/Container";
import Nav from "react-bootstrap/Nav";
import Navbar from "react-bootstrap/Navbar";
import Button from "react-bootstrap/Button";
import Modal from "react-bootstrap/Modal";

import Tasks from "./Tasks";
import TaskGraph from "./TaskGraph";
import Experiments from "./Experiments";
import Services from "./Services";
import Warnings from "./Warnings";
import Actions from "./Actions";
import Orphans from "./Orphans";
import LogViewer from "./LogViewer";
import { ExperimentSelector } from "./ExperimentSelector";
import { RunSelector } from "./RunSelector";

import { useAppSelector } from "./store";
import { actions } from "./reducers";
import client from "./client";

type View = "experiments" | "tasks" | "graph" | "orphans" | "services" | "actions";

export default () => {
  const dispatch = useDispatch();
  const connected = useAppSelector((state) => state.db.connected);
  const connectionStatus = useAppSelector((state) => state.db.connectionStatus);
  const currentExperiment = useAppSelector((state) => state.db.currentExperiment);
  const [showQuitModal, setShowQuitModal] = useState(false);
  const [showLog, setShowLog] = useState(false);
  const [view, setView] = useState<View>("experiments");

  const handleQuit = () => {
    client.quit();
    setShowQuitModal(false);
  };

  const openExperiment = (id: string) => {
    dispatch(actions.selectExperiment(id));
    setView("tasks");
  };

  return (
    <>
      <div
        id="clipboard-holder"
        style={{ overflow: "hidden", width: 0, height: 0 }}
      ></div>

      {/* Quit Confirmation Modal */}
      <Modal show={showQuitModal} onHide={() => setShowQuitModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Stop Server</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          Are you sure you want to stop the experimaestro server? This will
          terminate the monitoring session.
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowQuitModal(false)}>
            Cancel
          </Button>
          <Button variant="danger" onClick={handleQuit}>
            Stop Server
          </Button>
        </Modal.Footer>
      </Modal>

      <Navbar bg="dark" variant="dark" expand="lg">
        <Container>
          <Navbar.Brand
            onClick={() => setView("experiments")}
            style={{ cursor: "pointer" }}
          >
            <i className="fas fa-diagram-project me-2" />
            Experimaestro
            <span
              title={
                connectionStatus === "connected"
                  ? "Connected"
                  : connectionStatus === "connecting"
                    ? "Connecting…"
                    : "Disconnected"
              }
              style={{
                display: "inline-block",
                width: 10,
                height: 10,
                borderRadius: "50%",
                marginLeft: 8,
                verticalAlign: "middle",
                background:
                  connectionStatus === "connected"
                    ? "#2ecc40"
                    : connectionStatus === "connecting"
                      ? "#ffdc00"
                      : "#ff4136",
              }}
            />
          </Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="me-auto" activeKey={view}>
              <Nav.Link eventKey="experiments" onClick={() => setView("experiments")}>
                Experiments
              </Nav.Link>
              <Nav.Link eventKey="tasks" onClick={() => setView("tasks")}>
                Tasks
              </Nav.Link>
              <Nav.Link eventKey="graph" onClick={() => setView("graph")}>
                Graph
              </Nav.Link>
              <Nav.Link eventKey="services" onClick={() => setView("services")}>
                Services
              </Nav.Link>
              <Nav.Link eventKey="actions" onClick={() => setView("actions")}>
                Actions
              </Nav.Link>
              <Nav.Link eventKey="orphans" onClick={() => setView("orphans")}>
                Orphans
              </Nav.Link>
              <Warnings />
            </Nav>
            {(view === "tasks" || view === "graph") && (
              <>
                <ExperimentSelector />
                <RunSelector />
              </>
            )}
            <Button
              variant="outline-light"
              size="sm"
              className="ms-2"
              title="Experimaestro log"
              onClick={() => setShowLog(true)}
            >
              <i className="fas fa-file-lines me-1" />
              Log
            </Button>
            {connected && (
              <Button
                variant="outline-light"
                size="sm"
                className="ms-2"
                onClick={() => setShowQuitModal(true)}
              >
                <i className="fas fa-power-off me-1" />
                Stop
              </Button>
            )}
          </Navbar.Collapse>
        </Container>
      </Navbar>

      <Container className="mt-2">
        {view === "experiments" && <Experiments onOpen={openExperiment} />}
        {view === "tasks" && (
          <>
            <div className="text-muted small mb-1">
              <span
                style={{ cursor: "pointer" }}
                onClick={() => setView("experiments")}
              >
                Experiments
              </span>
              {currentExperiment ? ` / ${currentExperiment}` : " / all"}
            </div>
            <Tasks />
          </>
        )}
        {view === "graph" && (
          <>
            <div className="text-muted small mb-1">
              <span
                style={{ cursor: "pointer" }}
                onClick={() => setView("experiments")}
              >
                Experiments
              </span>
              {currentExperiment ? ` / ${currentExperiment}` : " / all"}
            </div>
            <TaskGraph />
          </>
        )}
        {view === "services" && <Services />}
        {view === "actions" && <Actions />}
        {view === "orphans" && <Orphans />}
      </Container>

      {showLog && (
        <LogViewer
          target={{ kind: "experimaestro", title: "experimaestro" }}
          onHide={() => setShowLog(false)}
        />
      )}
    </>
  );
};
