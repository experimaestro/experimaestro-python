import React, { useEffect, useRef, useState } from "react";
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

import { useAppSelector, useNavigate } from "./store";
import { actions, jobsKey } from "./reducers";
import client from "./client";
import {
  Route,
  View,
  currentLocationRoute,
  formatRoute,
  isScoped,
  linkProps,
  route as viewRoute,
} from "./routes";

// Keeps the browser URL in sync with the route of the store: a navigation
// pushes a history entry, and back/forward shows the entry's route
function useHistory() {
  const dispatch = useDispatch();
  const route = useAppSelector((state) => state.db.route);
  const first = useRef(true);

  useEffect(() => {
    const url = formatRoute(route);
    if (url !== window.location.pathname + window.location.search) {
      // The URL the page was opened with is normalised in place
      if (first.current) window.history.replaceState(null, "", url);
      else window.history.pushState(null, "", url);
    }
    first.current = false;
  }, [route]);

  useEffect(() => {
    const onPop = () => dispatch(actions.navigate(currentLocationRoute()));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
}

export default () => {
  const dispatch = useDispatch();
  const connected = useAppSelector((state) => state.db.connected);
  const connectionStatus = useAppSelector((state) => state.db.connectionStatus);
  const currentExperiment = useAppSelector((state) => state.db.currentExperiment);
  const [showQuitModal, setShowQuitModal] = useState(false);
  const [showLog, setShowLog] = useState(false);
  const route = useAppSelector((state) => state.db.route);
  const view = route.view;
  const navigate = useNavigate();
  useHistory();
  const currentRun = useAppSelector((state) => state.db.currentRun);
  const jobsRequested = useAppSelector((state) => state.db.jobsRequested);
  const experimentsLoaded = useAppSelector((state) => state.db.experimentsLoaded);

  // Jobs are requested for the experiment run shown, when it is first shown
  // (all experiments when none is selected). Waits for the experiments, as
  // the first one is then selected.
  useEffect(() => {
    if (connectionStatus !== "connected" || !experimentsLoaded) return;
    const key = jobsKey(currentExperiment, currentRun);
    const all = currentRun === null && jobsRequested.includes(jobsKey(null, null));
    if (all || jobsRequested.includes(key)) return;
    dispatch(actions.markJobsRequested(key));
    client.request_jobs(currentExperiment, currentRun);
  }, [connectionStatus, experimentsLoaded, currentExperiment, currentRun, jobsRequested]);

  const handleQuit = () => {
    client.quit();
    setShowQuitModal(false);
  };

  // The scoped views keep the shown experiment and run
  const routeOf = (view: View): Route =>
    isScoped(view)
      ? isScoped(route.view)
        ? { ...route, view, job: null }
        : viewRoute(view, currentExperiment, currentRun)
      : viewRoute(view);
  const link = (view: View) => linkProps(routeOf(view), navigate);

  const openExperiment = (id: string) => navigate(viewRoute("tasks", id));

  const breadcrumb = (
    <div className="text-muted small mb-1">
      <a className="text-reset text-decoration-none" {...link("experiments")}>
        Experiments
      </a>
      {currentExperiment ? ` / ${currentExperiment}` : " / all"}
    </div>
  );

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
            {...link("experiments")}
            style={{ cursor: "pointer", display: "inline-flex", alignItems: "center" }}
          >
            {/* Logo sits in a white, dark-bordered disc that straddles the
                navbar's bottom edge: top half over the dark navbar, bottom half
                over the white content below. */}
            <span
              style={{
                position: "relative",
                zIndex: 1031,
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                width: 60,
                height: 60,
                marginTop: -6,
                marginBottom: -26,
                marginRight: 12,
                borderRadius: "50%",
                background: "#fff",
                border: "2px solid #212529",
                boxShadow: "0 2px 6px rgba(0, 0, 0, 0.35)",
              }}
            >
              <img
                src="/favicon.svg"
                alt=""
                style={{ width: 42, height: "auto" }}
              />
            </span>
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
              <Nav.Link eventKey="experiments" {...link("experiments")}>
                Experiments
              </Nav.Link>
              <Nav.Link eventKey="tasks" {...link("tasks")}>
                Tasks
              </Nav.Link>
              <Nav.Link eventKey="graph" {...link("graph")}>
                Graph
              </Nav.Link>
              <Nav.Link eventKey="services" {...link("services")}>
                Services
              </Nav.Link>
              <Nav.Link eventKey="actions" {...link("actions")}>
                Actions
              </Nav.Link>
              <Nav.Link eventKey="orphans" {...link("orphans")}>
                Orphans
              </Nav.Link>
              <Warnings />
            </Nav>
            {isScoped(view) && <ExperimentSelector />}
            {(view === "tasks" || view === "graph") && <RunSelector />}
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

      <Container
        className="mt-2"
        // The graph uses the whole width of the page
        fluid={view === "graph"}
        style={{ paddingLeft: view === "graph" ? undefined : 80 }}
      >
        {view === "experiments" && <Experiments onOpen={openExperiment} />}
        {isScoped(view) && breadcrumb}
        {view === "tasks" && <Tasks />}
        {view === "graph" && <TaskGraph />}
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
