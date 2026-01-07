import React, { useState } from "react";

import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import NavDropdown from 'react-bootstrap/NavDropdown';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';


import Tasks from "./Tasks";
import Services from "./Services";
import { ExperimentSelector } from "./ExperimentSelector";

import { useAppSelector } from "./store";
import client from "./client";

export default () => {
  const connected = useAppSelector((state) => state.db.connected);
  const jobs = useAppSelector((state) => state.db.jobs);
  const experiment = useAppSelector((state) => state.db.experiment);
  const [showQuitModal, setShowQuitModal] = useState(false);

  const handleQuit = () => {
    client.quit();
    setShowQuitModal(false);
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
          Are you sure you want to stop the experimaestro server?
          This will terminate the monitoring session.
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

      <Navbar bg="light" expand="lg">
      <Container>
        <Navbar.Brand href="/">
          Experimaestro {experiment ? " – " + experiment : ""}{" "}
        </Navbar.Brand>
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="me-auto">
            <Nav.Link href="/">Tasks</Nav.Link>
            <Services/>
          </Nav>
          <ExperimentSelector />
          {connected && (
            <Button
              variant="outline-danger"
              size="sm"
              className="ms-2"
              onClick={() => setShowQuitModal(true)}
            >
              <i className="fas fa-power-off me-1" />
              Stop
            </Button>
          )}
          <i
            className={`fab fa-staylinked ws-status ${
              connected ? "ws-link" : "ws-no-link"
            }`}
          />
        </Navbar.Collapse>
      </Container>
    </Navbar>

    <Container>
      <Tasks />
    </Container>

    </>
  );
};
