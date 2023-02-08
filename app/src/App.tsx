import React from "react";

import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import NavDropdown from 'react-bootstrap/NavDropdown';


import Tasks from "./Tasks";
import Services from "./Services";

import { useAppSelector } from "./store";

export default () => {
  const connected = useAppSelector((state) => state.db.connected);
  const jobs = useAppSelector((state) => state.db.jobs);
  const experiment = useAppSelector((state) => state.db.experiment);

  return (
    <>
      <div
        id="clipboard-holder"
        style={{ overflow: "hidden", width: 0, height: 0 }}
      ></div>

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
