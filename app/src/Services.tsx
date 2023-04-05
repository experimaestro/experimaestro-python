import React, { useState, useMemo } from "react";
import { Spinner } from "react-bootstrap";
import NavDropdown from 'react-bootstrap/NavDropdown';

import { useAppSelector } from "./store";

const Services = () => {
    const services = useAppSelector((state) => state.db.services);

    if (services.ids.length == 0) return <></>

    const STATE_COMPONENTS = {
        STOPPED: <i className="fas fa-circle" style={{color: "red"}}></i>,
        STARTING: <i className="fa-solid fa-circle-notch fa-spin" style={{color: "green"}}></i>,
        RUNNING: <i className="fa-solid fa-circle" style={{color: "green"}}></i>,
        STOPPING: <i className="fa-solid fa-circle-notch fa-spin" style={{color: "red"}}></i>
    }
    return <NavDropdown title="Services" id="basic-nav-dropdown">{
        services.ids.map(id =>
            <NavDropdown.Item key={id} href={`/services/${id}`} target='_blank'>
                <code>{id}</code> {services.byId[id].description}{" "}
                {STATE_COMPONENTS[services.byId[id].state] ?? services.byId[id].state}
            </NavDropdown.Item>
        )
        }
    </NavDropdown>

}


export default Services
