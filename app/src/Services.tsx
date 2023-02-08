import React, { useState, useMemo } from "react";
import NavDropdown from 'react-bootstrap/NavDropdown';

import { useAppSelector } from "./store";

const Services = () => {
    const services = useAppSelector((state) => state.db.services);

    if (services.ids.length == 0) return <></>

    return <NavDropdown title="Services" id="basic-nav-dropdown">{
        services.ids.map(id =>
            <NavDropdown.Item key={id} href={`/services/${id}`}>
                <code>{id}</code> {services.byId[id]}
            </NavDropdown.Item>
        )
        }
    </NavDropdown>

}


export default Services
