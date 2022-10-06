import React from "react";
import Tasks from "./Tasks";
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
      <header className="App-header">
        <h1 className="App-title">
          Experimaestro {experiment ? " – " + experiment : ""}{" "}
          <i
            className={`fab fa-staylinked ws-status ${
              connected ? "ws-link" : "ws-no-link"
            }`}
          />{" "}
        </h1>
      </header>
      <Tasks />
    </>
  );
};
