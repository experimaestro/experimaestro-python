import React from "react";

import { createRoot } from "react-dom/client";
import { JssProvider } from "react-jss";

import store from "./store";
import App from "./App";
import { Provider } from "react-redux";
import { Messages } from "./ui/messages";
import { create, Rule } from "jss";
import preset from "jss-preset-default";

const container = document.getElementById("root");
const jss = create(preset());

if (container) {
  const root = createRoot(container);
  root.render(
    <React.StrictMode>
      <JssProvider jss={jss}>
        <Provider store={store}>
          <Messages />
          <App />
        </Provider>
      </JssProvider>
    </React.StrictMode>
  );
}
