import store from "./store";
import { actions } from "./reducers";

function error(message: string) {
  console.error("Not implemented");
}
function info(message: string) {
  console.error("Not implemented");
}

/// Connects to the Websocket server
class Client {
  ws: WebSocket;
  waiting: { [key: number]: object };
  queued: Array<any>;

  constructor() {
    console.log("Connecting to websocket");

    this.queued = [];
    this.waiting = {};

    let location = window.location;
    var url =
      "ws://" +
      location.hostname +
      (location.port ? ":" + location.port : "") +
      "/api";

    this.ws = new WebSocket(url);
    this.ws.addEventListener("open", this.open);
    this.ws.addEventListener("close", this.close);
    this.ws.addEventListener("message", this.message);
  }

  open = () => {
    console.log("Connection opened");
    store.dispatch(actions.setConnected(true));
    this.send({ type: "refresh" });
    this.send({ type: "services" });
  };

  close = (event) => {
    console.log("Closing in WS", this, event);
    console.log("Connection closed");
    store.dispatch(actions.setConnected(true));
    // info("Websocket connexion closed");
  };

  message = (event: any) => {
    console.log("[WS:in]", event.data);
    if (event.data == "unauthorized") {
      window.location.href = "/login.html";
      return;
    }

    let action = JSON.parse(event.data);
    if (action.error) {
      error(action.message);
    } else {
      switch (action.type) {
        case "JOB_ADD":
          store.dispatch(actions.addJob(action.payload));
          break;
        case "JOB_UPDATE":
          store.dispatch(actions.updateJob(action.payload));
          break;
        case "SERVICES_LIST":
          store.dispatch(actions.updateServices(action.payload));
          break;
        default:
          console.error("Unhandled action type", action.type)
      }
    }
  };

  /** Send without waiting for an answer */
  send = (data: any, message?: string) => {
    if (this.ws.readyState === WebSocket.OPEN) {
      console.log("[WS:out]", data);
      return this.ws.send(JSON.stringify(data));
    } else {
      console.log("Connection not ready");
      if (message) {
        error(`No websocket connection: could not ${message}`);
      }
      return false;
    }
  };

  /** Wait for an answer */
  query = (data: any, timeout: number = 60) => {
    return this.ws.send(JSON.stringify(data));
  };
}

export default new Client();
