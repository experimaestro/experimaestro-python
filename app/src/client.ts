import { info, error } from "xpm/ui/notifications";
import { connected, process } from "./store";

/// Connects to the Websocket server
class Client {
  ws: WebSocket;
  waiting: Map<number, any>;
  queued: Array<any>;

  constructor() {
    console.log("Connecting to websocket");

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
    connected.update((_) => true);
    this.send({ type: "refresh" });
  };

  close = () => {
    console.log("Connection closed");
    connected.update((_) => false);
    info("Websocket connexion closed");
  };

  message = (event: any) => {
    console.log("[WS:in]", event.data);
    let action = JSON.parse(event.data);
    if (action.error) {
      error(action.message);
    } else {
      process(action);
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
        error("No websocket connection: could not " + message);
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
