// Native WebSocket client for experimaestro WebUI
import store from "./store"
import { actions } from "./reducers"

// Message types for JSON protocol
interface Message {
  type: string;
  payload?: any;
}

class Client {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private reconnectTimer: number | null = null;

  // Callbacks for the action execution round-trip (set by the Actions UI)
  onActionPrompt: ((payload: any) => void) | null = null;
  onActionResult: ((payload: any) => void) | null = null;

  constructor() {
    this.connect();
  }

  private getWebSocketUrl(): string {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    // Get token from cookie
    const token = this.getCookie("experimaestro_token") || "";
    return `${protocol}//${host}/ws?token=${encodeURIComponent(token)}`;
  }

  private getCookie(name: string): string | null {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) {
      return parts.pop()?.split(";").shift() || null;
    }
    return null;
  }

  private connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
      return;
    }

    const url = this.getWebSocketUrl();
    console.log("Connecting to WebSocket:", url.replace(/token=[^&]+/, "token=***"));

    store.dispatch(actions.setConnectionStatus("connecting"));
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      console.log("WebSocket connected");
      this.reconnectAttempts = 0;
      store.dispatch(actions.setConnectionStatus("connected"));

      // Request initial data
      this.send("experiments");
      this.send("refresh");
      this.send("services");
    };

    this.ws.onclose = (event) => {
      console.log("WebSocket closed:", event.code, event.reason);
      store.dispatch(actions.setConnectionStatus("connecting"));
      this.ws = null;

      // Handle authentication error
      if (event.code === 1008 && event.reason === "Invalid token") {
        console.log("Invalid token, redirecting to login");
        window.location.href = `${window.location.protocol}//${window.location.host}/login.html`;
        return;
      }

      // Attempt reconnection
      this.scheduleReconnect();
    };

    this.ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    this.ws.onmessage = (event) => {
      this.handleMessage(event.data);
    };
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log("Max reconnection attempts reached");
      store.dispatch(actions.setConnectionStatus("disconnected"));
      return;
    }

    if (this.reconnectTimer !== null) {
      return;
    }

    // Exponential backoff with jitter
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts) + Math.random() * 1000,
      30000
    );
    this.reconnectAttempts++;

    console.log(`Reconnecting in ${Math.round(delay)}ms (attempt ${this.reconnectAttempts})`);
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }

  private handleMessage(data: string): void {
    try {
      const message: Message = JSON.parse(data);
      const { type, payload } = message;

      switch (type) {
        case "experiment.add":
          store.dispatch(actions.addExperiment(payload));
          break;
        case "experiment.update":
          store.dispatch(actions.updateExperiment(payload));
          break;
        case "job.add":
          store.dispatch(actions.addJob(payload));
          break;
        case "job.update":
          store.dispatch(actions.updateJob(payload));
          break;
        case "job.removed":
          store.dispatch(actions.removeJob(payload));
          break;
        case "service.add":
          store.dispatch(actions.addService(payload));
          break;
        case "service.update":
          store.dispatch(actions.updateService(payload));
          break;
        case "experiment.runs":
          store.dispatch(actions.setRuns(payload?.runs || []));
          break;
        case "warning.add":
          store.dispatch(actions.addWarning(payload));
          break;
        case "warning.resolved":
          store.dispatch(actions.resolveWarning(payload));
          break;
        case "action.add":
          store.dispatch(actions.addAction(payload));
          break;
        case "action.prompt":
          if (this.onActionPrompt) this.onActionPrompt(payload);
          break;
        case "action.result":
          if (this.onActionResult) this.onActionResult(payload);
          break;
        case "orphans":
          store.dispatch(actions.setOrphans(payload?.jobs || []));
          break;
        case "orphan.removed":
          store.dispatch(actions.removeOrphan(payload));
          break;
        case "error":
          console.error("Server error:", payload?.message);
          if (payload?.message === "Invalid token") {
            window.location.href = `${window.location.protocol}//${window.location.host}/login.html`;
          }
          break;
        default:
          console.log("Unknown message type:", type);
      }
    } catch (e) {
      console.error("Failed to parse message:", e);
    }
  }

  private send(type: string, payload?: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const message: Message = { type };
      if (payload !== undefined) {
        message.payload = payload;
      }
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn("WebSocket not connected, cannot send:", type);
    }
  }

  job_details(jobId: string, taskId?: string, experimentId?: string): void {
    this.send("job.details", { jobId, taskId, experimentId });
  }

  job_kill(jobId: string, taskId?: string): void {
    this.send("job.kill", { jobId, taskId });
  }

  job_delete(jobId: string, taskId?: string): void {
    this.send("job.delete", { jobId, taskId });
  }

  refresh(experimentId?: string, runId?: string): void {
    const payload: any = {};
    if (experimentId) payload.experimentId = experimentId;
    if (runId) payload.runId = runId;
    this.send("refresh", payload);
  }

  request_runs(experimentId: string): void {
    this.send("experiment.runs", { experimentId });
  }

  request_warnings(): void {
    this.send("warnings");
  }

  warning_action(
    warningKey: string,
    actionKey: string,
    experimentId?: string,
    runId?: string,
  ): void {
    this.send("warning.action", { warningKey, actionKey, experimentId, runId });
  }

  request_actions(experimentId?: string): void {
    this.send("actions", experimentId ? { experimentId } : {});
  }

  action_execute(
    actionId: string,
    experimentId: string,
    inputs: { [key: string]: string },
  ): void {
    this.send("action.execute", { actionId, experimentId, inputs });
  }

  request_orphans(): void {
    this.send("orphans");
  }

  orphan_kill(jobId: string, taskId: string): void {
    this.send("orphan.kill", { jobId, taskId });
  }

  orphan_delete(jobId: string, taskId: string): void {
    this.send("orphan.delete", { jobId, taskId });
  }

  service_start(id: string): void {
    this.send("service.start", { id });
  }

  service_stop(id: string): void {
    this.send("service.stop", { id });
  }

  quit(): void {
    this.send("quit");
  }
}

export default new Client();
