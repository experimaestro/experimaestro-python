// Native WebSocket client for experimaestro WebUI
import store from "./store"
import { actions } from "./reducers"
import { showMessage } from "./ui/messages"
import { ExperimentGraph, GraphView, ViewParams } from "./dag/types"

// Message types for JSON protocol
interface Message {
  type: string;
  payload?: any;
}

// Answer to a "jobs.filter" request (filter DSL evaluated server-side)
export type JobsFilterResult = {
  requestId: number;
  query: string;
  jobIds: string[] | null;
  error: string | null;
};

// A completion of the filter expression (see complete_filter in cli/filter.py)
export type FilterSuggestion = {
  label: string;
  // Replaces the text between start and end of the query
  text: string;
  kind: "field" | "operator" | "value" | "keyword";
  detail: string;
};

// Answer to a "jobs.complete" request
export type JobsCompleteResult = {
  requestId: number;
  start: number;
  end: number;
  items: FilterSuggestion[];
};

// Answer to an "experiment.graph" request (see experimaestro.ui.graph)
export type ExperimentGraphResult = {
  requestId: number;
  graph?: ExperimentGraph;
  view?: GraphView;
  error?: string;
};

class Client {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private reconnectTimer: number | null = null;

  // Callbacks for the action execution round-trip (set by the Actions UI)
  onActionPrompt: ((payload: any) => void) | null = null;
  onActionResult: ((payload: any) => void) | null = null;
  // Callback for server-side job filter results (set by the Tasks view)
  onJobsFilter: ((payload: JobsFilterResult) => void) | null = null;
  // Callback for the completions of the filter (set by the filter input)
  onJobsComplete: ((payload: JobsCompleteResult) => void) | null = null;
  // Callback for experiment graph results (set by the graph view)
  onExperimentGraph: ((payload: ExperimentGraphResult) => void) | null = null;

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

      // Request initial data (jobs are requested for the experiment shown, see
      // App.tsx)
      store.dispatch(actions.resetJobsRequested());
      this.send("experiments");
      // The started services (shown on every page)
      this.send("services");
    };

    this.ws.onclose = (event) => {
      console.log("WebSocket closed:", event.code, event.reason);
      store.dispatch(actions.setConnectionStatus("connecting"));
      this.ws = null;

      // Handle authentication error
      if (event.code === 1008 && event.reason === "Invalid token") {
        console.log("Invalid token, redirecting to login");
        window.location.href = `${window.location.protocol}//${window.location.host}/login.html?next=${encodeURIComponent(window.location.pathname + window.location.search)}`;
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
        case "experiments.loaded":
          store.dispatch(actions.setExperimentsLoaded());
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
        case "jobs.filter":
          if (this.onJobsFilter) this.onJobsFilter(payload);
          break;
        case "jobs.complete":
          if (this.onJobsComplete) this.onJobsComplete(payload);
          break;
        case "experiment.graph":
          if (this.onExperimentGraph) this.onExperimentGraph(payload);
          break;
        case "experiment.removed":
          store.dispatch(actions.removeExperiment(payload));
          break;
        case "notify": {
          const severity: string = payload?.severity || "info";
          store.dispatch(
            showMessage({
              severity,
              message: payload?.message || "",
              title: severity === "danger" ? "Error" : "Information",
              delay: severity === "danger" ? undefined : 5,
            }) as any,
          );
          break;
        }
        case "orphans":
          store.dispatch(actions.setOrphans(payload?.jobs || []));
          break;
        case "orphan.removed":
          store.dispatch(actions.removeOrphan(payload));
          break;
        case "error":
          console.error("Server error:", payload?.message);
          if (payload?.message === "Invalid token") {
            window.location.href = `${window.location.protocol}//${window.location.host}/login.html?next=${encodeURIComponent(window.location.pathname + window.location.search)}`;
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

  jobs_filter(
    query: string,
    requestId: number,
    experimentId?: string | null,
    runId?: string | null,
  ): void {
    this.send("jobs.filter", {
      query,
      requestId,
      experimentId: experimentId || undefined,
      runId: runId || undefined,
    });
  }

  jobs_complete(
    query: string,
    cursor: number,
    requestId: number,
    experimentId?: string | null,
    runId?: string | null,
  ): void {
    this.send("jobs.complete", {
      query,
      cursor,
      requestId,
      experimentId: experimentId || undefined,
      runId: runId || undefined,
    });
  }

  experiment_graph(
    requestId: number,
    experimentId: string,
    runId: string | null,
    params: ViewParams,
  ): void {
    this.send("experiment.graph", {
      requestId,
      experimentId,
      runId: runId || undefined,
      ...params,
    });
  }

  experiment_kill(experimentId: string): void {
    this.send("experiment.kill", { experimentId });
  }

  experiment_delete(experimentId: string): void {
    this.send("experiment.delete", { experimentId });
  }

  refresh(experimentId?: string, runId?: string): void {
    const payload: any = {};
    if (experimentId) payload.experimentId = experimentId;
    if (runId) payload.runId = runId;
    this.send("refresh", payload);
  }

  // Jobs of an experiment run (of all the experiments by default), from the
  // server cache. The connection then watches that experiment: the server
  // only sends the job, action and service events of the watched experiment
  // (and of the started services)
  request_jobs(experimentId: string | null, runId: string | null): void {
    const payload: any = {};
    if (experimentId) payload.experimentId = experimentId;
    if (runId) payload.runId = runId;
    this.send("jobs", payload);
  }

  // Services of an experiment (shown by the services page)
  request_services(experimentId: string): void {
    this.send("services", { experimentId });
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

  /** Kill every stray (running orphan) job */
  orphans_kill(): void {
    this.send("orphans.kill");
  }

  /** Delete every finished orphan job */
  orphans_delete(): void {
    this.send("orphans.delete");
  }

  service_start(experimentId: string, id: string): void {
    this.send("service.start", { experimentId, id });
  }

  service_stop(experimentId: string, id: string): void {
    this.send("service.stop", { experimentId, id });
  }

  quit(): void {
    this.send("quit");
  }
}

export default new Client();
