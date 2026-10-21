import { combineReducers, createSlice, PayloadAction } from "@reduxjs/toolkit";
import { messageSlice } from "./ui/messages";
import _ from "lodash";
import { Route, currentLocationRoute, isScoped, route } from "./routes";

export type CarbonImpact = {
  co2kg?: number;
  energyKwh?: number;
};

export type Experiment = {
  experiment_id: string;
  workdir?: string;
  current_run_id?: string | null;
  status?: string | null;
  // Displayed status, shared with the TUI: failed, running, done, stopped or empty
  statusCategory: string;
  hostname?: string | null;
  // Tags of the experiment (experiment(..., tags=[...]))
  tags: string[];
  started_at?: string | null;
  ended_at?: string | null;
  total_jobs: number;
  finished_jobs: number;
  failed_jobs: number;
  carbon?: CarbonImpact | null;
  created_at?: string;
  updated_at?: string;
};

export type Experiments = {
  byId: { [key: string]: Experiment };
  ids: Array<string>;
};

// A run of an experiment (same shape as Experiment, plus run metadata)
export type Run = Experiment & {
  run_id?: string | null;
  isCurrent?: boolean;
};

export type CarbonMetrics = {
  co2kg?: number;
  energyKwh?: number;
  cpuPowerW?: number;
  gpuPowerW?: number;
  ramPowerW?: number;
  durationS?: number;
  region?: string | null;
  isFinal?: boolean;
};

export type ProcessInfo = {
  pid?: number | null;
  type?: string | null;
  running?: boolean;
  cpuPercent?: number | null;
  memoryMb?: number | null;
  numThreads?: number | null;
};

// Known job statuses; the backend may also send scheduler states (e.g.
// "scheduled"), so the field is typed as a string.
export type JobStatus =
  | "running"
  | "done"
  | "error"
  | "ready"
  | "waiting"
  | string;

export type Job = {
  jobId: string;
  taskId: string;

  locator: string;
  status: JobStatus;
  // Sort rank of the status, lower first (scheduler/job_status.py)
  statusRank: number;
  schedulerState?: string | null;

  start: string;
  end: string;
  submitted: string;

  failureReason?: string | null;
  exitCode?: number | null;
  retryCount?: number;

  tags: Array<[string, number | string | boolean]>;

  progress: Array<{
    level: number;
    desc: string | null;
    progress: number;
  }>;

  carbon?: CarbonMetrics | null;
  process?: ProcessInfo | null;

  experimentIds?: Array<string>; // Jobs can belong to multiple experiments
  dependsOn?: Array<string>; // Job IDs this job depends on
  runId?: string;
};

export type Jobs = {
  byId: { [key: string]: Job };
  ids: Array<string>;
};

export type OrphanJob = Job & {
  isStray: boolean;
  sizeBytes: number;
};

type ServiceStatus = "STOPPED" | "STARTING" | "RUNNING" | "STOPPING" | "ERROR";

export type ServiceInformation = {
  // Unique key ("<experiment>/<service>"): service ids are only unique within
  // an experiment
  key: string;
  experimentId: string;
  id: string;
  description: string;
  state: ServiceStatus;
  url?: string | null;
  syncStatus?: string | null;
  error?: string | null;
};

export type Services = {
  byId: { [key: string]: ServiceInformation };
  ids: string[];
};

export type Warning = {
  warningKey: string;
  experimentId: string;
  runId: string;
  description: string;
  severity: string;
  actions: { [key: string]: string };
  context: any;
};

export type ActionInfo = {
  actionId: string;
  experimentId: string;
  description: string;
  actionClass: string;
};

const status2int = (status: JobStatus): number => {
  switch (status) {
    case "running":
      return 5;
    case "error":
      return 2;
    case "waiting":
      return 2;
    case "ready":
      return 1;
    case "done":
      return 0;
    default:
      return 0;
  }
};
const jobComparator = (jobs: { [key: string]: Job }) => {
  return (id1: string, id2: string): number => {
    let j1 = jobs[id1];
    let j2 = jobs[id2];
    let z = status2int(j2.status) - status2int(j1.status);
    if (z !== 0) return z;
    return id1.localeCompare(id2);
  };
};

// Key of the jobs of an experiment run in jobsRequested (null: all
// experiments, current run)
export function jobsKey(experimentId: string | null, runId: string | null): string {
  return `${experimentId ?? "*"}/${runId ?? ""}`;
}

export type ConnectionStatus = "connected" | "connecting" | "disconnected";

export type State = {
  connected: boolean;
  connectionStatus: ConnectionStatus;
  experiment: string; // Kept for backward compatibility
  currentExperiment: string | null; // Currently selected experiment filter (null = all)
  currentRun: string | null; // Currently selected run (null = current/latest)
  route: Route; // What is shown (mirrored in the URL, see routes.ts)
  experiments: Experiments; // All experiments
  experimentsLoaded: boolean; // The server answered the experiments request
  jobs: Jobs;
  jobsRequested: string[]; // Experiment runs whose jobs were requested (jobsKey)
  services: Services;
  warnings: { byKey: { [key: string]: Warning }; keys: string[] };
  actions: { byId: { [key: string]: ActionInfo }; ids: string[] };
  orphans: OrphanJob[];
  runs: Run[];
};

// The page opens on the route of its URL
const initialRoute = currentLocationRoute();
const initialScope = isScoped(initialRoute.view) ? initialRoute : route("experiments");

export const slice = createSlice({
  name: "db",
  initialState: {
    connected: false,
    connectionStatus: "connecting",
    experiment: initialScope.experiment || "",
    currentExperiment: initialScope.experiment,
    currentRun: initialScope.run,
    route: initialRoute,
    experiments: { byId: {}, ids: [] },
    experimentsLoaded: false,
    jobs: { byId: {}, ids: [] },
    jobsRequested: [],
    services: { byId: {}, ids: [] },
    warnings: { byKey: {}, keys: [] },
    actions: { byId: {}, ids: [] },
    orphans: [],
    runs: [],
  } as State,
  reducers: {
    addJob(draft, action: PayloadAction<Job>) {
      if (draft.jobs.byId[action.payload.jobId] === undefined) {
        draft.jobs.ids.push(action.payload.jobId);
      }
      draft.jobs.byId[action.payload.jobId] = action.payload;
      draft.jobs.ids.sort(jobComparator(draft.jobs.byId));
    },

    updateJob(draft, action: PayloadAction<Job>) {
      const jobUpdate = action.payload;

      if (draft.jobs.byId[jobUpdate.jobId] === undefined) {
      } else {
        let job = draft.jobs.byId[jobUpdate.jobId];
        _.merge(job, jobUpdate);
        if (job.progress.length > jobUpdate.progress.length) {
          job.progress = jobUpdate.progress.slice(0, jobUpdate.progress.length);
        }
      }
      draft.jobs.ids.sort(jobComparator(draft.jobs.byId));
    },

    markJobsRequested(draft, { payload }: PayloadAction<string>) {
      draft.jobsRequested.push(payload);
    },

    resetJobsRequested(draft) {
      draft.jobsRequested = [];
    },

    removeJob(draft, { payload }: PayloadAction<{ jobId: string }>) {
      if (draft.jobs.byId[payload.jobId] !== undefined) {
        delete draft.jobs.byId[payload.jobId];
        draft.jobs.ids = draft.jobs.ids.filter((id) => id !== payload.jobId);
      }
    },

    addService(draft, { payload }: PayloadAction<ServiceInformation>) {
      draft.services.byId[payload.key] = payload;
      draft.services.ids = Object.keys(draft.services.byId);
    },

    updateService(draft, { payload }: PayloadAction<Partial<ServiceInformation>>) {
      if (!payload.key) {
        return;
      }
      draft.services.byId[payload.key] = {
        ...draft.services.byId[payload.key],
        ...payload,
      };
    },

    setConnected(draft, action: PayloadAction<boolean>) {
      draft.connected = action.payload;
    },

    setConnectionStatus(draft, action: PayloadAction<ConnectionStatus>) {
      draft.connectionStatus = action.payload;
      draft.connected = action.payload === "connected";
    },

    addExperiment(draft, action: PayloadAction<Experiment>) {
      const exp = action.payload;
      if (draft.experiments.byId[exp.experiment_id] === undefined) {
        draft.experiments.ids.push(exp.experiment_id);
      }
      draft.experiments.byId[exp.experiment_id] = exp;

      // Set as current experiment if it's the first one, unless the route
      // names the experiment(s) shown
      if (draft.experiments.ids.length === 1 && !isScoped(draft.route.view)) {
        draft.currentExperiment = exp.experiment_id;
        draft.experiment = exp.experiment_id; // Backward compatibility
      }
    },

    setExperimentsLoaded(draft) {
      draft.experimentsLoaded = true;
    },

    updateExperiment(
      draft,
      action: PayloadAction<Partial<Experiment> & { experiment_id: string }>,
    ) {
      const exp = action.payload;
      if (draft.experiments.byId[exp.experiment_id] !== undefined) {
        _.merge(draft.experiments.byId[exp.experiment_id], exp);
      }
    },

    removeExperiment(draft, { payload }: PayloadAction<{ experimentId: string }>) {
      const id = payload.experimentId;
      delete draft.experiments.byId[id];
      draft.experiments.ids = draft.experiments.ids.filter((e) => e !== id);
      if (draft.currentExperiment === id) {
        draft.currentExperiment = null;
        draft.experiment = "";
        draft.currentRun = null;
        draft.runs = [];
      }
      if (isScoped(draft.route.view) && draft.route.experiment === id) {
        draft.route = route("experiments");
      }
      // Jobs are kept on disk, but drop those that only belonged to it
      draft.jobs.ids = draft.jobs.ids.filter((jobId) => {
        const xps = draft.jobs.byId[jobId].experimentIds || [];
        if (xps.length === 1 && xps[0] === id) {
          delete draft.jobs.byId[jobId];
          return false;
        }
        return true;
      });
    },

    // Shows a route; a scoped view selects its experiment and run
    navigate(draft, { payload }: PayloadAction<Route>) {
      draft.route = payload;
      if (!isScoped(payload.view)) return;
      if (payload.experiment !== draft.currentExperiment) {
        draft.currentExperiment = payload.experiment;
        draft.experiment = payload.experiment || "";
        draft.currentRun = null;
        draft.runs = [];
      }
      if (payload.run !== draft.currentRun) {
        draft.currentRun = payload.run;
        // The jobs of the run are then requested (App.tsx)
        draft.jobs = { byId: {}, ids: [] };
        draft.jobsRequested = [];
      }
    },


    setRuns(draft, action: PayloadAction<Run[]>) {
      draft.runs = action.payload;
    },


    addWarning(draft, { payload }: PayloadAction<Warning>) {
      if (draft.warnings.byKey[payload.warningKey] === undefined) {
        draft.warnings.keys.push(payload.warningKey);
      }
      draft.warnings.byKey[payload.warningKey] = payload;
    },

    resolveWarning(draft, { payload }: PayloadAction<{ warningKey: string }>) {
      delete draft.warnings.byKey[payload.warningKey];
      draft.warnings.keys = draft.warnings.keys.filter(
        (k) => k !== payload.warningKey,
      );
    },

    addAction(draft, { payload }: PayloadAction<ActionInfo>) {
      if (draft.actions.byId[payload.actionId] === undefined) {
        draft.actions.ids.push(payload.actionId);
      }
      draft.actions.byId[payload.actionId] = payload;
    },

    setOrphans(draft, { payload }: PayloadAction<OrphanJob[]>) {
      draft.orphans = payload;
    },

    removeOrphan(draft, { payload }: PayloadAction<{ jobId: string }>) {
      draft.orphans = draft.orphans.filter((j) => j.jobId !== payload.jobId);
    },
  },
});

const rootReducer = combineReducers({
  db: slice.reducer,
  messages: messageSlice.reducer,
});

export const actions = slice.actions;

export default rootReducer;
