import { combineReducers, createSlice, PayloadAction } from "@reduxjs/toolkit";
import { messageSlice } from "./ui/messages";
import _ from "lodash";

export type Experiment = {
  experiment_id: string;
  workdir?: string;
  total_jobs: number;
  finished_jobs: number;
  failed_jobs: number;
  created_at?: string;
  updated_at?: string;
};

export type Experiments = {
  byId: { [key: string]: Experiment };
  ids: Array<string>;
};

type JobStatus = "running" | "done" | "error" | "ready" | "waiting";

export type Job = {
  jobId: string;
  taskId: string;

  locator: string;
  status: JobStatus;

  start: number;
  end: number;
  submitted: number;

  tags: Array<[string, number | string | boolean]>;

  progress: Array<{
    level: number;
    desc: string | null;
    progress: number;
  }>;

  experimentIds?: Array<string>;  // Jobs can belong to multiple experiments
  dependsOn?: Array<string>;  // Job IDs this job depends on
};

export type Jobs = {
  byId: { [key: string]: Job };
  ids: Array<string>;
};

type ServiceStatus = "STOPPED" | "STARTING" | "RUNNING" | "STOPPING"

type  ServiceInformation = {
  "id": string,
  "description": string,
  "state": ServiceStatus
}

export type Services = {
  byId: { [key: string]: ServiceInformation };
  ids: string[]
}

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

export type State = {
  connected: boolean;
  experiment: string;  // Kept for backward compatibility
  currentExperiment: string | null;  // Currently selected experiment filter (null = all)
  experiments: Experiments;  // All experiments
  jobs: Jobs;
  services: Services;
};



export const slice = createSlice({
  name: "db",
  initialState: {
    connected: false,
    experiment: "",
    currentExperiment: null,
    experiments: { byId: {}, ids: [] },
    jobs: { byId: {}, ids: [] },
    services: { byId: {}, ids: [] }
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

    addService(draft, {payload}: PayloadAction<ServiceInformation>) {
      draft.services.byId[payload.id] = payload
      draft.services.ids = Object.keys(draft.services.byId)
    },

    updateService(draft, {payload}: PayloadAction<Partial<ServiceInformation>>) {
      if (!payload.id) {
        return;
      }
      draft.services.byId[payload.id] = {...draft.services.byId[payload.id], ...payload}
    },

    setConnected(draft, action: PayloadAction<boolean>) {
      draft.connected = action.payload;
    },

    addExperiment(draft, action: PayloadAction<Experiment>) {
      const exp = action.payload;
      if (draft.experiments.byId[exp.experiment_id] === undefined) {
        draft.experiments.ids.push(exp.experiment_id);
      }
      draft.experiments.byId[exp.experiment_id] = exp;

      // Set as current experiment if it's the first one
      if (draft.experiments.ids.length === 1) {
        draft.currentExperiment = exp.experiment_id;
        draft.experiment = exp.experiment_id;  // Backward compatibility
      }
    },

    updateExperiment(draft, action: PayloadAction<Partial<Experiment> & { experiment_id: string }>) {
      const exp = action.payload;
      if (draft.experiments.byId[exp.experiment_id] !== undefined) {
        _.merge(draft.experiments.byId[exp.experiment_id], exp);
      }
    },

    selectExperiment(draft, action: PayloadAction<string | null>) {
      draft.currentExperiment = action.payload;
      // Update backward compatibility field
      draft.experiment = action.payload || "";
    },
  },
});

const rootReducer = combineReducers({
  db: slice.reducer,
  messages: messageSlice.reducer,
});

export const actions = slice.actions;

export default rootReducer;
