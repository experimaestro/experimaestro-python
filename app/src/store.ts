import { writable } from "svelte/store";
import produce from "immer";
import _ from "lodash";

type Experiment = {
  name: string;
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
};

export type Jobs = {
  byId: { [key: string]: Job };
  ids: Array<string>;
};

// export type State = {
//     connected: boolean;

//     experiment: ?string;
//     jobs: Jobs,
//     experiments: {[string]: Experiment};
// }
// export const state = writable<State>({
//     connected: false,
//     jobs: {
//         byId: {},
//         ids: []
//     },
//     experiments: {},
//     experiment: null
// })

export const connected = writable(false);

export const experiment = writable("");

export const jobs = writable<Jobs>({ byId: {}, ids: [] });

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

export function process(action: any) {
  switch (action.type) {
    case "JOB_ADD": {
      jobs.update((current) =>
        produce(current, (draft) => {
          if (draft.byId[action.payload.jobId] === undefined) {
            draft.ids.push(action.payload.jobId);
          }
          draft.byId[action.payload.jobId] = action.payload;
          draft.ids.sort(jobComparator(draft.byId));
        })
      );
      break;
    }

    case "JOB_UPDATE":
      jobs.update((current) =>
        produce(current, (draft) => {
          const jobUpdate = <Job>action.payload;

          if (draft.byId[jobUpdate.jobId] === undefined) {
          } else {
            let job = draft.byId[jobUpdate.jobId];
            _.merge(job, jobUpdate);
            if (job.progress.length > jobUpdate.progress.length) {
              job.progress = jobUpdate.progress.slice(
                0,
                jobUpdate.progress.length
              );
            }
          }
          draft.ids.sort(jobComparator(draft.byId));
        })
      );
      break;
  }
}
