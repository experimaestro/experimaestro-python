// @flow

import { Reducer } from 'redux'
import produce from "immer"
import _ from 'lodash';

// ---- States

type Experiment = {
    name: string;
}

type JobStatus = "running" | "done" | "error" | "ready" | "waiting";

export type Job = {
    jobId: string;
    taskId: string;

    locator: string;
    status: JobStatus;

    start: number;
    end: number;
    submitted: number;

    tags: Array<[string, number|string|boolean]>;

    progress: number;
}
export type Jobs = {
    byId: { [string]: Job },
    ids: Array<string>
}

export type State = {
    connected: boolean;

    experiment: ?string;
    jobs: Jobs,
    experiments: {[string]: Experiment};
}


// ---- Actions

type Action = 
    { type: "CONNECTED", payload: boolean } // true if the connexion is open, false otherwise
    
    | { type: "CLEAN_INFORMATION" }

    | { type: "EXPERIMENT_ADD", payload: Experiment }
    | { type: "EXPERIMENT_SET_MAIN", payload: string }

    | { type: "JOB_ADD", payload: Job }
    | { type: "JOB_UPDATE", payload: $Shape<Job> }
;



// --- Reducer

export const initialState : State = {
    connected: false,
    jobs: {
        byId: {},
        ids: []
    },
    experiments: {},
    experiment: null
}


const status2int = (status: JobStatus) : number => {
    switch(status) {
        case "running": return 5;
        case "error": return 2;
        case "waiting": return 2;
        case "ready": return 1;
        case "done": return 0;
        default: return 0;
    }
}
const jobComparator = (jobs: {[string]: Job}) => {
    return (id1: string, id2: string) : number => {
        let j1 = jobs[id1];
        let j2 = jobs[id2];
        let z =  status2int(j2.status) - status2int(j1.status);
        if (z !== 0) return z;
        return id1.localeCompare(id2);
    }
}

export const reducer : Reducer<State,Action> =
    produce((draft: State, action: Action) : void => {
        switch (action.type) {
            case "CONNECTED":
                draft.connected = action.payload;
                break;
            case "CLEAN_INFORMATION":
                draft.jobs = {
                    byId: {},
                    ids: []
                };
                draft.experiments = {};
                break;

            case "EXPERIMENT_SET_MAIN":
                draft.experiment = action.payload;
                break;

            case "EXPERIMENT_ADD":
                draft.experiments[action.payload.name] = action.payload;
                break;

            case "JOB_ADD":
                if (draft.jobs.byId[action.payload.jobId] === undefined) {
                    draft.jobs.ids.push(action.payload.jobId);
                }
                draft.jobs.byId[action.payload.jobId] = action.payload;
                draft.jobs.ids.sort(jobComparator(draft.jobs.byId));
                break;

            case "JOB_UPDATE":
                if (draft.jobs.byId[action.payload.jobId] === undefined) {
                } else {
                    _.merge(draft.jobs.byId[action.payload.jobId], action.payload);
                }
                draft.jobs.ids.sort(jobComparator(draft.jobs.byId));
                break;

            default: 
                break;
        }
    }, initialState)
