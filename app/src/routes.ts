// Semantic URLs of the web UI, e.g. /xps/<experiment>/services
//
//   /                                     experiments
//   /orphans                              orphan jobs
//   /xps/<xp>/<view>[?run=<run>]          view of an experiment (run)
//   /all/<view>                           view of all the experiments
//   .../<view>/<task>/<job>[/logs]        job detail (details or logs tab)
//
// with <view> one of jobs, graph, services, actions. The route is kept in
// the store (state.db.route): the components navigate by dispatching
// actions.navigate, and App keeps the browser URL in sync with it.

import type React from "react";

import type { JobTab } from "./JobModal";

export type View =
  | "experiments"
  | "tasks"
  | "graph"
  | "orphans"
  | "services"
  | "actions";

// Views shown for an experiment (or all of them)
export type ScopedView = "tasks" | "graph" | "services" | "actions";

export type JobRoute = { taskId: string; jobId: string; tab: JobTab };

export type Route = {
  view: View;
  // Experiment and run of a scoped view (null = all / current run)
  experiment: string | null;
  run: string | null;
  // Job whose detail is shown (tasks and graph views)
  job: JobRoute | null;
};

const SEGMENTS: { [view in ScopedView]: string } = {
  tasks: "jobs",
  graph: "graph",
  services: "services",
  actions: "actions",
};

const VIEWS: { [segment: string]: ScopedView } = Object.fromEntries(
  Object.entries(SEGMENTS).map(([view, segment]) => [segment, view as ScopedView]),
);

export function isScoped(view: View): view is ScopedView {
  return view in SEGMENTS;
}

export function route(
  view: View,
  experiment: string | null = null,
  run: string | null = null,
): Route {
  return { view, experiment, run, job: null };
}

export function parseRoute(pathname: string, search: string): Route {
  const parts = pathname
    .split("/")
    .filter((p) => p !== "")
    .map(decodeURIComponent);
  const run = new URLSearchParams(search).get("run");

  if (parts[0] === "orphans") return route("orphans");

  let experiment: string | null;
  let rest: string[];
  if (parts[0] === "xps" && parts.length >= 2) {
    experiment = parts[1];
    rest = parts.slice(2);
  } else if (parts[0] === "all") {
    experiment = null;
    rest = parts.slice(1);
  } else {
    return route("experiments");
  }

  const view = VIEWS[rest[0] ?? "jobs"] ?? "tasks";
  const job =
    (view === "tasks" || view === "graph") && rest.length >= 3
      ? {
          taskId: rest[1],
          jobId: rest[2],
          tab: (rest[3] === "logs" ? "logs" : "details") as JobTab,
        }
      : null;
  return { view, experiment, run: experiment === null ? null : run, job };
}

export function formatRoute(r: Route): string {
  if (r.view === "experiments") return "/";
  if (r.view === "orphans") return "/orphans";

  const parts = r.experiment === null ? ["all"] : ["xps", r.experiment];
  parts.push(SEGMENTS[r.view]);
  if (r.job) {
    parts.push(r.job.taskId, r.job.jobId);
    if (r.job.tab === "logs") parts.push("logs");
  }
  const path = "/" + parts.map(encodeURIComponent).join("/");
  return r.run !== null && r.experiment !== null
    ? `${path}?run=${encodeURIComponent(r.run)}`
    : path;
}

export function currentLocationRoute(): Route {
  return parseRoute(window.location.pathname, window.location.search);
}

// A link to a route: a real href (middle-click opens a new tab), while a
// plain click navigates within the page
export function linkProps(
  r: Route,
  navigate: (r: Route) => void,
): { href: string; onClick: (e: React.MouseEvent) => void } {
  return {
    href: formatRoute(r),
    onClick: (e: React.MouseEvent) => {
      if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      e.preventDefault();
      navigate(r);
    },
  };
}
