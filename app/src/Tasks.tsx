import React, { useState, useMemo, useEffect, useRef } from "react";
import { Badge, Button, Dropdown, Modal, Form, Table } from "react-bootstrap";

import TaskJobs, { JOB_COLUMNS } from "./TaskJobs";
import JobModal, { JobTab } from "./JobModal";
import client, { JobsFilterResult } from "./client";
import { Job } from "./reducers";
import { useMessages } from "./ui/messages";
import { useAppSelector, useNavigate } from "./store";
import { clearOnEscape, useShortcuts } from "./shortcuts";
import { KeyHints, SortHeader, SortLabel, sortedMessage } from "./ui/sorting";

type SortKey = "status" | "task" | "date" | "name";

// Direction of a sort chosen with its key, as in the TUI (by rank for
// statuses, otherwise oldest first or alphabetical)
const SORT_ASC: { [key in SortKey]: boolean } = {
  status: false,
  task: true,
  date: true,
  name: true,
};
type GroupKey = "none" | "status" | "task" | "tag";

const SORT_LABELS: { [key in SortKey]: { label: string; shortcut: string } } = {
  status: { label: "Status", shortcut: "S" },
  task: { label: "Task", shortcut: "T" },
  date: { label: "Submitted", shortcut: "D" },
  name: { label: "Job ID", shortcut: "O" },
};

// Same hint as the TUI search bar (FILTER_SYNTAX_HINT in cli/filter.py)
const FILTER_SYNTAX_HINT =
  "Syntax: @state = 'done' | @name ~ 'regex' | tag = 'value' | and/or";

// Delay before sending the filter expression to the server
const FILTER_DEBOUNCE_MS = 300;

// Rank of a status, from the server-side order shared with the TUI
// (scheduler/job_status.py): the best rank among the jobs having it
function statusRanks(jobs: Job[]): Map<string, number> {
  const ranks = new Map<string, number>();
  for (const job of jobs) {
    const rank = ranks.get(job.status);
    if (rank === undefined || job.statusRank < rank) ranks.set(job.status, job.statusRank);
  }
  return ranks;
}

type JobGroup = {
  key: string;
  label: string;
  jobs: Job[];
};

function groupJobs(jobs: Job[], groupBy: GroupKey, tagKey: string): JobGroup[] {
  const groups = new Map<string, Job[]>();
  const keyOf = (job: Job): string => {
    switch (groupBy) {
      case "status":
        return job.status;
      case "task":
        return job.taskId;
      default: {
        const tag = job.tags.find(([k]) => k === tagKey);
        return tag === undefined ? "" : String(tag[1]);
      }
    }
  };
  for (const job of jobs) {
    const key = keyOf(job);
    const group = groups.get(key);
    if (group) group.push(job);
    else groups.set(key, [job]);
  }

  const result: JobGroup[] = Array.from(groups.entries()).map(([key, jobs]) => ({
    key,
    label: key === "" ? `(no ${tagKey})` : key,
    jobs,
  }));
  const ranks = statusRanks(jobs);
  result.sort((a, b) => {
    if (groupBy === "status") {
      const z = (ranks.get(a.key) ?? 0) - (ranks.get(b.key) ?? 0);
      if (z !== 0) return z;
    }
    // Jobs without the tag go last
    if (a.key === "" || b.key === "") return a.key === "" ? 1 : -1;
    return a.key.localeCompare(b.key, undefined, { numeric: true });
  });
  return result;
}

function statusCounts(jobs: Job[]): Array<[string, number]> {
  const counts = new Map<string, number>();
  for (const job of jobs) counts.set(job.status, (counts.get(job.status) || 0) + 1);
  const ranks = statusRanks(jobs);
  return Array.from(counts.entries()).sort(
    (a, b) => (ranks.get(a[0]) ?? 0) - (ranks.get(b[0]) ?? 0) || a[0].localeCompare(b[0]),
  );
}

// Tag keys hidden in the job list (for all the experiments)
const HIDDEN_TAGS_STORAGE_KEY = "experimaestro.jobs.hiddenTags";

function loadHiddenTags(): Set<string> {
  try {
    const saved = JSON.parse(localStorage.getItem(HIDDEN_TAGS_STORAGE_KEY) ?? "[]");
    if (Array.isArray(saved)) return new Set(saved.map(String));
  } catch {
    // Storage unavailable or invalid content: show all the tags
  }
  return new Set();
}

function saveHiddenTags(keys: Set<string>) {
  try {
    localStorage.setItem(HIDDEN_TAGS_STORAGE_KEY, JSON.stringify(Array.from(keys)));
  } catch {
    // Storage unavailable: the hidden tags are not remembered
  }
}

const STATUS_COLOR: { [key: string]: string } = {
  running: "primary",
  done: "success",
  error: "danger",
  waiting: "warning",
  ready: "info",
};

export default () => {
  const [query, setQuery] = useState<string>("");
  const [filterResult, setFilterResult] = useState<{
    ids: Set<string> | null;
    error: string | null;
  }>({ ids: null, error: null });
  const [confirm, setConfirm] = useState<{ job: Job; kind: "kill" | "delete" }>();
  const [sortKey, setSortKey] = useState<SortKey>("status");
  const [sortAsc, setSortAsc] = useState<boolean>(false);
  const [groupBy, setGroupBy] = useState<GroupKey>("none");
  const [tagKey, setTagKey] = useState<string>("");
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [hiddenTags, setHiddenTags] = useState<Set<string>>(loadHiddenTags);
  const { info } = useMessages();
  const jobs = useAppSelector((state) => state.db.jobs);
  const currentExperiment = useAppSelector((state) => state.db.currentExperiment);
  const currentRun = useAppSelector((state) => state.db.currentRun);
  const route = useAppSelector((state) => state.db.route);
  const navigate = useNavigate();
  const searchInput = useRef<HTMLInputElement>(null);

  function changeSort(key: SortKey, ascending: boolean) {
    setSortKey(key);
    setSortAsc(ascending);
    info(sortedMessage(SORT_LABELS[key].label, ascending));
  }

  // The key of the current sort reverses it
  function sortBy(key: SortKey) {
    changeSort(key, key === sortKey ? !sortAsc : SORT_ASC[key]);
  }

  const sortProps = (key: SortKey) => ({
    column: key,
    label: SORT_LABELS[key].label,
    shortcut: SORT_LABELS[key].shortcut,
    active: sortKey === key,
    ascending: sortAsc,
    onClick: () => sortBy(key),
  });

  const sortHeader = (key: SortKey) => (
    <SortHeader
      column={key}
      label={SORT_LABELS[key].label}
      shortcut={SORT_LABELS[key].shortcut}
      active={sortKey === key}
      ascending={sortAsc}
      onClick={() => sortBy(key)}
    />
  );

  useShortcuts({
    "/": () => searchInput.current?.focus(),
    S: () => sortBy("status"),
    T: () => sortBy("task"),
    D: () => sortBy("date"),
    O: () => sortBy("name"),
  });

  // --- Server-side filtering (same DSL and semantics as the TUI)

  const requestId = useRef(0);
  const trimmedQuery = query.trim();

  useEffect(() => {
    client.onJobsFilter = (result: JobsFilterResult) => {
      // Ignore answers to outdated requests
      if (result.requestId !== requestId.current) return;
      setFilterResult({
        ids: result.jobIds ? new Set(result.jobIds) : null,
        error: result.error,
      });
    };
    return () => {
      client.onJobsFilter = null;
    };
  }, []);

  // The filter only depends on the job set, states and tags: re-evaluate it
  // when jobs are added/removed or change state, not on progress updates
  const jobsSignature = useMemo(
    () => jobs.ids.map((id) => `${id}:${jobs.byId[id].status}`).join(","),
    [jobs.ids, jobs.byId],
  );

  useEffect(() => {
    if (!trimmedQuery) {
      requestId.current += 1;
      setFilterResult({ ids: null, error: null });
      return;
    }
    const timer = window.setTimeout(() => {
      requestId.current += 1;
      client.jobs_filter(trimmedQuery, requestId.current, currentExperiment, currentRun);
    }, FILTER_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [trimmedQuery, currentExperiment, currentRun, jobsSignature]);

  const filteredJobs = useMemo(() => {
    function filter(job: Job) {
      // Filter by experiment if one is selected
      if (currentExperiment !== null) {
        if (!job.experimentIds || !job.experimentIds.includes(currentExperiment)) {
          return false;
        }
      }
      if (trimmedQuery && filterResult.ids !== null) {
        return filterResult.ids.has(job.jobId);
      }
      return true;
    }

    const result = jobs.ids.map((id) => jobs.byId[id]).filter(filter);

    const cmp = (a: Job, b: Job): number => {
      let z = 0;
      switch (sortKey) {
        case "status":
          // Lower ranks come first in the default (descending) order
          z = b.statusRank - a.statusRank;
          if (z === 0) z = a.jobId.localeCompare(b.jobId);
          break;
        case "task":
          z = a.taskId.localeCompare(b.taskId);
          break;
        case "date":
          z = (a.submitted || "").localeCompare(b.submitted || "");
          break;
        case "name":
          z = a.jobId.localeCompare(b.jobId);
          break;
      }
      return sortAsc ? z : -z;
    };

    return result.sort(cmp);
  }, [
    jobs.ids,
    jobs.byId,
    trimmedQuery,
    filterResult,
    currentExperiment,
    sortKey,
    sortAsc,
  ]);

  // --- Tags shown

  // Keys of the tags of the experiment jobs (and those hidden)
  const allTagKeys = useMemo(() => {
    const keys = new Set<string>(hiddenTags);
    for (const id of jobs.ids) {
      const job = jobs.byId[id];
      if (currentExperiment !== null && !job.experimentIds?.includes(currentExperiment))
        continue;
      for (const [k] of job.tags) keys.add(k);
    }
    return Array.from(keys).sort();
  }, [jobs.ids, jobs.byId, currentExperiment, hiddenTags]);

  function toggleTag(key: string) {
    const next = new Set(hiddenTags);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    setHiddenTags(next);
    saveHiddenTags(next);
  }

  // --- Grouping

  const tagKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const job of filteredJobs) for (const [k] of job.tags) keys.add(k);
    return Array.from(keys).sort();
  }, [filteredJobs]);

  const effectiveTagKey = tagKeys.includes(tagKey) ? tagKey : tagKeys[0] || "";

  const groups = useMemo(
    () =>
      groupBy === "none" || (groupBy === "tag" && !effectiveTagKey)
        ? null
        : groupJobs(filteredJobs, groupBy, effectiveTagKey),
    [filteredJobs, groupBy, effectiveTagKey],
  );

  function toggleGroup(key: string) {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  function changeGrouping(value: GroupKey) {
    setGroupBy(value);
    setCollapsed(new Set());
  }

  // --- Job actions

  function cancelConfirm() {
    setConfirm(undefined);
    info("Action cancelled");
  }

  function runConfirm() {
    if (!confirm) return;
    const { job, kind } = confirm;
    if (kind === "kill") client.job_kill(job.jobId, job.taskId);
    else client.job_delete(job.jobId, job.taskId);
    setConfirm(undefined);
  }

  const showJob = (job: Job, tab: JobTab) =>
    navigate({ ...route, job: { taskId: job.taskId, jobId: job.jobId, tab } });

  const renderJob = (job: Job) => (
    <TaskJobs
      key={job.jobId}
      job={job}
      onKill={() => setConfirm({ job, kind: "kill" })}
      onDelete={() => setConfirm({ job, kind: "delete" })}
      onShow={() => showJob(job, "details")}
      onLogs={() => showJob(job, "logs")}
      hiddenTags={hiddenTags}
    />
  );

  return (
    <div id="resources">
      <>
        <div className="search">
          <div
            style={{
              display: "flex",
              alignItems: "flex-end",
              gap: "0.5rem",
              flexWrap: "wrap",
            }}
          >
            <Form.Group style={{ flex: "1 1 24rem" }}>
              <Form.Label htmlFor="searchjobs">Filter</Form.Label>
              <Form.Control
                id="searchjobs"
                ref={searchInput}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={clearOnEscape(() => setQuery(""))}
                placeholder="Filter (/): @state = 'done', @name ~ 'pattern', tag = 'value'"
                isInvalid={!!trimmedQuery && filterResult.error !== null}
                isValid={
                  !!trimmedQuery &&
                  filterResult.error === null &&
                  filterResult.ids !== null
                }
              />
            </Form.Group>
            <Form.Group>
              <Form.Label htmlFor="sortkey">Sort</Form.Label>
              <Form.Select
                id="sortkey"
                size="sm"
                value={sortKey}
                onChange={(e) => sortBy(e.target.value as SortKey)}
              >
                {(Object.keys(SORT_LABELS) as SortKey[]).map((key) => (
                  <option key={key} value={key}>
                    {SORT_LABELS[key].label} ({SORT_LABELS[key].shortcut})
                  </option>
                ))}
              </Form.Select>
            </Form.Group>
            <Button
              size="sm"
              variant="outline-secondary"
              title="Toggle sort direction"
              onClick={() => changeSort(sortKey, !sortAsc)}
            >
              <i className={`fas fa-arrow-${sortAsc ? "up" : "down"}`} />
            </Button>
            <Form.Group>
              <Form.Label htmlFor="tagvisibility">Tags</Form.Label>
              <Dropdown autoClose="outside">
                <Dropdown.Toggle id="tagvisibility" size="sm" variant="outline-secondary">
                  {hiddenTags.size > 0 ? `${hiddenTags.size} hidden` : "All shown"}
                </Dropdown.Toggle>
                <Dropdown.Menu className="tag-visibility-menu">
                  {allTagKeys.length === 0 && (
                    <Dropdown.ItemText className="text-muted">No tags</Dropdown.ItemText>
                  )}
                  {allTagKeys.map((k) => (
                    <Dropdown.ItemText key={k}>
                      <Form.Check
                        type="checkbox"
                        id={`tag-visible-${k}`}
                        data-tag-key={k}
                        label={k}
                        checked={!hiddenTags.has(k)}
                        onChange={() => toggleTag(k)}
                      />
                    </Dropdown.ItemText>
                  ))}
                </Dropdown.Menu>
              </Dropdown>
            </Form.Group>
            <Form.Group>
              <Form.Label htmlFor="groupby">Group by</Form.Label>
              <Form.Select
                id="groupby"
                size="sm"
                value={groupBy}
                onChange={(e) => changeGrouping(e.target.value as GroupKey)}
              >
                <option value="none">None</option>
                <option value="status">Status</option>
                <option value="task">Task</option>
                <option value="tag">Tag</option>
              </Form.Select>
            </Form.Group>
            {groupBy === "tag" && (
              <Form.Group>
                <Form.Label htmlFor="grouptag">Tag</Form.Label>
                <Form.Select
                  id="grouptag"
                  size="sm"
                  value={effectiveTagKey}
                  disabled={tagKeys.length === 0}
                  onChange={(e) => {
                    setTagKey(e.target.value);
                    setCollapsed(new Set());
                  }}
                >
                  {tagKeys.length === 0 && <option value="">(no tags)</option>}
                  {tagKeys.map((k) => (
                    <option key={k} value={k}>
                      {k}
                    </option>
                  ))}
                </Form.Select>
              </Form.Group>
            )}
            {groups !== null && (
              <>
                <Button
                  size="sm"
                  variant="outline-secondary"
                  title="Expand all groups"
                  onClick={() => setCollapsed(new Set())}
                >
                  <i className="fas fa-angles-down" />
                </Button>
                <Button
                  size="sm"
                  variant="outline-secondary"
                  title="Collapse all groups"
                  onClick={() => setCollapsed(new Set(groups.map((g) => g.key)))}
                >
                  <i className="fas fa-angles-up" />
                </Button>
              </>
            )}
          </div>
          {trimmedQuery && filterResult.error !== null ? (
            <div className="small text-danger mt-1">{filterResult.error}</div>
          ) : (
            <div className="small text-muted mt-1">{FILTER_SYNTAX_HINT}</div>
          )}
          <KeyHints
            hints={[
              ["/", "filter"],
              ["S", "sort by status"],
              ["T", "sort by task"],
              ["D", "sort by date"],
              ["O", "sort by job ID"],
            ]}
          />
        </div>

        {confirm && (
          <Modal show={true} onHide={cancelConfirm}>
            <Modal.Header closeButton>
              <Modal.Title>Are you sure?</Modal.Title>
            </Modal.Header>
            <Modal.Body>
              {confirm.kind === "kill" ? "Kill" : "Delete"} job{" "}
              <b>{confirm.job.taskId}</b>?
              {confirm.kind === "delete" && (
                <div className="text-danger small mt-2">
                  This removes the job directory from disk.
                </div>
              )}
            </Modal.Body>
            <Modal.Footer>
              <Button variant="secondary" onClick={cancelConfirm}>
                Cancel
              </Button>
              <Button
                variant={confirm.kind === "delete" ? "danger" : "primary"}
                onClick={runConfirm}
              >
                {confirm.kind === "kill" ? "Kill" : "Delete"}
              </Button>
            </Modal.Footer>
          </Modal>
        )}

        <Table hover responsive className="align-middle jobs-table">
          <thead>
            <tr>
              {sortHeader("task")}
              {sortHeader("status")}
              {/* The duration is below the submission date */}
              <th>
                <SortLabel {...sortProps("date")} />
                {" · Duration"}
              </th>
              <th>CO₂</th>
              <th></th>
            </tr>
          </thead>
          {groups === null ? (
            <tbody>{filteredJobs.map(renderJob)}</tbody>
          ) : (
            groups.map((group) => {
              const isCollapsed = collapsed.has(group.key);
              return (
                <tbody key={`group-${group.key}`}>
                  <tr
                    className="job-group-row"
                    onClick={() => toggleGroup(group.key)}
                    title={isCollapsed ? "Expand" : "Collapse"}
                  >
                    <td colSpan={JOB_COLUMNS}>
                      <i
                        className={`fas fa-chevron-${isCollapsed ? "right" : "down"} me-2`}
                      />
                      {groupBy === "tag" && (
                        <span className="text-muted me-1">{effectiveTagKey} =</span>
                      )}
                      <span className="fw-semibold">{group.label}</span>
                      <Badge bg="secondary" className="ms-2">
                        {group.jobs.length}
                      </Badge>
                      {groupBy !== "status" &&
                        statusCounts(group.jobs).map(([status, count]) => (
                          <Badge
                            key={status}
                            bg={STATUS_COLOR[status] || "light"}
                            text={
                              ["warning", "info", undefined].includes(
                                STATUS_COLOR[status],
                              )
                                ? "dark"
                                : undefined
                            }
                            className="ms-1 fw-normal"
                          >
                            {count} {status}
                          </Badge>
                        ))}
                    </td>
                  </tr>
                  {!isCollapsed && group.jobs.map(renderJob)}
                </tbody>
              );
            })
          )}
        </Table>
        {filteredJobs.length === 0 && (
          <p className="text-muted">No jobs match the current filter.</p>
        )}
        <JobModal />
      </>
    </div>
  );
};
