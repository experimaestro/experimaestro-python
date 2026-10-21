import React, { useMemo, useRef, useState } from "react";
import { Table, Badge, Modal, Button, Form } from "react-bootstrap";

import { useAppSelector } from "./store";
import { Experiment } from "./reducers";
import { formatShortDate, formatDuration, formatCO2 } from "./format";
import client from "./client";
import { useMessages } from "./ui/messages";
import { clearOnEscape, useShortcuts } from "./shortcuts";
import { KeyHints, SortHeader, sortedMessage } from "./ui/sorting";
import { ExperimentTags } from "./ui/tags";

// Status computed by the server, shared with the TUI (status_category)
const STATUS_INFO: { [key: string]: { icon: string; color: string } } = {
  failed: { icon: "❌", color: "danger" },
  running: { icon: "🏃", color: "primary" },
  done: { icon: "✓", color: "success" },
  stopped: { icon: "⏹", color: "secondary" },
  empty: { icon: "∅", color: "secondary" },
};

function statusInfo(exp: Experiment): { icon: string; color: string; label: string } {
  const label = exp.statusCategory;
  return { ...(STATUS_INFO[label] ?? STATUS_INFO.empty), label };
}

type Confirm = { experimentId: string; kind: "kill" | "delete" };

// Sort orders of the TUI (tui/widgets/experiments.py), with the same keys
type SortColumn = "status" | "started" | "id";
type Sort = { column: SortColumn; reverse: boolean };

// Newest first by default
const DEFAULT_SORT: Sort = { column: "started", reverse: true };
const SORT_STORAGE_KEY = "experimaestro.experiments.sort";

// Running experiments first, then those needing attention
const STATUS_ORDER: { [key: string]: number } = {
  running: 0,
  failed: 1,
  stopped: 2,
  done: 3,
  empty: 4,
};

function compareExperiments(column: SortColumn) {
  const started = (exp: Experiment) => exp.started_at ?? "";
  return (a: Experiment, b: Experiment): number => {
    switch (column) {
      case "status":
        return (
          (STATUS_ORDER[a.statusCategory] ?? 99) -
            (STATUS_ORDER[b.statusCategory] ?? 99) ||
          // Newest first among experiments with the same status
          started(b).localeCompare(started(a))
        );
      case "started":
        return started(a).localeCompare(started(b));
      case "id":
        return a.experiment_id
          .toLowerCase()
          .localeCompare(b.experiment_id.toLowerCase());
    }
  };
}

function loadSort(): Sort {
  try {
    const saved = JSON.parse(localStorage.getItem(SORT_STORAGE_KEY) ?? "null");
    if (saved && ["status", "started", "id"].includes(saved.column)) {
      return { column: saved.column, reverse: !!saved.reverse };
    }
  } catch {
    // Storage unavailable or invalid content: use the default
  }
  return DEFAULT_SORT;
}

function saveSort(sort: Sort) {
  try {
    localStorage.setItem(SORT_STORAGE_KEY, JSON.stringify(sort));
  } catch {
    // Storage unavailable: the order is not remembered
  }
}

// Grouping of the experiments, the same as in the TUI (ui/experiments.py)
type Grouping = "none" | "prefix" | "tag";
const GROUPINGS: Grouping[] = ["none", "prefix", "tag"];
const GROUPING_LABELS: { [grouping in Grouping]: string } = {
  none: "None",
  prefix: "Prefix",
  tag: "Tag",
};
const GROUPING_STORAGE_KEY = "experimaestro.experiments.grouping";

// Group of the experiments without tags
const UNTAGGED = "(untagged)";

// What comes before the first "-", "_" or "." (the whole ID when there is none)
function experimentPrefix(id: string): string {
  return id.split(/[-_.]/, 1)[0];
}

// The experiment ID or one of its tags contains the text
function matches(exp: Experiment, text: string): boolean {
  return (
    exp.experiment_id.toLowerCase().includes(text) ||
    (exp.tags ?? []).some((tag) => tag.toLowerCase().includes(text))
  );
}

type ExperimentGroup = { name: string; experiments: Experiment[] };

// Sorted by name (as in Python), the experiments without tags last
function compareGroupNames(a: string, b: string): number {
  if (a === UNTAGGED || b === UNTAGGED)
    return Number(a === UNTAGGED) - Number(b === UNTAGGED);
  return a < b ? -1 : a > b ? 1 : 0;
}

/**
 * Groups sorted by name, keeping the order of the experiments; with the tag
 * grouping, an experiment is in the group of each of its tags
 */
function groupExperiments(
  experiments: Experiment[],
  grouping: Grouping,
): ExperimentGroup[] {
  const groups = new Map<string, Experiment[]>();
  for (const exp of experiments) {
    let names: string[];
    if (grouping === "prefix") names = [experimentPrefix(exp.experiment_id)];
    else names = exp.tags?.length ? Array.from(new Set(exp.tags)) : [UNTAGGED];
    for (const name of names) {
      const group = groups.get(name);
      if (group) group.push(exp);
      else groups.set(name, [exp]);
    }
  }
  return Array.from(groups.entries())
    .map(([name, experiments]) => ({ name, experiments }))
    .sort((a, b) => compareGroupNames(a.name, b.name));
}

function loadGrouping(): Grouping {
  try {
    const saved = localStorage.getItem(GROUPING_STORAGE_KEY);
    if (saved && (GROUPINGS as string[]).includes(saved)) return saved as Grouping;
  } catch {
    // Storage unavailable: no grouping
  }
  return "none";
}

function saveGrouping(grouping: Grouping) {
  try {
    localStorage.setItem(GROUPING_STORAGE_KEY, grouping);
  } catch {
    // Storage unavailable: the grouping is not remembered
  }
}

// Number of experiments per status, in the status order
function statusCounts(experiments: Experiment[]): Array<[string, number]> {
  const counts = new Map<string, number>();
  for (const exp of experiments)
    counts.set(exp.statusCategory, (counts.get(exp.statusCategory) ?? 0) + 1);
  return Array.from(counts.entries()).sort(
    (a, b) => (STATUS_ORDER[a[0]] ?? 99) - (STATUS_ORDER[b[0]] ?? 99),
  );
}

const SORT_HEADERS: { [column in SortColumn]: { label: string; key: string } } = {
  status: { label: "Status", key: "S" },
  id: { label: "Experiment", key: "O" },
  started: { label: "Started", key: "D" },
};

export default ({ onOpen }: { onOpen: (id: string) => void }) => {
  const experiments = useAppSelector((s) => s.db.experiments);
  const loaded = useAppSelector((s) => s.db.experimentsLoaded);
  const [confirm, setConfirm] = useState<Confirm>();
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<Sort>(loadSort);
  const [grouping, setGrouping] = useState<Grouping>(loadGrouping);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const searchInput = useRef<HTMLInputElement>(null);
  const { info } = useMessages();

  function changeSort(next: Sort) {
    setSort(next);
    saveSort(next);
    info(sortedMessage(SORT_HEADERS[next.column].label, !next.reverse));
  }

  // The same key again reverses the order; dates start with the newest
  function sortBy(column: SortColumn) {
    changeSort(
      sort.column === column
        ? { column, reverse: !sort.reverse }
        : { column, reverse: column === "started" },
    );
  }

  function changeGrouping(next: Grouping) {
    setGrouping(next);
    saveGrouping(next);
    setCollapsed(new Set());
    info(next === "none" ? "Not grouped" : `Grouped by ${next}`);
  }

  function toggleGroup(name: string) {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  useShortcuts({
    "/": () => searchInput.current?.focus(),
    S: () => sortBy("status"),
    D: () => sortBy("started"),
    O: () => sortBy("id"),
    g: () =>
      changeGrouping(GROUPINGS[(GROUPINGS.indexOf(grouping) + 1) % GROUPINGS.length]),
  });

  const shown = useMemo(() => {
    const text = search.trim().toLowerCase();
    const cmp = compareExperiments(sort.column);
    return experiments.ids
      .map((id) => experiments.byId[id])
      .filter((exp) => matches(exp, text))
      .sort((a, b) => (sort.reverse ? cmp(b, a) : cmp(a, b)));
  }, [experiments, search, sort]);

  const groups = useMemo(
    () => (grouping === "none" ? null : groupExperiments(shown, grouping)),
    [shown, grouping],
  );

  if (experiments.ids.length === 0) {
    return (
      <p className="text-muted mt-3">
        {loaded ? "No experiments yet." : "Loading the list of experiments…"}
      </p>
    );
  }

  function cancelConfirm() {
    setConfirm(undefined);
    info("Action cancelled");
  }

  function runConfirm() {
    if (!confirm) return;
    if (confirm.kind === "kill") client.experiment_kill(confirm.experimentId);
    else client.experiment_delete(confirm.experimentId);
    setConfirm(undefined);
  }

  const renderRow = (exp: Experiment, group?: string) => {
    const id = exp.experiment_id;
    const st = statusInfo(exp);
    const isRunning = exp.status === "running";
    return (
      <tr
        key={group === undefined ? id : `${group}/${id}`}
        data-experiment-id={id}
        data-status={exp.statusCategory}
        style={{ cursor: "pointer" }}
        onClick={() => onOpen(id)}
        title="Open jobs"
      >
        <td>
          <span title={st.label}>{st.icon}</span>
        </td>
        <td>
          <span className="fw-semibold">{exp.experiment_id}</span>
          {exp.current_run_id && (
            <span className="text-muted small ms-2">{exp.current_run_id}</span>
          )}
          <ExperimentTags tags={exp.tags ?? []} />
        </td>
        <td>
          <Badge bg={st.color}>
            {exp.finished_jobs}/{exp.total_jobs}
          </Badge>
          {exp.failed_jobs > 0 && (
            <Badge bg="danger" className="ms-1">
              {exp.failed_jobs} failed
            </Badge>
          )}
        </td>
        <td className="text-muted small">{exp.hostname || "—"}</td>
        <td className="text-muted small">{formatShortDate(exp.started_at)}</td>
        <td className="text-muted small">
          {formatDuration(exp.started_at, exp.ended_at)}
        </td>
        <td className="text-muted small">{formatCO2(exp.carbon?.co2kg)}</td>
        <td
          className="experiment-actions text-nowrap"
          onClick={(e) => e.stopPropagation()}
          title=""
        >
          <i
            className="fa fa-skull-crossbones action"
            title="Kill all running jobs"
            onClick={() => setConfirm({ experimentId: id, kind: "kill" })}
          />
          <i
            className={`fa fa-trash action${isRunning ? " disabled" : ""}`}
            title={
              isRunning
                ? "Cannot delete: experiment is still running"
                : "Delete experiment"
            }
            onClick={() => {
              if (!isRunning) setConfirm({ experimentId: id, kind: "delete" });
            }}
          />
        </td>
      </tr>
    );
  };

  const sortHeader = (column: SortColumn) => (
    <SortHeader
      column={column}
      label={SORT_HEADERS[column].label}
      shortcut={SORT_HEADERS[column].key}
      active={sort.column === column}
      ascending={!sort.reverse}
      onClick={() => sortBy(column)}
    />
  );

  return (
    <div className="mt-3">
      <h5 className="mb-3">Experiments</h5>

      <div className="d-flex align-items-center flex-wrap gap-2 mb-1">
        <Form.Control
          id="searchexperiments"
          ref={searchInput}
          style={{ maxWidth: "24rem" }}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          onKeyDown={clearOnEscape(() => setSearch(""))}
          placeholder="Search experiments or tags (/)"
        />
        <Form.Label htmlFor="sortexperiments" className="mb-0 ms-2">
          Sort
        </Form.Label>
        <Form.Select
          id="sortexperiments"
          size="sm"
          style={{ width: "auto" }}
          value={sort.column}
          onChange={(e) => sortBy(e.target.value as SortColumn)}
        >
          {(["started", "status", "id"] as SortColumn[]).map((column) => (
            <option key={column} value={column}>
              {SORT_HEADERS[column].label} ({SORT_HEADERS[column].key})
            </option>
          ))}
        </Form.Select>
        <Button
          size="sm"
          variant="outline-secondary"
          title="Toggle sort direction"
          onClick={() => changeSort({ ...sort, reverse: !sort.reverse })}
        >
          <i className={`fas fa-arrow-${sort.reverse ? "down" : "up"}`} />
        </Button>
        <Form.Label htmlFor="groupexperiments" className="mb-0 ms-2">
          Group by
        </Form.Label>
        <Form.Select
          id="groupexperiments"
          size="sm"
          style={{ width: "auto" }}
          value={grouping}
          onChange={(e) => changeGrouping(e.target.value as Grouping)}
        >
          {GROUPINGS.map((g) => (
            <option key={g} value={g}>
              {GROUPING_LABELS[g]}
            </option>
          ))}
        </Form.Select>
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
              onClick={() => setCollapsed(new Set(groups.map((g) => g.name)))}
            >
              <i className="fas fa-angles-up" />
            </Button>
          </>
        )}
      </div>
      <KeyHints
        hints={[
          ["/", "search"],
          ["S", "sort by status"],
          ["O", "sort by name"],
          ["D", "sort by date"],
          ["g", "group (none, prefix, tag)"],
        ]}
      />

      {confirm && (
        <Modal show={true} onHide={cancelConfirm}>
          <Modal.Header closeButton>
            <Modal.Title>Are you sure?</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            {confirm.kind === "kill" ? (
              <>
                Kill all running jobs of experiment <b>{confirm.experimentId}</b>?
              </>
            ) : (
              <>
                Delete experiment <b>{confirm.experimentId}</b>?
                <div className="text-danger small mt-2">
                  This removes the experiment runs and events from disk. Job
                  directories are kept (they become orphans).
                </div>
              </>
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

      <Table hover responsive className="align-middle experiments-table">
        <thead>
          <tr>
            {sortHeader("status")}
            {sortHeader("id")}
            <th>Jobs</th>
            <th>Host</th>
            {sortHeader("started")}
            <th>Duration</th>
            <th>CO₂</th>
            <th></th>
          </tr>
        </thead>
        {shown.length === 0 && (
          <tbody>
            <tr>
              <td colSpan={8} className="text-muted">
                No experiment matches “{search.trim()}”.
              </td>
            </tr>
          </tbody>
        )}
        {groups === null ? (
          <tbody>{shown.map((exp) => renderRow(exp))}</tbody>
        ) : (
          groups.map((group) => {
            const isCollapsed = collapsed.has(group.name);
            return (
              <tbody key={`group-${group.name}`}>
                <tr
                  className="experiment-group-row"
                  data-group={group.name}
                  onClick={() => toggleGroup(group.name)}
                  title={isCollapsed ? "Expand" : "Collapse"}
                >
                  <td colSpan={8}>
                    <i
                      className={`fas fa-chevron-${isCollapsed ? "right" : "down"} me-2`}
                    />
                    <span className="fw-semibold">{group.name}</span>
                    <Badge bg="secondary" className="ms-2">
                      {group.experiments.length}
                    </Badge>
                    {statusCounts(group.experiments).map(([status, count]) => (
                      <span key={status} className="ms-2 small" title={status}>
                        {(STATUS_INFO[status] ?? STATUS_INFO.empty).icon} {count}
                      </span>
                    ))}
                  </td>
                </tr>
                {!isCollapsed &&
                  group.experiments.map((exp) => renderRow(exp, group.name))}
              </tbody>
            );
          })
        )}
      </Table>
    </div>
  );
};
