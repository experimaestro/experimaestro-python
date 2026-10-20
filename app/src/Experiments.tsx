import React, { useMemo, useRef, useState } from "react";
import { Table, Badge, Modal, Button, Form } from "react-bootstrap";

import { useAppSelector } from "./store";
import { Experiment } from "./reducers";
import { formatShortDate, formatDuration, formatCO2 } from "./format";
import client from "./client";
import { useMessages } from "./ui/messages";
import { clearOnEscape, useShortcuts } from "./shortcuts";

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
  const searchInput = useRef<HTMLInputElement>(null);
  const { info } = useMessages();

  // The same key again reverses the order; dates start with the newest
  function sortBy(column: SortColumn) {
    const next =
      sort.column === column
        ? { column, reverse: !sort.reverse }
        : { column, reverse: column === "started" };
    setSort(next);
    saveSort(next);
  }

  useShortcuts({
    "/": () => searchInput.current?.focus(),
    S: () => sortBy("status"),
    D: () => sortBy("started"),
    O: () => sortBy("id"),
  });

  const shown = useMemo(() => {
    const text = search.trim().toLowerCase();
    const cmp = compareExperiments(sort.column);
    return experiments.ids
      .map((id) => experiments.byId[id])
      .filter((exp) => exp.experiment_id.toLowerCase().includes(text))
      .sort((a, b) => (sort.reverse ? cmp(b, a) : cmp(a, b)));
  }, [experiments, search, sort]);

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

  const sortHeader = (column: SortColumn) => {
    const { label, key } = SORT_HEADERS[column];
    return (
      <th
        role="button"
        data-sort={column}
        title={`Sort by ${label.toLowerCase()} (${key})`}
        onClick={() => sortBy(column)}
      >
        {label}
        {sort.column === column && (sort.reverse ? " ▼" : " ▲")}
      </th>
    );
  };

  return (
    <div className="mt-3">
      <h5 className="mb-3">Experiments</h5>

      <Form.Control
        id="searchexperiments"
        ref={searchInput}
        className="mb-3"
        style={{ maxWidth: "24rem" }}
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        onKeyDown={clearOnEscape(() => setSearch(""))}
        placeholder="Search experiments (/)"
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
        <tbody>
          {shown.length === 0 && (
            <tr>
              <td colSpan={8} className="text-muted">
                No experiment matches “{search.trim()}”.
              </td>
            </tr>
          )}
          {shown.map((exp) => {
            const id = exp.experiment_id;
            const st = statusInfo(exp);
            const isRunning = exp.status === "running";
            return (
              <tr
                key={id}
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
                    <div className="text-muted small">{exp.current_run_id}</div>
                  )}
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
          })}
        </tbody>
      </Table>
    </div>
  );
};
