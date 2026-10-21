import React from "react";

/** Message shown when the order changes, as in the TUI ("Sorted by status (desc)") */
export function sortedMessage(label: string, ascending: boolean): string {
  return `Sorted by ${label.toLowerCase()} (${ascending ? "asc" : "desc"})`;
}

type SortProps = {
  label: string;
  shortcut: string;
  active: boolean;
  ascending: boolean;
  onClick: () => void;
  column: string;
};

/** Label sorting the table when clicked (or with its key) */
export const SortLabel = ({
  label,
  shortcut,
  active,
  ascending,
  onClick,
  column,
}: SortProps) => (
  <span
    role="button"
    className="sort-label"
    data-sort={column}
    title={`Sort by ${label.toLowerCase()} (${shortcut})`}
    onClick={onClick}
  >
    {label}
    {active && (ascending ? " ▲" : " ▼")}
  </span>
);

/** Column header sorting the table when clicked (or with its key) */
export const SortHeader = ({
  label,
  shortcut,
  active,
  ascending,
  onClick,
  column,
}: SortProps) => (
  <th
    role="button"
    data-sort={column}
    title={`Sort by ${label.toLowerCase()} (${shortcut})`}
    onClick={onClick}
  >
    {label}
    {active && (ascending ? " ▲" : " ▼")}
  </th>
);

/** The keys of a view, as in the TUI footer */
export const KeyHints = ({ hints }: { hints: Array<[string, string]> }) => (
  <div className="key-hints small text-muted">
    {hints.map(([key, label]) => (
      <span key={key} className="me-3">
        <kbd>{key}</kbd> {label}
      </span>
    ))}
  </div>
);
