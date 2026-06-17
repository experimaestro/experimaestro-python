import { DateTime } from "luxon";

/** Format an ISO timestamp as a short "yyyy-LL-dd HH:mm" string. */
export function formatShortDate(t?: string | null): string {
  if (!t) return "—";
  const d = DateTime.fromISO(t);
  return d.isValid ? d.toFormat("yyyy-LL-dd HH:mm") : "—";
}

/** Format an ISO timestamp with full date/time. */
export function formatFullDate(t?: string | null): string {
  if (!t) return "N/A";
  const d = DateTime.fromISO(t);
  return d.isValid ? d.toLocaleString(DateTime.DATETIME_FULL_WITH_SECONDS) : "N/A";
}

/**
 * Human-readable duration between start and end (or now if still running).
 * Returns "—" when there is no start time.
 */
export function formatDuration(start?: string | null, end?: string | null): string {
  if (!start) return "—";
  const s = DateTime.fromISO(start);
  if (!s.isValid) return "—";
  const e = end ? DateTime.fromISO(end) : DateTime.now();
  const secs = e.diff(s, "seconds").seconds;
  if (isNaN(secs) || secs < 0) return "—";
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const sec = Math.floor(secs % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}

/** Format a CO₂ figure given in kilograms (g below 1 kg). */
export function formatCO2(kg?: number | null): string {
  if (kg == null || kg <= 0) return "—";
  if (kg < 1) return `${(kg * 1000).toFixed(1)} g`;
  return `${kg.toFixed(2)} kg`;
}
