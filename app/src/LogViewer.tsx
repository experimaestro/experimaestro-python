import React, {
  useState,
  useEffect,
  useLayoutEffect,
  useRef,
  useCallback,
} from "react";
import { Modal, Tabs, Tab, Button, Form } from "react-bootstrap";

type Stream = "stdout" | "stderr";

export type LogTarget =
  | { kind: "job"; taskId: string; jobId: string; title?: string }
  | { kind: "service"; experimentId: string; serviceId: string; title?: string }
  | { kind: "experimaestro"; title?: string };

function baseUrl(target: LogTarget, stream: Stream): string {
  if (target.kind === "job") {
    return `/api/jobs/${encodeURIComponent(target.taskId)}/${encodeURIComponent(
      target.jobId,
    )}/logs/${stream}`;
  }
  if (target.kind === "experimaestro") {
    return `/api/logs/experimaestro`;
  }
  return `/api/services/${encodeURIComponent(target.experimentId)}/${encodeURIComponent(
    target.serviceId,
  )}/logs/${stream}`;
}

const POLL_MS = 1500;
// Delay before showing that a poll is waiting for its answer
const LOADING_DELAY_MS = 300;
// Distance (pixels) from the top or the bottom counting as being there
const EDGE_PX = 40;

// An answer of the log routes (see read_log_slice)
type LogSlice = {
  content: string;
  partial: string;
  offset: number;
  start: number;
  size: number;
  generation: number;
  // The end of the log was read: replaces what is shown
  tail: boolean;
  exists: boolean;
};

const Loading = ({ children }: { children: React.ReactNode }) => (
  <div className="small text-muted" data-testid="log-loading">
    <span className="spinner-border spinner-border-sm me-1" />
    {children}
  </div>
);

const StreamView = ({
  target,
  stream,
  follow,
}: {
  target: LogTarget;
  stream: Stream;
  follow: boolean;
}) => {
  // Completed lines (appended at each poll, earlier ones prepended) and the
  // lines being written (replaced at each poll: progress bars rewrite them)
  const [content, setContent] = useState("");
  const [partial, setPartial] = useState("");
  // Why the last request failed (null when it succeeded)
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [exists, setExists] = useState(true);
  // A poll has been waiting for its answer for a while
  const [waiting, setWaiting] = useState(false);
  const [loadingEarlier, setLoadingEarlier] = useState(false);
  // Where what is shown starts in the file (0: its start is shown)
  const [start, setStart] = useState(0);

  // Where the next poll reads, the generation of the file read, and where
  // what is shown starts (earlier content is read before it)
  const offsetRef = useRef<number | null>(null);
  const generationRef = useRef<number | null>(null);
  const startRef = useRef(0);
  const followRef = useRef(follow);
  followRef.current = follow;
  const scrollRef = useRef<HTMLPreElement>(null);
  // Keep the view at the bottom (when it was there before an update), or
  // keep what is shown in place when earlier lines are added (the height
  // before adding them)
  const stickRef = useRef(true);
  const prependRef = useRef<number | null>(null);
  const cancelledRef = useRef(false);

  // The log is identified by its URL (the target object is rebuilt at each
  // render of the parent, e.g. at each job update)
  const url = baseUrl(target, stream);

  const request = useCallback(
    async (params: Record<string, number | null>): Promise<LogSlice | null> => {
      const query = Object.entries(params)
        .filter(([, value]) => value !== null)
        .map(([key, value]) => `${key}=${value}`)
        .join("&");
      try {
        const res = await fetch(query ? `${url}?${query}` : url, {
          credentials: "same-origin",
        });
        if (cancelledRef.current) return null;
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          if (!cancelledRef.current)
            setError(`${res.status} ${body.error ?? res.statusText}`);
          return null;
        }
        const data: LogSlice = await res.json();
        if (cancelledRef.current) return null;
        setError(null);
        return data;
      } catch (e) {
        if (!cancelledRef.current) setError(String(e));
        return null;
      }
    },
    [url],
  );

  const atBottom = () => {
    const el = scrollRef.current;
    return !el || el.scrollHeight - el.scrollTop - el.clientHeight < EDGE_PX;
  };

  // Polls one after the other (an answer can take longer than the polling
  // period, e.g. when the log is read from a remote host)
  useEffect(() => {
    cancelledRef.current = false;
    let timer: number | undefined;
    offsetRef.current = null;
    generationRef.current = null;
    startRef.current = 0;
    stickRef.current = true;
    setContent("");
    setPartial("");
    setError(null);
    setLoaded(false);

    const poll = async () => {
      const slow = window.setTimeout(() => setWaiting(true), LOADING_DELAY_MS);
      const data = await request({
        offset: offsetRef.current,
        generation: generationRef.current,
      });
      window.clearTimeout(slow);
      if (cancelledRef.current) return;
      setWaiting(false);
      if (data === null) return;
      stickRef.current = atBottom();
      if (data.tail) {
        setContent(data.content);
        startRef.current = data.start;
        setStart(data.start);
      } else if (data.content) {
        setContent((prev) => prev + data.content);
      }
      setPartial(data.partial);
      offsetRef.current = data.offset;
      generationRef.current = data.generation;
      setExists(data.exists);
      setLoaded(true);
    };

    const loop = async () => {
      await poll();
      if (!cancelledRef.current) {
        timer = window.setTimeout(async function next() {
          if (followRef.current) await poll();
          if (!cancelledRef.current) timer = window.setTimeout(next, POLL_MS);
        }, POLL_MS);
      }
    };
    loop();
    return () => {
      cancelledRef.current = true;
      window.clearTimeout(timer);
    };
  }, [url, request]);

  const loadEarlier = async () => {
    if (loadingEarlier || startRef.current <= 0) return;
    setLoadingEarlier(true);
    const generation = generationRef.current;
    const data = await request({ before: startRef.current, generation });
    if (cancelledRef.current) return;
    setLoadingEarlier(false);
    // Stale when the file was replaced meanwhile (the next poll reads it)
    if (data === null || data.generation !== generation) return;
    prependRef.current = scrollRef.current?.scrollHeight ?? null;
    startRef.current = data.start;
    setStart(data.start);
    setContent((prev) => data.content + prev);
  };

  const onScroll = () => {
    const el = scrollRef.current;
    if (el && el.scrollTop < EDGE_PX && loaded) loadEarlier();
  };

  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (prependRef.current !== null) {
      el.scrollTop += el.scrollHeight - prependRef.current;
      prependRef.current = null;
    } else if (follow && stickRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [content, partial, follow]);

  return (
    <>
      {error && (
        <div className="small text-danger" data-testid="log-error">
          Cannot read the log: {error}
        </div>
      )}
      {loadingEarlier && <Loading>Loading earlier lines…</Loading>}
      <pre
        ref={scrollRef}
        onScroll={onScroll}
        className="log-viewer-content"
        data-start={start}
        style={{
          height: "60vh",
          overflow: "auto",
          background: "#1e1e1e",
          color: "#e0e0e0",
          padding: "0.5rem",
          margin: 0,
          fontSize: "0.8rem",
          whiteSpace: "pre-wrap",
          wordBreak: "break-all",
        }}
      >
        {content + partial ||
          (!loaded ? "" : exists ? "(no output)" : "(no log file yet)")}
      </pre>
      {(waiting || !loaded) && <Loading>Loading…</Loading>}
    </>
  );
};

export const LogPanel = ({ target }: { target: LogTarget }) => {
  // stderr first and shown by default: errors and progress bars go there
  const [stream, setStream] = useState<Stream>("stderr");
  const [follow, setFollow] = useState(true);

  const singleStream = target.kind === "experimaestro";

  return (
    <>
      <div
        style={{ display: "flex", justifyContent: "flex-end", marginBottom: 4 }}
      >
        <Form.Check
          type="switch"
          id="log-follow"
          label="Follow"
          checked={follow}
          onChange={(e) => setFollow(e.target.checked)}
        />
      </div>
      {singleStream ? (
        <StreamView target={target} stream="stdout" follow={follow} />
      ) : (
        // Only the stream shown is polled
        <Tabs
          activeKey={stream}
          onSelect={(k) => setStream((k as Stream) || "stderr")}
          mountOnEnter
          unmountOnExit
        >
          <Tab eventKey="stderr" title="stderr">
            <StreamView target={target} stream="stderr" follow={follow} />
          </Tab>
          <Tab eventKey="stdout" title="stdout">
            <StreamView target={target} stream="stdout" follow={follow} />
          </Tab>
        </Tabs>
      )}
    </>
  );
};

export default ({
  target,
  onHide,
}: {
  target: LogTarget;
  onHide: () => void;
}) => {
  const title =
    target.title ||
    (target.kind === "job"
      ? target.jobId
      : target.kind === "service"
        ? target.serviceId
        : "experimaestro");

  return (
    <Modal show={true} size="xl" onHide={onHide}>
      <Modal.Header closeButton>
        <Modal.Title>Logs — {title}</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <LogPanel target={target} />
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={onHide}>
          Close
        </Button>
      </Modal.Footer>
    </Modal>
  );
};
