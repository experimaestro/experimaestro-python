import React, { useState, useEffect, useRef, useCallback } from "react";
import { Modal, Tabs, Tab, Button, Form } from "react-bootstrap";

type Stream = "stdout" | "stderr";

export type LogTarget =
  | { kind: "job"; taskId: string; jobId: string; title?: string }
  | { kind: "service"; serviceId: string; title?: string }
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
  return `/api/services/${encodeURIComponent(target.serviceId)}/logs/${stream}`;
}

const POLL_MS = 1500;

const StreamView = ({
  target,
  stream,
  follow,
}: {
  target: LogTarget;
  stream: Stream;
  follow: boolean;
}) => {
  const [content, setContent] = useState("");
  const offsetRef = useRef<number | null>(null);
  const scrollRef = useRef<HTMLPreElement>(null);

  const poll = useCallback(async () => {
    try {
      const offset = offsetRef.current;
      const url =
        offset === null ? baseUrl(target, stream) : `${baseUrl(target, stream)}?offset=${offset}`;
      const res = await fetch(url, { credentials: "same-origin" });
      if (!res.ok) return;
      const data = await res.json();
      // Truncation/rotation: server returns offset < previous → reset
      if (offsetRef.current !== null && data.offset < offsetRef.current) {
        setContent(data.content || "");
      } else if (data.content) {
        setContent((prev) => prev + data.content);
      }
      offsetRef.current = data.offset ?? offsetRef.current;
    } catch (e) {
      // ignore transient fetch errors
    }
  }, [target, stream]);

  // Reset when the target/stream changes
  useEffect(() => {
    offsetRef.current = null;
    setContent("");
    poll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [poll]);

  useEffect(() => {
    if (!follow) return;
    const t = window.setInterval(poll, POLL_MS);
    return () => window.clearInterval(t);
  }, [follow, poll]);

  useEffect(() => {
    if (follow && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [content, follow]);

  return (
    <pre
      ref={scrollRef}
      className="log-viewer-content"
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
      {content || "(no output)"}
    </pre>
  );
};

export const LogPanel = ({ target }: { target: LogTarget }) => {
  const [stream, setStream] = useState<Stream>("stdout");
  const [follow, setFollow] = useState(true);

  const singleStream = target.kind === "experimaestro";

  return (
    <>
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 4 }}>
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
        <Tabs activeKey={stream} onSelect={(k) => setStream((k as Stream) || "stdout")}>
          <Tab eventKey="stdout" title="stdout">
            <StreamView target={target} stream="stdout" follow={follow} />
          </Tab>
          <Tab eventKey="stderr" title="stderr">
            <StreamView target={target} stream="stderr" follow={follow} />
          </Tab>
        </Tabs>
      )}
    </>
  );
};

export default ({ target, onHide }: { target: LogTarget; onHide: () => void }) => {
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
