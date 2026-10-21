import React from "react";

const DOCS_URL = "https://experimaestro-python.readthedocs.io/en/stable/";

/** Help icon linking to a page of the documentation (relative to DOCS_URL) */
export function DocLink({ page }: { page: string }) {
  return (
    <a
      href={DOCS_URL + page}
      target="_blank"
      rel="noopener noreferrer"
      className="text-muted ms-2"
      title="Documentation"
    >
      <i className="fas fa-circle-question" />
    </a>
  );
}
