// Interactive Parameters & Recursive Sub-Config Inspector for DagView
import React, {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  GraphNode,
  SerializedEnumRef,
  SerializedFieldRef,
  SerializedObject,
  SerializedPathRef,
} from "./types";

interface ParamsInspectorProps {
  node: GraphNode;
  objects?: { [id: string]: SerializedObject };
  status?: string;
  onOpenJob?: (node: GraphNode) => void;
}

type ViewMode = "tree" | "code" | "json";

const isFieldRef = (v: unknown): v is SerializedFieldRef =>
  typeof v === "object" && v !== null && (v as any).type === "python" && "value" in (v as any);

const isPathRef = (v: unknown): v is SerializedPathRef =>
  typeof v === "object" &&
  v !== null &&
  ((v as any).type === "path" || (v as any).type === "path.serialized");

const isEnumRef = (v: unknown): v is SerializedEnumRef =>
  typeof v === "object" && v !== null && (v as any).type === "enum";

function copyText(text: string) {
  if (navigator.clipboard) {
    navigator.clipboard.writeText(text);
  } else {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
  }
}

/** Search context to track query, active match index, and auto-registration of matches */
interface SearchContextType {
  filter: string;
  activeMatchIndex: number;
  registerMatch: () => { index: number; isActive: boolean };
  activeRef: React.MutableRefObject<HTMLElement | null>;
}

const SearchContext = createContext<SearchContextType>({
  filter: "",
  activeMatchIndex: -1,
  registerMatch: () => ({ index: -1, isActive: false }),
  activeRef: { current: null },
});

export const HighlightedText = ({
  text,
  className = "",
}: {
  text: string;
  className?: string;
}) => {
  const { filter, registerMatch, activeRef } = useContext(SearchContext);

  if (!filter.trim() || !text) {
    return <span className={className}>{text}</span>;
  }

  const escaped = filter.trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const regex = new RegExp(`(${escaped})`, "gi");
  const parts = text.split(regex);

  if (parts.length === 1) {
    return <span className={className}>{text}</span>;
  }

  return (
    <span className={className}>
      {parts.map((part, i) => {
        if (regex.test(part)) {
          const { index, isActive } = registerMatch();
          return (
            <mark
              key={i}
              ref={isActive ? (el) => { if (el) activeRef.current = el; } : undefined}
              data-match-index={index}
              className={`xpm-dag-highlight ${isActive ? "active" : ""}`}
            >
              {part}
            </mark>
          );
        }
        return part;
      })}
    </span>
  );
};

function countMatchesInString(text: string, query: string): number {
  if (!query.trim() || !text) return 0;
  const escaped = query.trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const matches = text.match(new RegExp(escaped, "gi"));
  return matches ? matches.length : 0;
}

function countMatchesInValue(
  name: string,
  val: unknown,
  query: string,
  objects: { [id: string]: SerializedObject },
  ancestors: Set<number>,
): number {
  let count = countMatchesInString(name, query);

  if (isFieldRef(val)) {
    const subId = Number(val.value);
    count += countMatchesInObject(subId, query, objects, ancestors);
  } else if (isPathRef(val)) {
    count += countMatchesInString(val.value, query);
    if (val.base) count += countMatchesInString(val.base, query);
  } else if (isEnumRef(val)) {
    count += countMatchesInString(`${val.enum}.${val.value}`, query);
  } else if (Array.isArray(val)) {
    val.forEach((item, idx) => {
      count += countMatchesInValue(`[${idx}]`, item, query, objects, ancestors);
    });
  } else if (typeof val === "object" && val !== null) {
    Object.entries(val as Record<string, unknown>).forEach(([k, item]) => {
      count += countMatchesInValue(k, item, query, objects, ancestors);
    });
  } else {
    count += countMatchesInString(formatScalar(val), query);
  }

  return count;
}

function countMatchesInObject(
  oid: number,
  query: string,
  objects: { [id: string]: SerializedObject },
  ancestors = new Set<number>(),
): number {
  if (ancestors.has(oid)) return 0;
  const obj = objects[String(oid)];
  if (!obj) return 0;

  const nextAncestors = new Set(ancestors);
  nextAncestors.add(oid);

  const typeName = obj.type || obj.typename?.split(".").pop() || `Config_${oid}`;
  const shortId = obj.identifier ? obj.identifier.slice(0, 8) : `#${oid}`;

  let count = countMatchesInString(typeName, query) + countMatchesInString(shortId, query);

  const fields = obj.fields || {};
  for (const [k, v] of Object.entries(fields)) {
    count += countMatchesInValue(k, v, query, objects, nextAncestors);
  }

  const initTasks = obj["init-tasks"] || [];
  initTasks.forEach((itId, idx) => {
    count += countMatchesInString(`init[${idx}]`, query);
    count += countMatchesInObject(Number(itId), query, objects, nextAncestors);
  });

  return count;
}

/** Synthesize Python .C(...) configuration snippet from serialized object graph */
function toPythonSnippet(
  oid: number,
  objects: { [id: string]: SerializedObject },
  indent = 0,
  ancestors: Set<number> = new Set(),
): string {
  const obj = objects[String(oid)];
  if (!obj) return `#${oid}`;

  const typeName = obj.type || obj.typename?.split(".").pop() || `Config_${oid}`;
  if (ancestors.has(oid)) {
    return `${typeName}.C(...)  # cycle to #${oid}`;
  }

  const nextAncestors = new Set(ancestors);
  nextAncestors.add(oid);

  const sp = "  ".repeat(indent);
  const spi = "  ".repeat(indent + 1);

  const fields = obj.fields || {};
  const sortedKeys = Object.keys(fields).sort();

  if (sortedKeys.length === 0) {
    return `${typeName}.C()`;
  }

  const args: string[] = [];
  for (const k of sortedKeys) {
    const v = fields[k];
    args.push(`${spi}${k}=${formatPythonValue(v, objects, indent + 1, nextAncestors)},`);
  }

  return `${typeName}.C(\n${args.join("\n")}\n${sp})`;
}

function formatPythonValue(
  v: unknown,
  objects: { [id: string]: SerializedObject },
  indent: number,
  ancestors: Set<number>,
): string {
  if (v === null) return "None";
  if (v === undefined) return "None";
  if (typeof v === "boolean") return v ? "True" : "False";
  if (typeof v === "number") return String(v);
  if (typeof v === "string") return JSON.stringify(v);

  if (isFieldRef(v)) {
    return toPythonSnippet(Number(v.value), objects, indent, ancestors);
  }
  if (isPathRef(v)) {
    const baseInfo = v.base ? `  # base: ${v.base}` : "";
    return `Path(${JSON.stringify(v.value)})${baseInfo}`;
  }
  if (isEnumRef(v)) {
    return `${v.enum}.${v.value}`;
  }
  if (Array.isArray(v)) {
    if (v.length === 0) return "[]";
    const items = v.map((item) => formatPythonValue(item, objects, indent + 1, ancestors));
    return `[${items.join(", ")}]`;
  }
  if (typeof v === "object") {
    const entries = Object.entries(v as Record<string, unknown>).map(
      ([k, val]) => `${JSON.stringify(k)}: ${formatPythonValue(val, objects, indent, ancestors)}`,
    );
    return `{${entries.join(", ")}}`;
  }
  return String(v);
}

/** Fully resolve nested objects into clean JSON */
function resolveToJson(
  oid: number,
  objects: { [id: string]: SerializedObject },
  ancestors: Set<number> = new Set(),
): unknown {
  const obj = objects[String(oid)];
  if (!obj) return `#${oid}`;

  if (ancestors.has(oid)) {
    return `[CYCLE #${oid}]`;
  }
  const nextAncestors = new Set(ancestors);
  nextAncestors.add(oid);

  const res: Record<string, unknown> = {
    _type: obj.type || obj.typename,
    _id: obj.id,
  };
  if (obj.identifier) res._identifier = obj.identifier;

  const fields = obj.fields || {};
  for (const [k, v] of Object.entries(fields)) {
    res[k] = resolveValueToJson(v, objects, nextAncestors);
  }
  return res;
}

function resolveValueToJson(
  v: unknown,
  objects: { [id: string]: SerializedObject },
  ancestors: Set<number>,
): unknown {
  if (isFieldRef(v)) {
    return resolveToJson(Number(v.value), objects, ancestors);
  }
  if (isPathRef(v)) {
    return { type: "path", value: v.value, base: v.base };
  }
  if (isEnumRef(v)) {
    return `${v.enum}.${v.value}`;
  }
  if (Array.isArray(v)) {
    return v.map((x) => resolveValueToJson(x, objects, ancestors));
  }
  if (typeof v === "object" && v !== null) {
    const res: Record<string, unknown> = {};
    for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
      res[k] = resolveValueToJson(val, objects, ancestors);
    }
    return res;
  }
  return v;
}

export const ParamsInspector = ({
  node,
  objects = {},
  status,
  onOpenJob,
}: ParamsInspectorProps) => {
  const [mode, setMode] = useState<ViewMode>("tree");
  const [filter, setFilter] = useState("");
  const [activeMatchIndex, setActiveMatchIndex] = useState(0);
  const [copied, setCopied] = useState<string | null>(null);
  const [groupFields, setGroupFields] = useState(true);
  const activeRef = useRef<HTMLElement | null>(null);

  // Find root object for this node
  const rootObj = useMemo(() => {
    if (node.objectId !== undefined && objects[String(node.objectId)]) {
      return objects[String(node.objectId)];
    }
    // Fallback: match by identifier hash
    for (const o of Object.values(objects)) {
      if (o.identifier && String(o.identifier) === node.id) {
        return o;
      }
    }
    return null;
  }, [node, objects]);

  const handleCopy = (text: string, label: string) => {
    copyText(text);
    setCopied(label);
    setTimeout(() => setCopied(null), 1500);
  };

  // Pre-calculate shared counts
  const sharedCounts = useMemo(() => {
    const counts = new Map<number, number>();
    const countRefs = (val: unknown) => {
      if (!val || typeof val !== "object") return;
      if (isFieldRef(val)) {
        const id = Number(val.value);
        counts.set(id, (counts.get(id) || 0) + 1);
        return;
      }
      if (Array.isArray(val)) {
        val.forEach(countRefs);
        return;
      }
      Object.values(val as Record<string, unknown>).forEach(countRefs);
    };

    for (const o of Object.values(objects)) {
      if (o.fields) {
        Object.values(o.fields).forEach(countRefs);
      }
      if (o["init-tasks"]) {
        o["init-tasks"].forEach((id) => counts.set(id, (counts.get(id) || 0) + 1));
      }
    }
    return counts;
  }, [objects]);

  // Compute total matches in tree
  const totalMatches = useMemo(() => {
    if (!filter.trim() || !rootObj) return 0;
    return countMatchesInObject(rootObj.id, filter, objects);
  }, [filter, rootObj, objects]);

  // Reset or clamp activeMatchIndex when totalMatches changes
  useEffect(() => {
    if (totalMatches === 0) {
      setActiveMatchIndex(0);
    } else if (activeMatchIndex >= totalMatches) {
      setActiveMatchIndex(0);
    }
  }, [totalMatches]);

  // Smooth scroll active match into view
  useEffect(() => {
    if (filter && activeRef.current) {
      activeRef.current.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    }
  }, [activeMatchIndex, filter]);

  const goToNextMatch = () => {
    if (totalMatches <= 0) return;
    setActiveMatchIndex((prev) => (prev + 1) % totalMatches);
  };

  const goToPrevMatch = () => {
    if (totalMatches <= 0) return;
    setActiveMatchIndex((prev) => (prev - 1 + totalMatches) % totalMatches);
  };

  if (!rootObj) {
    // Graceful fallback to scalar tags when object definitions are not embedded
    return (
      <div className="xpm-dag-inspector-fallback">
        <div className="xpm-dag-inspector-note">
          Detailed configuration parameters not available for this node.
        </div>
        <div className="xpm-dag-tags">
          {Object.entries(node.tags).map(([k, v]) => (
            <span key={k} className="xpm-dag-tag">
              <span className="xpm-dag-tag-name">{k}</span>
              <span className="xpm-dag-tag-value">{String(v)}</span>
            </span>
          ))}
        </div>
      </div>
    );
  }

  // Mutable counter for assigning match indices during render
  let matchCounter = 0;
  const registerMatch = () => {
    const index = matchCounter++;
    return {
      index,
      isActive: index === activeMatchIndex,
    };
  };

  const searchContextValue: SearchContextType = {
    filter,
    activeMatchIndex,
    registerMatch,
    activeRef,
  };

  return (
    <SearchContext.Provider value={searchContextValue}>
      <div className="xpm-dag-inspector">
        <div className="xpm-dag-inspector-toolbar">
          <div className="xpm-dag-inspector-modes">
            <button
              className={`xpm-dag-mode-btn ${mode === "tree" ? "active" : ""}`}
              onClick={() => setMode("tree")}
              title="Interactive Tree Explorer"
            >
              Tree
            </button>
            <button
              className={`xpm-dag-mode-btn ${mode === "code" ? "active" : ""}`}
              onClick={() => setMode("code")}
              title="Python .C(...) Snippet"
            >
              Python
            </button>
            <button
              className={`xpm-dag-mode-btn ${mode === "json" ? "active" : ""}`}
              onClick={() => setMode("json")}
              title="JSON view"
            >
              JSON
            </button>
          </div>

          {mode === "tree" && (
            <div className="xpm-dag-inspector-tools">
              <button
                className="xpm-dag-tool-btn"
                onClick={() => setGroupFields(!groupFields)}
                title={groupFields ? "Show alphabetical flat list" : "Show categorized groups"}
              >
                {groupFields ? "Flat" : "Categorized"}
              </button>
            </div>
          )}
        </div>

        {mode === "tree" && (
          <div className="xpm-dag-search-bar">
            <div className="xpm-dag-search-input-wrap">
              <input
                type="text"
                className="xpm-dag-search-input"
                placeholder="Filter parameters (e.g. lr, model, epochs)..."
                value={filter}
                onChange={(e) => {
                  setFilter(e.target.value);
                  setActiveMatchIndex(0);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    if (e.shiftKey) {
                      goToPrevMatch();
                    } else {
                      goToNextMatch();
                    }
                  }
                }}
              />
              {filter && (
                <button
                  className="xpm-dag-search-clear"
                  onClick={() => {
                    setFilter("");
                    setActiveMatchIndex(0);
                  }}
                  title="Clear search"
                >
                  ×
                </button>
              )}
            </div>

            {filter.trim() && (
              <>
                <span
                  className={`xpm-dag-match-badge ${
                    totalMatches === 0 ? "no-match" : ""
                  }`}
                  title={`${totalMatches} matches found across parameters`}
                >
                  {totalMatches === 0
                    ? "0 matches"
                    : `${activeMatchIndex + 1} of ${totalMatches}`}
                </span>

                <div className="xpm-dag-search-nav">
                  <button
                    className="xpm-dag-search-nav-btn"
                    disabled={totalMatches <= 1}
                    onClick={goToPrevMatch}
                    title="Previous match (Shift+Enter)"
                  >
                    ▲
                  </button>
                  <button
                    className="xpm-dag-search-nav-btn"
                    disabled={totalMatches <= 1}
                    onClick={goToNextMatch}
                    title="Next match (Enter)"
                  >
                    ▼
                  </button>
                </div>
              </>
            )}
          </div>
        )}

        {mode === "tree" && (
          <div className="xpm-dag-tree-scroll">
            {filter.trim() && totalMatches === 0 ? (
              <div className="xpm-dag-no-matches">
                No parameters or sub-configs match "{filter}"
              </div>
            ) : (
              <ObjectTreeNode
                oid={rootObj.id}
                objects={objects}
                ancestors={[]}
                isRoot={true}
                filter={filter}
                groupFields={groupFields}
                sharedCounts={sharedCounts}
                onOpenJob={onOpenJob}
              />
            )}
          </div>
        )}

        {mode === "code" && (
          <div className="xpm-dag-code-pane">
            <div className="xpm-dag-code-actions">
              <button
                className="xpm-dag-copy-btn"
                onClick={() =>
                  handleCopy(toPythonSnippet(rootObj.id, objects), "python")
                }
              >
                {copied === "python" ? "✓ Copied" : "Copy Python"}
              </button>
            </div>
            <pre className="xpm-dag-code-block">
              {toPythonSnippet(rootObj.id, objects)}
            </pre>
          </div>
        )}

        {mode === "json" && (
          <div className="xpm-dag-code-pane">
            <div className="xpm-dag-code-actions">
              <button
                className="xpm-dag-copy-btn"
                onClick={() =>
                  handleCopy(
                    JSON.stringify(resolveToJson(rootObj.id, objects), null, 2),
                    "json",
                  )
                }
              >
                {copied === "json" ? "✓ Copied" : "Copy JSON"}
              </button>
            </div>
            <pre className="xpm-dag-code-block">
              {JSON.stringify(resolveToJson(rootObj.id, objects), null, 2)}
            </pre>
          </div>
        )}
      </div>
    </SearchContext.Provider>
  );
};

// Tree node rendering an Experimaestro object and its categorized fields
interface ObjectTreeNodeProps {
  oid: number;
  objects: { [id: string]: SerializedObject };
  ancestors: number[];
  label?: string;
  isRoot?: boolean;
  filter: string;
  groupFields: boolean;
  sharedCounts: Map<number, number>;
  onOpenJob?: (node: GraphNode) => void;
}

const ObjectTreeNode = ({
  oid,
  objects,
  ancestors,
  label,
  isRoot = false,
  filter,
  groupFields,
  sharedCounts,
  onOpenJob,
}: ObjectTreeNodeProps) => {
  const [userExpanded, setUserExpanded] = useState<boolean>(true);
  const obj = objects[String(oid)];

  if (!obj) {
    return (
      <div className="xpm-dag-tree-row">
        {label && <span className="xpm-dag-field-name">{label}: </span>}
        <span className="xpm-dag-val-missing">#{oid} (missing)</span>
      </div>
    );
  }

  const typeName = obj.type || obj.typename?.split(".").pop() || `Config_${oid}`;
  const shortId = obj.identifier ? obj.identifier.slice(0, 8) : `#${oid}`;

  // Cycle check
  if (ancestors.includes(oid)) {
    return (
      <div className="xpm-dag-tree-row xpm-dag-cycle-row">
        {label && (
          <span className="xpm-dag-field-name">
            <HighlightedText text={label} />:{" "}
          </span>
        )}
        <span className="xpm-dag-badge-cycle" title="Circular reference cycle">
          🔁 [Cycle to {typeName} #{oid}]
        </span>
      </div>
    );
  }

  const sharedCount = sharedCounts.get(oid) || 0;
  const isShared = sharedCount > 1;

  const fields = obj.fields || {};
  const nextAncestors = [...ancestors, oid];

  // Match detection: check if self or children match query
  const selfMatches = filter.trim()
    ? countMatchesInString(label || "", filter) +
      countMatchesInString(typeName, filter) +
      countMatchesInString(shortId, filter)
    : 0;

  const totalMatchesInSubtree = filter.trim()
    ? countMatchesInObject(oid, filter, objects, new Set(ancestors))
    : 0;

  // If search filter is active and this node + subtree has 0 matches, hide this entire branch!
  if (filter.trim() && totalMatchesInSubtree === 0 && selfMatches === 0 && !isRoot) {
    return null;
  }

  // When search filter is active and matches exist in subtree or self, GUARANTEE unfolding!
  const expanded = filter.trim() ? true : userExpanded;

  // Partition fields into categories matching experimaestro inspect
  const hparams: Array<[string, unknown]> = [];
  const paths: Array<[string, SerializedPathRef]> = [];
  const subConfigs: Array<[string, SerializedFieldRef]> = [];
  const collections: Array<[string, unknown]> = [];

  for (const [k, v] of Object.entries(fields)) {
    if (isFieldRef(v)) {
      subConfigs.push([k, v]);
    } else if (isPathRef(v)) {
      paths.push([k, v]);
    } else if (Array.isArray(v) || (typeof v === "object" && v !== null && !isEnumRef(v))) {
      collections.push([k, v]);
    } else {
      hparams.push([k, v]);
    }
  }

  // Sort alphabetically within categories
  hparams.sort(([a], [b]) => a.localeCompare(b));
  paths.sort(([a], [b]) => a.localeCompare(b));
  subConfigs.sort(([a], [b]) => a.localeCompare(b));
  collections.sort(([a], [b]) => a.localeCompare(b));

  const initTasks = obj["init-tasks"] || [];

  return (
    <div className={`xpm-dag-tree-branch ${isRoot ? "root-branch" : ""}`}>
      {!isRoot && (
        <div
          className="xpm-dag-tree-header"
          onClick={() => setUserExpanded(!userExpanded)}
          title="Click to toggle expansion"
        >
          <span className="xpm-dag-tree-toggle">{expanded ? "▼" : "▶"}</span>
          {label && (
            <span className="xpm-dag-field-name">
              <HighlightedText text={label} />:{" "}
            </span>
          )}
          <HighlightedText text={typeName} className="xpm-dag-type-badge" />
          <span className="xpm-dag-tree-hash" title={obj.identifier || String(oid)}>
            [<HighlightedText text={shortId} />]
          </span>
          {isShared && (
            <span
              className="xpm-dag-badge-shared"
              title={`Referenced ${sharedCount} times in graph`}
            >
              ↪ shared ×{sharedCount}
            </span>
          )}
        </div>
      )}

      {expanded && (
        <div className={`xpm-dag-tree-content ${!isRoot ? "indented" : ""}`}>
          {groupFields ? (
            <>
              {hparams.length > 0 && (
                <div className="xpm-dag-category">
                  <div className="xpm-dag-cat-title">⚙️ Hyperparameters</div>
                  {hparams.map(([k, v]) => (
                    <FieldValueRow
                      key={k}
                      name={k}
                      value={v}
                      filter={filter}
                    />
                  ))}
                </div>
              )}

              {paths.length > 0 && (
                <div className="xpm-dag-category">
                  <div className="xpm-dag-cat-title">📁 Paths & Outputs</div>
                  {paths.map(([k, v]) => (
                    <PathValueRow
                      key={k}
                      name={k}
                      spec={v}
                      filter={filter}
                    />
                  ))}
                </div>
              )}

              {subConfigs.length > 0 && (
                <div className="xpm-dag-category">
                  <div className="xpm-dag-cat-title">🧩 Sub-Configurations</div>
                  {subConfigs.map(([k, v]) => (
                    <ObjectTreeNode
                      key={k}
                      oid={Number(v.value)}
                      objects={objects}
                      ancestors={nextAncestors}
                      label={k}
                      filter={filter}
                      groupFields={groupFields}
                      sharedCounts={sharedCounts}
                      onOpenJob={onOpenJob}
                    />
                  ))}
                </div>
              )}

              {collections.length > 0 && (
                <div className="xpm-dag-category">
                  <div className="xpm-dag-cat-title">📦 Collections</div>
                  {collections.map(([k, v]) => (
                    <CollectionRow
                      key={k}
                      name={k}
                      value={v}
                      objects={objects}
                      ancestors={nextAncestors}
                      filter={filter}
                      groupFields={groupFields}
                      sharedCounts={sharedCounts}
                      onOpenJob={onOpenJob}
                    />
                  ))}
                </div>
              )}

              {initTasks.length > 0 && (
                <div className="xpm-dag-category">
                  <div className="xpm-dag-cat-title">⚡ Init Tasks</div>
                  {initTasks.map((itOid, idx) => (
                    <ObjectTreeNode
                      key={itOid}
                      oid={Number(itOid)}
                      objects={objects}
                      ancestors={nextAncestors}
                      label={`init[${idx}]`}
                      filter={filter}
                      groupFields={groupFields}
                      sharedCounts={sharedCounts}
                      onOpenJob={onOpenJob}
                    />
                  ))}
                </div>
              )}
            </>
          ) : (
            // Flat display
            <>
              {Object.entries(fields)
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([k, v]) => {
                  if (isFieldRef(v)) {
                    return (
                      <ObjectTreeNode
                        key={k}
                        oid={Number(v.value)}
                        objects={objects}
                        ancestors={nextAncestors}
                        label={k}
                        filter={filter}
                        groupFields={groupFields}
                        sharedCounts={sharedCounts}
                        onOpenJob={onOpenJob}
                      />
                    );
                  }
                  if (isPathRef(v)) {
                    return (
                      <PathValueRow
                        key={k}
                        name={k}
                        spec={v}
                        filter={filter}
                      />
                    );
                  }
                  if (Array.isArray(v) || (typeof v === "object" && v !== null && !isEnumRef(v))) {
                    return (
                      <CollectionRow
                        key={k}
                        name={k}
                        value={v}
                        objects={objects}
                        ancestors={nextAncestors}
                        filter={filter}
                        groupFields={groupFields}
                        sharedCounts={sharedCounts}
                        onOpenJob={onOpenJob}
                      />
                    );
                  }
                  return (
                    <FieldValueRow
                      key={k}
                      name={k}
                      value={v}
                      filter={filter}
                    />
                  );
                })}
            </>
          )}
        </div>
      )}
    </div>
  );
};

// Row displaying an individual scalar parameter
const FieldValueRow = ({
  name,
  value,
  filter,
}: {
  name: string;
  value: unknown;
  filter: string;
}) => {
  const valStr = formatScalar(value);
  const match = filter.trim()
    ? name.toLowerCase().includes(filter.toLowerCase()) ||
      valStr.toLowerCase().includes(filter.toLowerCase())
    : false;

  if (filter.trim() && !match) {
    return null;
  }

  return (
    <div className={`xpm-dag-tree-row ${match ? "matched" : ""}`}>
      <span className="xpm-dag-field-name">
        <HighlightedText text={name} />:{" "}
      </span>
      <span className={`xpm-dag-field-val ${scalarTypeClass(value)}`}>
        <HighlightedText text={valStr} />
      </span>
    </div>
  );
};

// Row displaying a path parameter with copy action
const PathValueRow = ({
  name,
  spec,
  filter,
}: {
  name: string;
  spec: SerializedPathRef;
  filter: string;
}) => {
  const [copied, setCopied] = useState(false);
  const pathStr = spec.value;
  const match = filter.trim()
    ? name.toLowerCase().includes(filter.toLowerCase()) ||
      pathStr.toLowerCase().includes(filter.toLowerCase()) ||
      (spec.base?.toLowerCase().includes(filter.toLowerCase()) ?? false)
    : false;

  if (filter.trim() && !match) {
    return null;
  }

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    copyText(pathStr);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <div className={`xpm-dag-tree-row xpm-dag-path-row ${match ? "matched" : ""}`}>
      <span className="xpm-dag-field-name">
        <HighlightedText text={name} />:{" "}
      </span>
      <span className="xpm-dag-path-val" title={pathStr}>
        <HighlightedText text={pathStr} />
      </span>
      {spec.base && (
        <span className="xpm-dag-path-base">
          [<HighlightedText text={spec.base} />]
        </span>
      )}
      <button
        className="xpm-dag-path-copy"
        onClick={handleCopy}
        title="Copy path to clipboard"
      >
        {copied ? "✓" : "📋"}
      </button>
    </div>
  );
};

// Row displaying a collection (array or dictionary)
const CollectionRow = ({
  name,
  value,
  objects,
  ancestors,
  filter,
  groupFields,
  sharedCounts,
  onOpenJob,
}: {
  name: string;
  value: unknown;
  objects: { [id: string]: SerializedObject };
  ancestors: number[];
  filter: string;
  groupFields: boolean;
  sharedCounts: Map<number, number>;
  onOpenJob?: (node: GraphNode) => void;
}) => {
  const [userExpanded, setUserExpanded] = useState<boolean>(true);
  const isArr = Array.isArray(value);
  const items = isArr
    ? (value as unknown[]).map((v, i) => [`[${i}]`, v] as [string, unknown])
    : Object.entries(value as Record<string, unknown>);

  const count = items.length;

  // Check if collection itself or any items match
  const selfMatches = filter.trim() ? countMatchesInString(name, filter) : 0;
  const totalItemMatches = filter.trim()
    ? items.reduce(
        (sum, [subName, subVal]) =>
          sum +
          countMatchesInValue(
            subName,
            subVal,
            filter,
            objects,
            new Set(ancestors),
          ),
        0,
      )
    : 0;

  if (filter.trim() && selfMatches === 0 && totalItemMatches === 0) {
    return null;
  }

  // When searching, guarantee unfolding!
  const expanded = filter.trim() ? true : userExpanded;

  return (
    <div className="xpm-dag-collection">
      <div
        className="xpm-dag-tree-header"
        onClick={() => setUserExpanded(!userExpanded)}
        title="Click to toggle collection"
      >
        <span className="xpm-dag-tree-toggle">{expanded ? "▼" : "▶"}</span>
        <span className="xpm-dag-field-name">
          <HighlightedText text={name} />:{" "}
        </span>
        <span className="xpm-dag-coll-type">
          {isArr ? `Array(${count})` : `Dict(${count})`}
        </span>
      </div>

      {expanded && (
        <div className="xpm-dag-tree-content indented">
          {items.map(([subName, subVal]) => {
            if (isFieldRef(subVal)) {
              return (
                <ObjectTreeNode
                  key={subName}
                  oid={Number(subVal.value)}
                  objects={objects}
                  ancestors={ancestors}
                  label={subName}
                  filter={filter}
                  groupFields={groupFields}
                  sharedCounts={sharedCounts}
                  onOpenJob={onOpenJob}
                />
              );
            }
            if (isPathRef(subVal)) {
              return (
                <PathValueRow
                  key={subName}
                  name={subName}
                  spec={subVal}
                  filter={filter}
                />
              );
            }
            if (
              Array.isArray(subVal) ||
              (typeof subVal === "object" && subVal !== null && !isEnumRef(subVal))
            ) {
              return (
                <CollectionRow
                  key={subName}
                  name={subName}
                  value={subVal}
                  objects={objects}
                  ancestors={ancestors}
                  filter={filter}
                  groupFields={groupFields}
                  sharedCounts={sharedCounts}
                  onOpenJob={onOpenJob}
                />
              );
            }
            return (
              <FieldValueRow
                key={subName}
                name={subName}
                value={subVal}
                filter={filter}
              />
            );
          })}
        </div>
      )}
    </div>
  );
};

function formatScalar(v: unknown): string {
  if (v === null) return "null";
  if (v === undefined) return "undefined";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "string") return `"${v}"`;
  if (typeof v === "number") return String(v);
  if (isEnumRef(v)) return `${v.enum}.${v.value}`;
  return JSON.stringify(v);
}

function scalarTypeClass(v: unknown): string {
  if (v === null || v === undefined) return "val-null";
  if (typeof v === "boolean") return "val-bool";
  if (typeof v === "string") return "val-str";
  if (typeof v === "number") return "val-num";
  if (isEnumRef(v)) return "val-enum";
  return "val-other";
}
