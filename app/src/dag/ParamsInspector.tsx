// Interactive Parameters & Recursive Sub-Config Inspector for DagView
import React, { useMemo, useState } from "react";
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
  const [copied, setCopied] = useState<string | null>(null);
  const [groupFields, setGroupFields] = useState(true);

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

  const matchesFilter = (text: string) => {
    if (!filter) return true;
    return text.toLowerCase().includes(filter.toLowerCase());
  };

  return (
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
          <input
            type="text"
            className="xpm-dag-search-input"
            placeholder="Filter parameters (e.g. lr, model, epochs)..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
          {filter && (
            <button
              className="xpm-dag-search-clear"
              onClick={() => setFilter("")}
              title="Clear search"
            >
              ×
            </button>
          )}
        </div>
      )}

      {mode === "tree" && (
        <div className="xpm-dag-tree-scroll">
          <ObjectTreeNode
            oid={rootObj.id}
            objects={objects}
            ancestors={[]}
            isRoot={true}
            filter={filter}
            matchesFilter={matchesFilter}
            groupFields={groupFields}
            sharedCounts={sharedCounts}
            onOpenJob={onOpenJob}
          />
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
  matchesFilter: (text: string) => boolean;
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
  matchesFilter,
  groupFields,
  sharedCounts,
  onOpenJob,
}: ObjectTreeNodeProps) => {
  const [expanded, setExpanded] = useState<boolean>(true);
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
        {label && <span className="xpm-dag-field-name">{label}: </span>}
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
          onClick={() => setExpanded(!expanded)}
          title="Click to toggle expansion"
        >
          <span className="xpm-dag-tree-toggle">{expanded ? "▼" : "▶"}</span>
          {label && <span className="xpm-dag-field-name">{label}: </span>}
          <span className="xpm-dag-type-badge">{typeName}</span>
          <span className="xpm-dag-tree-hash" title={obj.identifier || String(oid)}>
            [{shortId}]
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
                      matchesFilter={matchesFilter}
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
                      matchesFilter={matchesFilter}
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
                      matchesFilter={matchesFilter}
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
                      matchesFilter={matchesFilter}
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
                      matchesFilter={matchesFilter}
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
                        matchesFilter={matchesFilter}
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
                        matchesFilter={matchesFilter}
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
                        matchesFilter={matchesFilter}
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
                      matchesFilter={matchesFilter}
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
  matchesFilter,
}: {
  name: string;
  value: unknown;
  filter: string;
  matchesFilter: (text: string) => boolean;
}) => {
  const valStr = formatScalar(value);
  const match = matchesFilter(name) || matchesFilter(valStr);

  if (filter && !match) {
    return null;
  }

  return (
    <div className={`xpm-dag-tree-row ${filter && match ? "matched" : ""}`}>
      <span className="xpm-dag-field-name">{name}: </span>
      <span className={`xpm-dag-field-val ${scalarTypeClass(value)}`}>{valStr}</span>
    </div>
  );
};

// Row displaying a path parameter with copy action
const PathValueRow = ({
  name,
  spec,
  filter,
  matchesFilter,
}: {
  name: string;
  spec: SerializedPathRef;
  filter: string;
  matchesFilter: (text: string) => boolean;
}) => {
  const [copied, setCopied] = useState(false);
  const pathStr = spec.value;
  const match = matchesFilter(name) || matchesFilter(pathStr);

  if (filter && !match) {
    return null;
  }

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    copyText(pathStr);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <div className={`xpm-dag-tree-row xpm-dag-path-row ${filter && match ? "matched" : ""}`}>
      <span className="xpm-dag-field-name">{name}: </span>
      <span className="xpm-dag-path-val" title={pathStr}>
        {pathStr}
      </span>
      {spec.base && <span className="xpm-dag-path-base">[{spec.base}]</span>}
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
  matchesFilter,
  groupFields,
  sharedCounts,
  onOpenJob,
}: {
  name: string;
  value: unknown;
  objects: { [id: string]: SerializedObject };
  ancestors: number[];
  filter: string;
  matchesFilter: (text: string) => boolean;
  groupFields: boolean;
  sharedCounts: Map<number, number>;
  onOpenJob?: (node: GraphNode) => void;
}) => {
  const [expanded, setExpanded] = useState<boolean>(true);
  const isArr = Array.isArray(value);
  const items = isArr
    ? (value as unknown[]).map((v, i) => [`[${i}]`, v] as [string, unknown])
    : Object.entries(value as Record<string, unknown>);

  const count = items.length;

  return (
    <div className="xpm-dag-collection">
      <div
        className="xpm-dag-tree-header"
        onClick={() => setExpanded(!expanded)}
        title="Click to toggle collection"
      >
        <span className="xpm-dag-tree-toggle">{expanded ? "▼" : "▶"}</span>
        <span className="xpm-dag-field-name">{name}: </span>
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
                  matchesFilter={matchesFilter}
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
                  matchesFilter={matchesFilter}
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
                  matchesFilter={matchesFilter}
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
                matchesFilter={matchesFilter}
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

