// Job filter input, with completions computed by the server from the filter
// language (complete_filter in cli/filter.py) and the displayed jobs
import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Form } from "react-bootstrap";

import client, { FilterSuggestion, JobsCompleteResult } from "./client";

type Completion = JobsCompleteResult & {
  // Query and cursor the completion was computed for
  query: string;
  cursor: number;
};

// "auto": opened by typing; "explicit": asked for (Ctrl+Space or ↓)
type Mode = "closed" | "auto" | "explicit";

type Props = {
  id: string;
  value: string;
  onChange: (value: string) => void;
  // Escape in the input (with no completion shown): clears the filter
  onClear: () => void;
  inputRef: React.RefObject<HTMLInputElement>;
  experimentId: string | null;
  runId: string | null;
  placeholder?: string;
  isInvalid?: boolean;
  isValid?: boolean;
};

/** The suggestions to show: while typing, the boolean operators are only
 * shown once their first letter is typed (Enter would insert them) */
function shownItems(mode: Mode, completion: Completion | null): FilterSuggestion[] {
  if (mode === "closed" || completion === null) return [];
  if (
    mode === "auto" &&
    completion.start === completion.end &&
    completion.items.every((item) => item.kind === "keyword")
  )
    return [];
  return completion.items;
}

export default function FilterInput({
  id,
  value,
  onChange,
  onClear,
  inputRef,
  experimentId,
  runId,
  placeholder,
  isInvalid,
  isValid,
}: Props) {
  const [mode, setMode] = useState<Mode>("closed");
  const [completion, setCompletion] = useState<Completion | null>(null);
  const [selected, setSelected] = useState(0);
  const requestId = useRef(0);
  const lastRequest = useRef<{ query: string; cursor: number } | null>(null);
  // Caret position to restore after accepting a suggestion
  const pendingCaret = useRef<number | null>(null);
  const listRef = useRef<HTMLUListElement>(null);

  useEffect(() => {
    client.onJobsComplete = (result: JobsCompleteResult) => {
      // Ignore answers to outdated requests
      if (result.requestId !== requestId.current || !lastRequest.current) return;
      setCompletion({ ...result, ...lastRequest.current });
      setSelected(0);
    };
    return () => {
      client.onJobsComplete = null;
    };
  }, []);

  // Another experiment or run: other tags and task identifiers
  useEffect(() => close(), [experimentId, runId]);

  useLayoutEffect(() => {
    if (pendingCaret.current !== null && inputRef.current) {
      inputRef.current.setSelectionRange(pendingCaret.current, pendingCaret.current);
      pendingCaret.current = null;
    }
  }, [value]);

  // Keep the selected suggestion visible
  useEffect(() => {
    listRef.current
      ?.querySelector(`[data-index="${selected}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [selected, completion]);

  function request(query: string, cursor: number) {
    const last = lastRequest.current;
    if (last && last.query === query && last.cursor === cursor) return;
    lastRequest.current = { query, cursor };
    requestId.current += 1;
    client.jobs_complete(query, cursor, requestId.current, experimentId, runId);
  }

  function close() {
    setMode("closed");
    setCompletion(null);
    lastRequest.current = null;
    requestId.current += 1;
  }

  function open(newMode: Mode) {
    const input = inputRef.current;
    if (!input) return;
    setMode(newMode);
    request(input.value, input.selectionStart ?? input.value.length);
  }

  function accept(item: FilterSuggestion) {
    if (!completion) return;
    const { start, end } = completion;
    const newValue = value.slice(0, start) + item.text + value.slice(end);
    const caret = start + item.text.length;
    pendingCaret.current = caret;
    onChange(newValue);
    // Suggest what comes next
    setMode("auto");
    request(newValue, caret);
  }

  const items = shownItems(mode, completion);

  function onKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    const input = event.currentTarget;
    // The completion was computed for what is in the input
    const upToDate =
      completion !== null &&
      completion.query === input.value &&
      completion.cursor === (input.selectionStart ?? input.value.length);
    if (items.length > 0) {
      switch (event.key) {
        case "ArrowDown":
        case "ArrowUp": {
          const delta = event.key === "ArrowDown" ? 1 : -1;
          setSelected((selected + delta + items.length) % items.length);
          event.preventDefault();
          return;
        }
        case "Tab":
        case "Enter":
          if (event.shiftKey) break;
          event.preventDefault();
          // Ignore the key until the completion of the current text arrives
          if (upToDate) accept(items[Math.min(selected, items.length - 1)]);
          return;
        case "Escape":
          event.preventDefault();
          close();
          return;
      }
    }
    if ((event.key === " " && event.ctrlKey) || event.key === "ArrowDown") {
      event.preventDefault();
      open("explicit");
    } else if (event.key === "Escape") {
      onClear();
      close();
      input.blur();
    }
  }

  const optionId = (index: number) => `${id}-completion-${index}`;

  return (
    <div className="filter-input">
      <Form.Control
        id={id}
        ref={inputRef}
        value={value}
        autoComplete="off"
        spellCheck={false}
        role="combobox"
        aria-autocomplete="list"
        aria-expanded={items.length > 0}
        aria-controls={`${id}-completions`}
        aria-activedescendant={items.length > 0 ? optionId(selected) : undefined}
        onChange={(e) => {
          onChange(e.target.value);
          setMode((current) => (current === "explicit" ? current : "auto"));
          const input = e.target as HTMLInputElement;
          request(input.value, input.selectionStart ?? input.value.length);
        }}
        onSelect={(e) => {
          // The caret moved: complete at its new position
          if (mode === "closed") return;
          const input = e.target as HTMLInputElement;
          request(input.value, input.selectionStart ?? input.value.length);
        }}
        onKeyDown={onKeyDown}
        onBlur={close}
        placeholder={placeholder}
        isInvalid={isInvalid}
        isValid={isValid}
      />
      {items.length > 0 && (
        <ul
          id={`${id}-completions`}
          ref={listRef}
          role="listbox"
          className="filter-completions dropdown-menu show"
        >
          {items.map((item, index) => (
            <li key={`${item.kind}:${item.label}`} role="none">
              <button
                type="button"
                id={optionId(index)}
                role="option"
                aria-selected={index === selected}
                data-index={index}
                data-kind={item.kind}
                className={`dropdown-item${index === selected ? " active" : ""}`}
                tabIndex={-1}
                // Keeps the focus in the input
                onMouseDown={(e) => {
                  e.preventDefault();
                  accept(item);
                }}
                onMouseEnter={() => setSelected(index)}
              >
                <code className="completion-label">{item.label}</code>
                {item.detail && (
                  <span className="completion-detail">{item.detail}</span>
                )}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
