// Keyboard shortcuts of a view, the same keys as in the TUI
import React, { useEffect, useRef } from "react";

export type Shortcuts = { [key: string]: () => void };

function isEditable(target: EventTarget | null): boolean {
  const element = target as HTMLElement | null;
  return (
    !!element &&
    (element.isContentEditable ||
      ["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName))
  );
}

/**
 * Calls the handler of a key (KeyboardEvent.key, e.g. "/" or "S") pressed
 * outside of text fields and dialogs, and without Ctrl, Alt or Meta (which
 * belong to the browser)
 */
export function useShortcuts(shortcuts: Shortcuts): void {
  const current = useRef(shortcuts);
  current.current = shortcuts;

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      if (isEditable(event.target)) return;
      if (document.querySelector(".modal.show")) return;
      const handler = current.current[event.key];
      if (handler) {
        event.preventDefault();
        handler();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
}

/** Escape in a search field clears it and leaves it (as in the TUI) */
export function clearOnEscape(clear: () => void) {
  return (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Escape") {
      clear();
      event.currentTarget.blur();
    }
  };
}
