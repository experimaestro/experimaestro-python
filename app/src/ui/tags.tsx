import React from "react";

/** Hue (0-360) of a string, stable across views and reloads */
export function hueOf(text: string): number {
  let hash = 0;
  for (let i = 0; i < text.length; i++) hash = (hash * 31 + text.charCodeAt(i)) | 0;
  return Math.abs(hash) % 360;
}

/**
 * Tag badges (key | value): each key has its own colour so that the same tag
 * is easy to spot from one row to the next
 */
export const TagBadges = ({ tags }: { tags: Array<[string, any]> }) => {
  if (tags.length === 0) return <span className="text-muted">—</span>;
  return (
    <>
      {tags.map(([key, value]) => {
        const hue = hueOf(key);
        return (
          <span key={key} className="tag" title={`${key} = ${value}`}>
            <span className="name" style={{ background: `hsl(${hue} 55% 38%)` }}>
              {key}
            </span>
            <span className="value" style={{ background: `hsl(${hue} 55% 50%)` }}>
              {String(value)}
            </span>
          </span>
        );
      })}
    </>
  );
};
