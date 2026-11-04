import React from "react";

interface CollapseExpandIconProps {
  collapsed: boolean;
  className?: string;
}

/**
 * Vertical >< (collapse) / <> (expand) icon.
 *
 * - When collapsed is true (action is expand all):
 *   renders vertical <> (chevrons pointing outward: ^ over v).
 * - When collapsed is false (action is collapse all):
 *   renders vertical >< (chevrons pointing inward: v over ^).
 */
export const CollapseExpandIcon: React.FC<CollapseExpandIconProps> = ({
  collapsed,
  className,
}) => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 16 16"
    className={className}
    style={{ display: "block" }}
    aria-hidden="true"
  >
    {collapsed ? (
      // Vertical <> : top points up (^), bottom points down (v)
      <>
        <path
          d="M3 5.5 L8 2 L13 5.5"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M3 10.5 L8 14 L13 10.5"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </>
    ) : (
      // Vertical >< : top points down (v), bottom points up (^)
      <>
        <path
          d="M3 2 L8 5.5 L13 2"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M3 14 L8 10.5 L13 14"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </>
    )}
  </svg>
);
