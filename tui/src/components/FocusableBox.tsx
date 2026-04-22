import React from "react";
import { Box, type BoxProps } from "ink";
import { useStore, type FocusTarget } from "../state/store.js";
import { resolveTheme } from "../themes/index.js";

export interface FocusableBoxProps extends Omit<BoxProps, "borderColor"> {
  /** 当 store.focus === target 时 borderColor 变为 borderActive。 */
  target: FocusTarget;
  children?: React.ReactNode;
}

/**
 * 带焦点视觉反馈的 Box。
 *
 * - 活跃时：`theme.colors.borderActive`（主色）
 * - 非活跃时：`theme.colors.border`（弱化灰）
 *
 * 默认 `borderStyle="round"`；外部可覆盖。
 */
export function FocusableBox({
  target,
  children,
  borderStyle = "round",
  ...rest
}: FocusableBoxProps): React.ReactElement {
  const focus = useStore((s) => s.focus);
  const theme = resolveTheme(useStore((s) => s.theme));
  const active = focus === target;
  return (
    <Box
      borderStyle={borderStyle}
      borderColor={active ? theme.colors.borderActive : theme.colors.border}
      {...rest}
    >
      {children}
    </Box>
  );
}
