import React from "react";
import { Box, Text, useInput } from "ink";
import { useStore } from "../state/store.js";
import { resolveTheme } from "../themes/index.js";
import { useAgent } from "../hooks/useAgent.js";

/**
 * 当有 confirm_required 挂起时在 prompt 上方展示，捕获 y/n 键。
 *
 * 用户按 y → approve → Python 恢复流；按 n → deny → Python 收到拒绝事件。
 * 其他按键忽略，避免误操作。
 */
export function ConfirmBanner(): React.ReactElement | null {
  const theme = resolveTheme(useStore((s) => s.theme));
  const pending = useStore((s) => s.pendingConfirm);
  const { approve, deny } = useAgent();

  useInput(
    (input) => {
      const ch = input.toLowerCase();
      if (ch === "y") approve();
      else if (ch === "n") deny();
    },
    { isActive: !!pending },
  );

  if (!pending) return null;

  const summary =
    (typeof pending.payload?.summary === "string"
      ? pending.payload.summary
      : JSON.stringify(pending.payload)) ?? "请求确认";

  return (
    <Box
      borderStyle="double"
      borderColor={theme.colors.confirm}
      paddingX={1}
      flexDirection="column"
    >
      <Text color={theme.colors.confirm} bold>
        ⚠ 等待确认
      </Text>
      <Text color={theme.colors.text}>{summary}</Text>
      <Text color={theme.colors.textDim}>
        按 <Text color={theme.colors.success} bold>y</Text> 允许 ·{" "}
        <Text color={theme.colors.error} bold>n</Text> 拒绝
      </Text>
    </Box>
  );
}
