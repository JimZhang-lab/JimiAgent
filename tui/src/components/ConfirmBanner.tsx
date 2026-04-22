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

  const payload = pending.payload ?? {};
  const summary =
    typeof payload.summary === "string"
      ? payload.summary
      : JSON.stringify(payload);
  // detail 可能是任意形状的对象，投影为 key=value 行。string 直接用，
  // 其他类型 JSON.stringify 再截断，避免一行太长撑爆布局。
  const detail = payload.detail && typeof payload.detail === "object"
    ? (payload.detail as Record<string, unknown>)
    : null;

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
      {detail && Object.keys(detail).length > 0 && (
        <Box flexDirection="column" marginTop={0}>
          {Object.entries(detail).slice(0, 6).map(([k, v]) => (
            <Text key={k} color={theme.colors.textDim}>
              <Text color={theme.colors.info}>{k}</Text>
              {": "}
              <Text color={theme.colors.text}>{formatDetailValue(v)}</Text>
            </Text>
          ))}
          {Object.keys(detail).length > 6 && (
            <Text color={theme.colors.textDim} italic>
              …还有 {Object.keys(detail).length - 6} 项
            </Text>
          )}
        </Box>
      )}
      <Text color={theme.colors.textDim}>
        按 <Text color={theme.colors.success} bold>[y]</Text> 允许 /{" "}
        <Text color={theme.colors.error} bold>[n]</Text> 拒绝
      </Text>
    </Box>
  );
}

/** 把 detail 值压平成一行文本，过长则截断保留前 120 字符。 */
function formatDetailValue(v: unknown): string {
  if (v === null || v === undefined) return String(v);
  if (typeof v === "string") return shrink(v);
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  try {
    return shrink(JSON.stringify(v));
  } catch {
    return "(unprintable)";
  }
}

function shrink(s: string): string {
  const flat = s.replace(/\s+/g, " ");
  return flat.length > 120 ? flat.slice(0, 117) + "…" : flat;
}
