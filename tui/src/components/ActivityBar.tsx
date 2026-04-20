import React from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";
import { useStore } from "../state/store.js";
import { resolveTheme } from "../themes/index.js";

/**
 * 活动追踪条：在 PromptInput 上方持续显示 "agent 正在做什么"。
 *
 * 来源：
 *   - `tool` 事件 → kind=tool, label=工具名
 *   - 首个 `text` chunk → kind=thinking, label="生成回复中"
 *   - `confirm_required` → kind=confirm, label=summary
 *   - `done` / `error` → 清零
 *
 * 为避免闪烁，动作变更会"滑过"最少 150ms 的 duration 之后才更新显示，但此
 * 最小展示时间由上层 `endActivity` 写日志的 duration 控制；此处仅渲染当前
 * 快照，所以瞬时切换也 OK——Ink 的 diff 本来就很省。
 */
export function ActivityBar(): React.ReactElement | null {
  const theme = resolveTheme(useStore((s) => s.theme));
  const activity = useStore((s) => s.activity);
  const streaming = useStore((s) => s.streaming);
  const [, tick] = React.useReducer((x: number) => x + 1, 0);

  // 每 500ms 重绘一次，用于刷新 elapsed
  React.useEffect(() => {
    if (!activity) return;
    const t = setInterval(tick, 500);
    return () => clearInterval(t);
  }, [activity]);

  if (!activity && !streaming) return null;

  const icon =
    activity?.kind === "tool"
      ? "⚙"
      : activity?.kind === "confirm"
        ? "⚠"
        : "◆";
  const color =
    activity?.kind === "tool"
      ? theme.colors.tool
      : activity?.kind === "confirm"
        ? theme.colors.confirm
        : theme.colors.assistant;
  const label =
    activity?.label ?? (streaming ? "生成回复中" : "");
  const elapsed = activity ? Date.now() - activity.since : 0;
  const elapsedStr = formatElapsed(elapsed);

  return (
    <Box
      paddingX={1}
      borderStyle="single"
      borderColor={color}
      borderTop={false}
      borderLeft={false}
      borderRight={false}
      borderBottom={true}
    >
      <Text color={color}>
        <Spinner type="dots" />
        <Text> {icon} </Text>
        <Text bold>{label}</Text>
      </Text>
      <Text color={theme.colors.textDim}> · {elapsedStr}</Text>
    </Box>
  );
}

function formatElapsed(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const rem = Math.floor(s % 60);
  return `${m}m${rem}s`;
}
