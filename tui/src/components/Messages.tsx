import React from "react";
import { Box, Text } from "ink";
import { useStore, type Message } from "../state/store.js";
import { resolveTheme } from "../themes/index.js";
import { MessageItem } from "./MessageItem.js";

export interface MessagesProps {
  /** 可见区域最大高度（行）。 */
  maxRows: number;
}

/**
 * 消息列表容器。
 *
 * 支持虚拟滚动：
 *   - `viewOffset=0` → 展示最近的一批消息（贴底）
 *   - `viewOffset>0` → 向上滑了 N 条（尾部跳过 N 条）
 *
 * 键位由 App 绑定（PageUp/Down/Home/End/Ctrl+U/D）。
 *
 * 裁剪算法：以行数为权重向前累加直到塞满；每条消息按 `\n` 数估算高度。
 */
export function Messages({ maxRows }: MessagesProps): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const messages = useStore((s) => s.messages);
  const connected = useStore((s) => s.connected);
  const viewOffset = useStore((s) => s.viewOffset);

  if (messages.length === 0) {
    return (
      <Box flexDirection="column" paddingX={1} overflow="hidden">
        <Text color={theme.colors.textDim}>
          {connected ? "欢迎使用 JimiAgent。输入内容开始对话，/help 查看命令。" : "正在连接 tui_worker…"}
        </Text>
      </Box>
    );
  }

  const innerRows = Math.max(maxRows - 2, 1); // 给顶部/底部滚动指示保留 2 行
  const picked = pickVisibleMessages(messages, innerRows, viewOffset);
  const visible = picked.slice;
  const olderCount = picked.startIdx; // 更早没看到的数量
  const newerCount = messages.length - picked.endIdx; // 更新没看到的数量

  return (
    <Box flexDirection="column" paddingX={1} overflow="hidden" height={maxRows}>
      <ScrollHint
        direction="up"
        count={olderCount}
        theme={theme}
      />
      {visible.map((m) => (
        <MessageItem key={m.id} message={m} />
      ))}
      <ScrollHint
        direction="down"
        count={newerCount}
        theme={theme}
      />
    </Box>
  );
}

function ScrollHint({
  direction,
  count,
  theme,
}: {
  direction: "up" | "down";
  count: number;
  theme: ReturnType<typeof resolveTheme>;
}): React.ReactElement | null {
  if (count <= 0) return null;
  const arrow = direction === "up" ? "↑" : "↓";
  const hint =
    direction === "up"
      ? "PgUp / k 继续向上"
      : "PgDn / End 回到最新";
  return (
    <Text color={theme.colors.textDim} italic>
      {arrow} 还有 {count} 条 · {hint}
    </Text>
  );
}

/**
 * 返回可见窗口：
 *   - `startIdx` - 窗口起点在 messages 中的位置
 *   - `endIdx`   - 窗口终点 (exclusive)
 *   - `slice`    - 切片
 */
function pickVisibleMessages(
  messages: readonly Message[],
  maxRows: number,
  viewOffset: number,
): { slice: readonly Message[]; startIdx: number; endIdx: number } {
  const total = messages.length;
  // 锚点：从尾部往前数 viewOffset 条做 endIdx
  const endIdx = Math.max(1, total - viewOffset);

  // 估算每条的行数，向前累加
  const rowsOf = (m: Message) =>
    Math.max(1, m.content.split(/\r?\n/).length + 1);

  let used = 0;
  let startIdx = endIdx;
  for (let i = endIdx - 1; i >= 0; i--) {
    const r = rowsOf(messages[i]!);
    if (used + r > maxRows && startIdx !== endIdx) break;
    used += r;
    startIdx = i;
  }
  return {
    slice: messages.slice(startIdx, endIdx),
    startIdx,
    endIdx,
  };
}
