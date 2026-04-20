import React, { useEffect, useState } from "react";
import { Box, Text, useInput } from "ink";
import { useStore } from "../state/store.js";
import { resolveTheme } from "../themes/index.js";
import { useAgent } from "../hooks/useAgent.js";

export interface HistoryPanelProps {
  onClose(): void;
  maxRows: number;
}

/**
 * 会话列表面板：
 *   - ↑/↓ 选择
 *   - Enter 切换到选中会话
 *   - d 删除会话（需二次确认）
 *   - n 新建会话
 */
export function HistoryPanel({
  onClose,
  maxRows,
}: HistoryPanelProps): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const sessions = useStore((s) => s.sessions);
  const currentId = useStore((s) => s.currentSessionId);
  const { switchSession, deleteSession, newSession } = useAgent();
  const [cursor, setCursor] = useState(0);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);

  useEffect(() => {
    // 当前会话默认高亮
    const idx = sessions.findIndex((s) => s.id === currentId);
    if (idx >= 0) setCursor(idx);
  }, [sessions, currentId]);

  useInput(
    (input, key) => {
      if (key.upArrow) {
        setCursor((c) => Math.max(0, c - 1));
        setPendingDelete(null);
      } else if (key.downArrow) {
        setCursor((c) => Math.min(sessions.length - 1, c + 1));
        setPendingDelete(null);
      } else if (key.return) {
        const s = sessions[cursor];
        if (!s) return;
        if (pendingDelete === s.id) {
          deleteSession(s.id);
          setPendingDelete(null);
          return;
        }
        switchSession(s.id);
        onClose();
      } else if (input === "d") {
        const s = sessions[cursor];
        if (!s) return;
        setPendingDelete(s.id);
      } else if (input === "n") {
        newSession();
        onClose();
      }
    },
    { isActive: true },
  );

  const headerRows = 2;
  const listRows = Math.max(1, maxRows - headerRows - 2);
  const visible = sessions.slice(0, listRows);

  return (
    <Box flexDirection="column" paddingX={1}>
      <Box>
        <Text color={theme.colors.primary} bold>
          会话列表（{sessions.length}）
        </Text>
      </Box>
      <Box marginTop={1} flexDirection="column">
        {visible.length === 0 ? (
          <Text color={theme.colors.textDim}>暂无会话。按 [n] 新建。</Text>
        ) : (
          visible.map((s, i) => (
            <SessionRow
              key={s.id}
              title={s.title}
              id={s.id}
              active={i === cursor}
              current={s.id === currentId}
              pendingDelete={pendingDelete === s.id}
              messages={s.message_count}
              updatedAt={s.updated_at}
            />
          ))
        )}
        {sessions.length > listRows && (
          <Text color={theme.colors.textDim}>
            …还有 {sessions.length - listRows} 条
          </Text>
        )}
      </Box>
      <Box marginTop={1}>
        <Text color={theme.colors.textDim}>
          ↑/↓ 选择 · Enter 切换 · n 新建 · d 删除（再按 Enter 确认）· Esc 关闭
        </Text>
      </Box>
    </Box>
  );
}

function SessionRow(props: {
  title: string;
  id: string;
  active: boolean;
  current: boolean;
  pendingDelete: boolean;
  messages: number;
  updatedAt: number;
}): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const { title, id, active, current, pendingDelete, messages, updatedAt } = props;

  let marker = " ";
  if (pendingDelete) marker = "✗";
  else if (active) marker = "▶";

  const titleColor = pendingDelete
    ? theme.colors.error
    : active
      ? theme.colors.primary
      : current
        ? theme.colors.success
        : theme.colors.text;

  return (
    <Box flexDirection="row">
      <Text color={pendingDelete ? theme.colors.error : theme.colors.primary}>
        {marker}{" "}
      </Text>
      <Text color={titleColor} bold={active || current}>
        {current ? "● " : "  "}
        {title}
      </Text>
      <Text color={theme.colors.textDim}>
        {" "}
        [{id}] · msgs:{messages} · {formatTime(updatedAt)}
      </Text>
      {pendingDelete && (
        <Text color={theme.colors.error}> (按 Enter 确认删除)</Text>
      )}
    </Box>
  );
}

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mi = String(d.getMinutes()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd} ${hh}:${mi}`;
}
