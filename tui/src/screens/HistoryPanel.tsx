import React, { useEffect, useMemo, useState } from "react";
import { Box, Text, useInput } from "ink";
import { SmartTextInput } from "../components/SmartTextInput.js";
import { useStore } from "../state/store.js";
import { resolveTheme } from "../themes/index.js";
import { useAgent } from "../hooks/useAgent.js";
import { useScopedBindings } from "../keybindings/useKeybinding.js";

export interface HistoryPanelProps {
  onClose(): void;
}

type Mode = "list" | "search";

/**
 * 会话列表面板：
 *   - ↑/↓ 选择（滑动窗口 + 滚动指示）
 *   - Enter 切换到选中会话
 *   - d 删除会话（需二次确认）
 *   - n 新建会话
 *   - / 进入搜索模式（按 title/id 子串匹配）
 */
export function HistoryPanel({
  onClose,
}: HistoryPanelProps): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const sessions = useStore((s) => s.sessions);
  // 面板自身计算可用高度：整屏高 - 底部动态 UI 预留（Header 2 + Activity 2 +
  // Prompt 3 + Status 2 ≈ 9 行，保守 10 行 buffer），避免溢出被 scrollback 吞。
  const termRows = useStore((s) => s.dims.rows);
  const maxRows = Math.max(10, termRows - 10);
  const currentId = useStore((s) => s.currentSessionId);
  const { switchSession, deleteSession, newSession } = useAgent();
  const [cursor, setCursor] = useState(0);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("list");
  const [query, setQuery] = useState("");
  const setOverlayOwnsEscape = useStore((s) => s.setOverlayOwnsEscape);

  // 子模式（search 或 pendingDelete）时占用 Esc，不允许 App 关面板
  useEffect(() => {
    const owns = mode === "search" || pendingDelete !== null;
    setOverlayOwnsEscape(owns);
    return () => setOverlayOwnsEscape(false);
  }, [mode, pendingDelete, setOverlayOwnsEscape]);

  useScopedBindings("overlay-history", [
    { key: "up/down", description: "上/下移光标" },
    { key: "return", description: "切换到选中会话（或确认删除）" },
    { key: "d", description: "删除会话（再按 Enter 确认）" },
    { key: "n", description: "新建会话" },
    { key: "/", description: "搜索会话标题 / id" },
    { key: "escape", description: "关闭面板 / 退出搜索" },
  ]);

  // 过滤后的可见列表；搜索为子串不区分大小写，同时对标题和 id 生效
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return sessions;
    return sessions.filter((s) =>
      s.title.toLowerCase().includes(q) || s.id.toLowerCase().includes(q),
    );
  }, [sessions, query]);

  useEffect(() => {
    // 过滤变化后 cursor 裁切到可见范围
    if (filtered.length === 0) {
      setCursor(0);
      return;
    }
    if (cursor >= filtered.length) setCursor(filtered.length - 1);
  }, [filtered.length, cursor]);

  useEffect(() => {
    // 初次展示时，把当前会话在列表里高亮
    if (query) return; // 搜索模式不覆盖 cursor
    const idx = filtered.findIndex((s) => s.id === currentId);
    if (idx >= 0) setCursor(idx);
  }, [filtered, currentId, query]);

  useInput(
    (input, key) => {
      // search 模式：TextInput 最大程度接管，仅截获 Esc
      if (mode === "search") {
        if (key.escape) {
          setMode("list");
          setQuery("");
        }
        return;
      }
      // Esc：有待删除确认先取消；否则交还给 App 关面板
      if (key.escape) {
        if (pendingDelete !== null) setPendingDelete(null);
        return;
      }
      if (key.upArrow) {
        setCursor((c) => Math.max(0, c - 1));
        setPendingDelete(null);
      } else if (key.downArrow) {
        setCursor((c) => Math.min(Math.max(0, filtered.length - 1), c + 1));
        setPendingDelete(null);
      } else if (key.return) {
        const s = filtered[cursor];
        if (!s) return;
        if (pendingDelete === s.id) {
          deleteSession(s.id);
          setPendingDelete(null);
          return;
        }
        switchSession(s.id);
        onClose();
      } else if (input === "d") {
        const s = filtered[cursor];
        if (!s) return;
        setPendingDelete(s.id);
      } else if (input === "n") {
        newSession();
        onClose();
      } else if (input === "/") {
        setMode("search");
      }
    },
    { isActive: true },
  );

  // 头部标题 + 结果统计占 1 行；搜索框 3 行；底部 hint 1 行 + marginTop 1
  const headerRows = 1 + (mode === "search" ? 3 : 0);
  const hintRows = 2;
  const listRows = Math.max(1, maxRows - headerRows - hintRows);

  // 滑动窗口：以 cursor 为中心，裁切到 listRows
  const windowStart = Math.max(
    0,
    Math.min(
      cursor - Math.floor(listRows / 2),
      Math.max(0, filtered.length - listRows),
    ),
  );
  const visible = filtered.slice(windowStart, windowStart + listRows);
  const hasAbove = windowStart > 0;
  const hasBelow = windowStart + listRows < filtered.length;

  return (
    <Box flexDirection="column" paddingX={1}>
      <Box>
        <Text color={theme.colors.primary} bold>
          会话列表（{filtered.length}
          {query ? `/${sessions.length}` : ""}）
        </Text>
        {query && (
          <Text color={theme.colors.textDim}>
            {"  "}搜索: <Text color={theme.colors.info}>{query}</Text>
          </Text>
        )}
      </Box>

      {mode === "search" && (
        <Box
          marginTop={1}
          borderStyle="round"
          borderColor={theme.colors.borderActive}
          paddingX={1}
        >
          <Text color={theme.colors.primary}>🔍 </Text>
          <SmartTextInput
            value={query}
            onChange={setQuery}
            onSubmit={() => setMode("list")}
            placeholder="按标题 / id 过滤…Enter 确认 · Esc 取消"
            showCursor
          />
        </Box>
      )}

      <Box marginTop={1} flexDirection="column">
        {hasAbove && (
          <Text color={theme.colors.textDim}>
            ↑ 还有 {windowStart} 条
          </Text>
        )}
        {visible.length === 0 ? (
          <Text color={theme.colors.textDim}>
            {query ? "（无匹配）" : "暂无会话。按 [n] 新建。"}
          </Text>
        ) : (
          visible.map((s, i) => (
            <SessionRow
              key={s.id}
              title={s.title}
              id={s.id}
              active={windowStart + i === cursor}
              current={s.id === currentId}
              pendingDelete={pendingDelete === s.id}
              messages={s.message_count}
              updatedAt={s.updated_at}
            />
          ))
        )}
        {hasBelow && (
          <Text color={theme.colors.textDim}>
            ↓ 还有 {filtered.length - windowStart - listRows} 条
          </Text>
        )}
      </Box>
      <Box marginTop={1}>
        <Text color={theme.colors.textDim}>
          ↑/↓ 选择 · Enter 切换 · n 新建 · d+Enter 删除 · / 搜索 · Esc 关闭
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

/**
 * 时间戳格式化：根据量级自动判断秒/毫秒。
 *
 * Python session.updated_at 目前写的是 `time.time()` 秒级；但列 gateway 统计时
 * 偶尔换成毫秒；历史日志也可能遗留非标准值。取 2000-01-01 ~ 2100-01-01 区间作
 * 正常域：
 *   - 秒级在 9.5e8 ~ 4.1e9
 *   - 毫秒级在 9.5e11 ~ 4.1e12
 * 超出这两档的都当"异常"显示原值便于调试。
 */
function formatTime(ts: number): string {
  if (!Number.isFinite(ts) || ts <= 0) return "--";
  let ms: number;
  if (ts >= 1e11) {
    ms = ts; // 毫秒
  } else if (ts >= 1e8) {
    ms = ts * 1000; // 秒
  } else {
    return String(ts); // 明显异常
  }
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) return String(ts);
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mi = String(d.getMinutes()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd} ${hh}:${mi}`;
}
