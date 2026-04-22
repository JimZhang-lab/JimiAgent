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
 * 会话列表面板。
 *
 * 支持光标移动、搜索、多选、批量删除和新建会话。
 */
export function HistoryPanel({
  onClose,
}: HistoryPanelProps): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const sessions = useStore((s) => s.sessions);
  // 预留底部 UI 高度，避免面板溢出进 scrollback。
  const termRows = useStore((s) => s.dims.rows);
  const maxRows = Math.max(10, termRows - 10);
  const currentId = useStore((s) => s.currentSessionId);
  const { switchSession, deleteSession, newSession } = useAgent();
  const selection = useStore((s) => s.sessionSelection);
  const toggleSelection = useStore((s) => s.toggleSessionSelection);
  const setSelection = useStore((s) => s.setSessionSelection);
  const clearSelection = useStore((s) => s.clearSessionSelection);

  const [cursor, setCursor] = useState(0);
  /** 单删时待确认的 id；批量删除走 pendingBulk。 */
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [pendingBulk, setPendingBulk] = useState(false);
  /** 批量删除剩余计数；0 表示空闲。 */
  const [bulkPending, setBulkPending] = useState(0);
  const [mode, setMode] = useState<Mode>("list");
  const [query, setQuery] = useState("");
  const setOverlayOwnsEscape = useStore((s) => s.setOverlayOwnsEscape);

  // search / pendingDelete / pendingBulk 时占用 Esc。
  useEffect(() => {
    const owns =
      mode === "search" || pendingDelete !== null || pendingBulk;
    setOverlayOwnsEscape(owns);
    return () => setOverlayOwnsEscape(false);
  }, [mode, pendingDelete, pendingBulk, setOverlayOwnsEscape]);

  useScopedBindings("overlay-history", [
    { key: "up/down", description: "上/下移光标" },
    { key: "return", description: "切换到选中会话（或确认删除）" },
    { key: "space", description: "切换当前项多选" },
    { key: "a", description: "全选当前过滤结果" },
    { key: "A", description: "清空选择" },
    { key: "d", description: "进入删除预备态" },
    { key: "n", description: "新建会话" },
    { key: "/", description: "搜索会话标题 / id" },
    { key: "escape", description: "关闭面板 / 退出搜索 / 取消待删除" },
  ]);

  // 搜索同时匹配 title 和 id，且不区分大小写。
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return sessions;
    return sessions.filter((s) =>
      s.title.toLowerCase().includes(q) || s.id.toLowerCase().includes(q),
    );
  }, [sessions, query]);

  useEffect(() => {
    // 过滤变化后把 cursor 裁到可见范围
    if (filtered.length === 0) {
      setCursor(0);
      return;
    }
    if (cursor >= filtered.length) setCursor(filtered.length - 1);
  }, [filtered.length, cursor]);

  useEffect(() => {
    // 初次展示时高亮当前会话
    if (query) return; // 搜索模式不覆盖 cursor
    const idx = filtered.findIndex((s) => s.id === currentId);
    if (idx >= 0) setCursor(idx);
  }, [filtered, currentId, query]);

  const resetDeletePending = () => {
    setPendingDelete(null);
    setPendingBulk(false);
  };

  useInput(
    (input, key) => {
      // search 模式里只截获 Esc
      if (mode === "search") {
        if (key.escape) {
          setMode("list");
          setQuery("");
        }
        return;
      }
      // Esc：优先取消待删除态
      if (key.escape) {
        if (pendingBulk || pendingDelete !== null) {
          resetDeletePending();
        }
        return;
      }
      if (key.upArrow) {
        setCursor((c) => Math.max(0, c - 1));
        resetDeletePending();
      } else if (key.downArrow) {
        setCursor((c) => Math.min(Math.max(0, filtered.length - 1), c + 1));
        resetDeletePending();
      } else if (key.return) {
        // 批量删除按 50ms 一条节流，避免一次打爆 worker。
        if (pendingBulk && selection.length > 0) {
          const queue = [...selection];
          setBulkPending(queue.length);
          setPendingBulk(false);
          clearSelection();
          const drain = () => {
            const id = queue.shift();
            if (id === undefined) {
              setBulkPending(0);
              return;
            }
            // 复用 useAgent.deleteSession，让后端事件把 UI 切到 fallback 会话。
            deleteSession(id);
            setBulkPending(queue.length);
            setTimeout(drain, 50);
          };
          drain();
          return;
        }
        const s = filtered[cursor];
        if (!s) return;
        if (pendingDelete === s.id) {
          deleteSession(s.id);
          setPendingDelete(null);
          return;
        }
        switchSession(s.id);
        onClose();
      } else if (input === " ") {
        // 空格切换当前行多选
        const s = filtered[cursor];
        if (s) {
          toggleSelection(s.id);
          resetDeletePending();
        }
      } else if (input === "a") {
        // 全选当前过滤结果
        setSelection(filtered.map((s) => s.id));
        resetDeletePending();
      } else if (input === "A") {
        // 清空选择
        clearSelection();
        resetDeletePending();
      } else if (input === "d") {
        // 有选中就走批删，否则单删当前行
        if (selection.length > 0) {
          setPendingBulk(true);
          setPendingDelete(null);
        } else {
          const s = filtered[cursor];
          if (s) setPendingDelete(s.id);
        }
      } else if (input === "n") {
        newSession();
        onClose();
      } else if (input === "/") {
        setMode("search");
      }
    },
    { isActive: true },
  );

  // 头部、搜索框和底部 hint 会先吃掉一部分预算行数。
  const headerRows = 1 + (mode === "search" ? 3 : 0);
  const hintRows = 2;
  const listRows = Math.max(1, maxRows - headerRows - hintRows);

  // 以 cursor 为中心做滑动窗口裁切
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
  const selectedSet = new Set(selection);

  return (
    <Box flexDirection="column" paddingX={1}>
      <Box>
        <Text color={theme.colors.primary} bold>
          会话列表（{filtered.length}
          {query ? `/${sessions.length}` : ""}）
        </Text>
        {selection.length > 0 && (
          <Text color={theme.colors.accent} bold>
            {"  "}已选 {selection.length}
          </Text>
        )}
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
              selected={selectedSet.has(s.id)}
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
      <Box marginTop={1} flexDirection="column">
        {bulkPending > 0 ? (
          <Text color={theme.colors.warning} bold>
            ⟲ 正在批量删除…剩余 {bulkPending} 条
          </Text>
        ) : pendingBulk && selection.length > 0 ? (
          <Text color={theme.colors.error} bold>
            ⚠️ 按 Enter 确认删除已选 {selection.length} 条；按 ↑/↓ 或 A / Esc 取消
          </Text>
        ) : (
          <Text color={theme.colors.textDim}>
            ↑/↓ 选 · Enter 切换 · <Text color={theme.colors.accent}>Space 多选</Text> · a 全选 · A 清选 · d+Enter 删除 · n 新建 · / 搜索 · Esc 关闭
          </Text>
        )}
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
  /** 多选集合是否包含本条；控制 checkbox 样式与 accent 强调色。 */
  selected: boolean;
  messages: number;
  updatedAt: number;
}): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const {
    title,
    id,
    active,
    current,
    pendingDelete,
    selected,
    messages,
    updatedAt,
  } = props;

  // 与 MemoryRow 一致：[✓] 选中 / [·] 光标位未选 / [ ] 既不选也不光标
  const checkbox = selected ? "[✓]" : active ? "[·]" : "[ ]";

  let marker = " ";
  if (pendingDelete) marker = "✗";
  else if (active) marker = "▶";

  const titleColor = pendingDelete
    ? theme.colors.error
    : selected
      ? theme.colors.accent
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
      <Text
        color={selected ? theme.colors.accent : theme.colors.textDim}
        bold={selected}
      >
        {checkbox}{" "}
      </Text>
      <Text color={titleColor} bold={active || current || selected}>
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
