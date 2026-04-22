import React, { useEffect, useState } from "react";
import { Box, Text, useInput } from "ink";
import { SmartTextInput } from "../components/SmartTextInput.js";
import { useStore } from "../state/store.js";
import { resolveTheme } from "../themes/index.js";
import { sendRequest } from "../state/wiring.js";
import { useScopedBindings } from "../keybindings/useKeybinding.js";

export interface MemoryPanelProps {
  onClose(): void;
}

type Mode = "list" | "search";

/**
 * 记忆管理面板：
 *   - 默认浏览最近 100 条（list_recent）
 *   - 按 `/` 切到搜索模式，输入后 Enter 触发 search_memories
 *   - ↑/↓ 选中；Enter 展开详情（toggle）；d → Enter 确认删除
 *   - k 过滤：按 k 循环 all/semantic/episodic/procedural/triple
 *   - r 刷新
 */
export function MemoryPanel({
  onClose: _onClose,
}: MemoryPanelProps): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const memories = useStore((s) => s.memories);
  // 面板自读终端高度，保留 10 行给底部动态 UI
  const termRows = useStore((s) => s.dims.rows);
  const maxRows = Math.max(10, termRows - 10);
  const loading = useStore((s) => s.memoryLoading);
  const currentQuery = useStore((s) => s.memoryQuery);
  const setMemoryLoading = useStore((s) => s.setMemoryLoading);
  const selection = useStore((s) => s.memorySelection);
  const toggleSelection = useStore((s) => s.toggleMemorySelection);
  const setSelection = useStore((s) => s.setMemorySelection);
  const clearSelection = useStore((s) => s.clearMemorySelection);

  const [cursor, setCursor] = useState(0);
  const [expanded, setExpanded] = useState(false);
  /** 单选的"待确认"id；多选时 pendingBulk=true 代替。 */
  const [pendingDelete, setPendingDelete] = useState<number | null>(null);
  const [pendingBulk, setPendingBulk] = useState(false);
  const [mode, setMode] = useState<Mode>("list");
  const [searchDraft, setSearchDraft] = useState(currentQuery);
  const [kindFilter, setKindFilter] = useState<string>("all");
  /** 批量删除进行中的剩余计数；0 表示空闲。用于面板底部显示进度。 */
  const [bulkPending, setBulkPending] = useState(0);
  const setOverlayOwnsEscape = useStore((s) => s.setOverlayOwnsEscape);

  // 把"是否有子模式占用 Esc"同步给全局 Esc 处理器。
  // - search 模式：Esc 退回列表，不能让 App 把整个面板关掉
  // - pendingDelete / pendingBulk：Esc 取消确认态，不能让 App 关面板
  useEffect(() => {
    const owns = mode === "search" || pendingDelete !== null || pendingBulk;
    setOverlayOwnsEscape(owns);
    return () => setOverlayOwnsEscape(false);
  }, [mode, pendingDelete, pendingBulk, setOverlayOwnsEscape]);

  useScopedBindings("overlay-memory", [
    { key: "up/down", description: "上/下移光标" },
    { key: "return", description: "展开/收起详情；批量模式下确认删除" },
    { key: "space", description: "切换当前项多选" },
    { key: "a", description: "全选当前过滤结果" },
    { key: "A", description: "清空选择" },
    { key: "d", description: "进入删除预备态" },
    { key: "r", description: "刷新列表或重跑当前查询" },
    { key: "/", description: "进入搜索模式" },
    { key: "k", description: "循环切换 kind 过滤器" },
  ]);

  // 首次打开：拉最近
  useEffect(() => {
    setMemoryLoading(true);
    sendRequest({ kind: "list_memories", limit: 100 });
  }, [setMemoryLoading]);

  // cursor clamp
  useEffect(() => {
    if (memories.length === 0) {
      setCursor(0);
      return;
    }
    if (cursor >= memories.length) {
      setCursor(Math.max(0, memories.length - 1));
    }
  }, [memories.length, cursor]);

  const kindCycle = ["all", "semantic", "episodic", "procedural", "triple"];
  const filtered = kindFilter === "all"
    ? memories
    : memories.filter((m) => m.kind === kindFilter);

  const resetDeletePending = () => {
    setPendingDelete(null);
    setPendingBulk(false);
  };

  useInput(
    (input, key) => {
      if (mode === "search") {
        if (key.escape) {
          setMode("list");
          setSearchDraft(currentQuery);
        }
        // TextInput 处理其他按键
        return;
      }
      // list 模式下 Esc：先取消待删除态；待删除为 null 时交回全局 Esc 关面板
      if (key.escape) {
        if (pendingBulk || pendingDelete !== null) {
          resetDeletePending();
        }
        return;
      }
      if (key.upArrow) {
        setCursor((c) => Math.max(0, c - 1));
        resetDeletePending();
        setExpanded(false);
      } else if (key.downArrow) {
        setCursor((c) => Math.min(Math.max(0, filtered.length - 1), c + 1));
        resetDeletePending();
        setExpanded(false);
      } else if (key.return) {
        // 批量删除确认优先。
        // 把请求节流到每 50ms 一个，避免 worker 端一次收到上百条请求后阻塞事件循环。
        // 同时记下剩余计数，status bar 会显示进度。
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
            sendRequest({ kind: "delete_memory", memory_id: id });
            setBulkPending(queue.length);
            setTimeout(drain, 50);
          };
          drain();
          return;
        }
        const m = filtered[cursor];
        if (!m) return;
        if (pendingDelete === m.id) {
          sendRequest({ kind: "delete_memory", memory_id: m.id });
          setPendingDelete(null);
          return;
        }
        setExpanded((e) => !e);
      } else if (input === " ") {
        // 空格切换当前项选中
        const m = filtered[cursor];
        if (m) {
          toggleSelection(m.id);
          resetDeletePending();
        }
      } else if (input === "a") {
        // 全选当前过滤结果
        setSelection(filtered.map((m) => m.id));
        resetDeletePending();
      } else if (input === "A") {
        // 清空选择
        clearSelection();
        resetDeletePending();
      } else if (input === "d") {
        if (selection.length > 0) {
          setPendingBulk(true);
          setPendingDelete(null);
        } else {
          const m = filtered[cursor];
          if (m) setPendingDelete(m.id);
        }
      } else if (input === "r") {
        resetDeletePending();
        setMemoryLoading(true);
        if (currentQuery) {
          sendRequest({ kind: "search_memories", query: currentQuery, k: 50 });
        } else {
          sendRequest({ kind: "list_memories", limit: 100 });
        }
      } else if (input === "/") {
        setMode("search");
        setSearchDraft("");
      } else if (input === "k") {
        const idx = kindCycle.indexOf(kindFilter);
        setKindFilter(kindCycle[(idx + 1) % kindCycle.length] ?? "all");
      }
    },
    { isActive: true },
  );

  const listRows = Math.max(1, maxRows - (mode === "search" ? 5 : 4) - (expanded ? 4 : 0));
  const windowStart = Math.max(
    0,
    Math.min(
      cursor - Math.floor(listRows / 2),
      Math.max(0, filtered.length - listRows),
    ),
  );
  const visible = filtered.slice(windowStart, windowStart + listRows);
  const active = filtered[cursor];

  const selectedSet = new Set(selection);

  return (
    <Box flexDirection="column" paddingX={1}>
      <Box>
        <Text color={theme.colors.primary} bold>
          记忆管理（{filtered.length}{kindFilter !== "all" ? `/${memories.length}` : ""}）
        </Text>
        {selection.length > 0 && (
          <Text color={theme.colors.accent} bold>
            {"  "}已选 {selection.length}
          </Text>
        )}
        {currentQuery && (
          <Text color={theme.colors.textDim}>
            {"  "}查询: <Text color={theme.colors.info}>{currentQuery}</Text>
          </Text>
        )}
        {kindFilter !== "all" && (
          <Text color={theme.colors.textDim}>
            {"  "}kind: <Text color={theme.colors.accent}>{kindFilter}</Text>
          </Text>
        )}
        {loading && (
          <Text color={theme.colors.warning}>{"  "}加载中…</Text>
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
            value={searchDraft}
            onChange={setSearchDraft}
            onSubmit={(q) => {
              setMode("list");
              setMemoryLoading(true);
              if (!q.trim()) {
                sendRequest({ kind: "list_memories", limit: 100 });
              } else {
                sendRequest({ kind: "search_memories", query: q, k: 50 });
              }
            }}
            placeholder="输入查询词，Enter 搜索 · Esc 取消"
            showCursor
          />
        </Box>
      )}

      <Box marginTop={1} flexDirection="column">
        {filtered.length === 0 ? (
          <Text color={theme.colors.textDim}>
            {loading ? "加载中…" : currentQuery ? "（无匹配）" : "暂无记忆。在对话中 agent 会自动抽取事实。"}
          </Text>
        ) : (
          visible.map((m, i) => (
            <MemoryRow
              key={m.id}
              mem={m}
              active={windowStart + i === cursor}
              pendingDelete={pendingDelete === m.id}
              selected={selectedSet.has(m.id)}
            />
          ))
        )}
        {filtered.length > listRows && (
          <Text color={theme.colors.textDim}>
            …显示 {windowStart + 1}-{Math.min(windowStart + listRows, filtered.length)} / {filtered.length}
          </Text>
        )}
      </Box>

      {expanded && active && (
        <Box
          marginTop={1}
          borderStyle="single"
          borderColor={theme.colors.border}
          paddingX={1}
          flexDirection="column"
          overflow="hidden"
        >
          <Text color={theme.colors.textDim}>
            #{active.id} · {active.kind}
            {active.source_session ? ` · from ${active.source_session.slice(0, 8)}` : ""}
            {" · "}hits={active.hits}
            {active.score ? ` · score=${active.score.toFixed(3)}` : ""}
          </Text>
          <Text color={theme.colors.text}>{active.text}</Text>
          {active.predicate && (
            <Text color={theme.colors.accent}>
              {active.subject} —{active.predicate}→ {active.object}
              {active.valid_until ? ` (过期 ${active.valid_until.slice(0, 10)})` : ""}
            </Text>
          )}
        </Box>
      )}

      <Box marginTop={1} flexDirection="column">
        {bulkPending > 0 ? (
          <Text color={theme.colors.warning} bold>
            ⟲ 正在批量删除…剩余 {bulkPending} 条
          </Text>
        ) : pendingBulk && selection.length > 0 ? (
          <Text color={theme.colors.error} bold>
            ⚠️ 按 Enter 确认删除已选 {selection.length} 条；按 ↑/↓ 或 A 取消
          </Text>
        ) : (
          <Text color={theme.colors.textDim}>
            ↑/↓ 选 · Enter 展开 · <Text color={theme.colors.accent}>Space 多选</Text> · a 全选 · A 清选 · d+Enter 删除 · / 搜索 · k 类型 · r 刷新 · Esc 关闭
          </Text>
        )}
      </Box>
    </Box>
  );
}

function MemoryRow(props: {
  mem: { id: number; kind: string; text: string; updated_at: string; hits: number; score?: number };
  active: boolean;
  pendingDelete: boolean;
  selected: boolean;
}): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const { mem, active, pendingDelete, selected } = props;

  // 选中框：[✓] 选中；[·] 光标位但未选；[ ] 既不选也不光标
  const checkbox = selected ? "[✓]" : active ? "[·]" : "[ ]";

  let marker = " ";
  if (pendingDelete) marker = "✗";
  else if (active) marker = "▶";

  const textColor = pendingDelete
    ? theme.colors.error
    : selected
      ? theme.colors.accent
      : active
        ? theme.colors.primary
        : theme.colors.text;
  const kindColor = {
    semantic: theme.colors.info,
    episodic: theme.colors.accent,
    procedural: theme.colors.warning,
    triple: theme.colors.success,
  }[mem.kind] ?? theme.colors.textDim;

  const oneLine = mem.text.replace(/\s+/g, " ").slice(0, 80);
  const day = mem.updated_at ? mem.updated_at.slice(0, 10) : "";

  return (
    <Box flexDirection="row">
      <Text color={pendingDelete ? theme.colors.error : theme.colors.primary}>
        {marker}{" "}
      </Text>
      <Text color={selected ? theme.colors.accent : theme.colors.textDim} bold={selected}>
        {checkbox}{" "}
      </Text>
      <Text color={theme.colors.textDim}>#{mem.id}</Text>
      <Text color={kindColor}>{" "}[{mem.kind}]</Text>
      <Text color={textColor} bold={active || selected}>{" "}{oneLine}</Text>
      <Text color={theme.colors.textDim}>
        {"  "}· {day}
        {mem.score ? ` · ${mem.score.toFixed(2)}` : ""}
      </Text>
      {pendingDelete && (
        <Text color={theme.colors.error}> (Enter 确认删除)</Text>
      )}
    </Box>
  );
}
