import React, { useEffect, useState } from "react";
import { Box, Text, useInput } from "ink";
import TextInput from "ink-text-input";
import { useStore } from "../state/store.js";
import { resolveTheme } from "../themes/index.js";
import { sendRequest } from "../state/wiring.js";

export interface MemoryPanelProps {
  onClose(): void;
  maxRows: number;
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
  maxRows,
}: MemoryPanelProps): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const memories = useStore((s) => s.memories);
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
      if (key.upArrow) {
        setCursor((c) => Math.max(0, c - 1));
        resetDeletePending();
        setExpanded(false);
      } else if (key.downArrow) {
        setCursor((c) => Math.min(Math.max(0, filtered.length - 1), c + 1));
        resetDeletePending();
        setExpanded(false);
      } else if (key.return) {
        // 批量删除确认优先
        if (pendingBulk && selection.length > 0) {
          for (const id of selection) {
            sendRequest({ kind: "delete_memory", memory_id: id });
          }
          setPendingBulk(false);
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
          <TextInput
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
        {pendingBulk && selection.length > 0 ? (
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
