import React, { useEffect, useMemo, useState } from "react";
import { Box, Text, useInput } from "ink";
import Fuse from "fuse.js";
import { useStore } from "../state/store.js";
import { resolveTheme } from "../themes/index.js";
import { CLIENT_COMMANDS } from "../utils/slashCommands.js";
import { sendRequest, subscribeTransport } from "../state/wiring.js";

export interface SlashSuggestion {
  name: string;
  description: string;
  source: "client" | "server";
}

export interface SlashSuggestionsProps {
  /** 是否激活（拥有键盘）。 */
  isActive: boolean;
  /** 用户主动 Esc 关闭时触发。 */
  onDismiss(): void;
  /** Enter 时提交完整命令（例如 `/clear`）。 */
  onSelect(text: string): void;
  /** 最多显示多少行候选（默认 6）。 */
  maxItems?: number;
}

/**
 * 行内斜杠命令补全。
 *
 * 只读 store.promptDraft，匹配 client/server 命令，再把选中结果写回 store。
 */
export function SlashSuggestions({
  isActive,
  onDismiss,
  onSelect,
  maxItems = 6,
}: SlashSuggestionsProps): React.ReactElement | null {
  const theme = resolveTheme(useStore((s) => s.theme));
  const value = useStore((s) => s.promptDraft);
  const setValue = useStore((s) => s.setPromptDraft);
  const setSlashSuggestRows = useStore((s) => s.setSlashSuggestRows);
  const [cursor, setCursor] = useState(0);
  const [serverItems, setServerItems] = useState<SlashSuggestion[]>([]);
  const [dismissed, setDismissed] = useState(false);
  // 记住上次选中的命令名，下次空查询时优先把 cursor 放回去。
  const lastPickedName = React.useRef<string | null>(null);

  // 只在从“非 /”切到“/”时重置 dismissed。
  const prevWasSlash = React.useRef(false);
  useEffect(() => {
    const isSlash = value.startsWith("/");
    if (isSlash && !prevWasSlash.current) {
      setDismissed(false);
      // cursor 留给后面的 filtered effect 再决定
    }
    prevWasSlash.current = isSlash;
  }, [value]);

  // 组件生命周期内只订阅一次 commands 事件。
  useEffect(() => {
    const off = subscribeTransport((ev) => {
      if (ev.type === "commands") {
        setServerItems(
          ev.items.map((it) => ({
            name: it.name,
            description: it.description,
            source: "server" as const,
          })),
        );
      }
    });
    return off;
  }, []);

  useEffect(() => {
    if (!value.startsWith("/") || dismissed) return;
    if (serverItems.length === 0) {
      sendRequest({ kind: "list_commands" });
    }
  }, [value, dismissed, serverItems.length]);

  const all = useMemo<SlashSuggestion[]>(() => {
    const client: SlashSuggestion[] = CLIENT_COMMANDS.map((c) => ({
      name: c.name,
      description: c.description,
      source: "client",
    }));
    const seen = new Set(client.map((x) => x.name));
    const rest = serverItems.filter((x) => !seen.has(x.name));
    return [...client, ...rest];
  }, [serverItems]);

  const fuse = useMemo(
    () =>
      new Fuse(all, {
        keys: ["name", "description"],
        threshold: 0.45,
        ignoreLocation: true,
      }),
    [all],
  );

  const filtered = useMemo<SlashSuggestion[]>(() => {
    const q = value.replace(/^\//, "").trim();
    if (!q) return all;
    return fuse.search(q).map((r) => r.item);
  }, [value, fuse, all]);

  // filtered 收缩后，把 cursor clamp 到可见范围
  useEffect(() => {
    if (filtered.length === 0) {
      if (cursor !== 0) setCursor(0);
      return;
    }
    if (cursor >= filtered.length) {
      setCursor(filtered.length - 1);
    }
  }, [filtered.length, cursor]);

  // 刚打开且空查询时，优先回到上次选中过的命令。
  useEffect(() => {
    if (!value.startsWith("/")) return;
    const q = value.replace(/^\//, "").trim();
    if (q.length > 0) return; // 用户已经开始筛，不覆盖
    const name = lastPickedName.current;
    if (!name) {
      if (cursor !== 0) setCursor(0);
      return;
    }
    const idx = filtered.findIndex((it) => it.name === name);
    if (idx >= 0 && cursor !== idx) setCursor(idx);
    // filtered 依赖链里已包含 value，这里只依赖 filtered 即可
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filtered]);

  const shown = value.startsWith("/") && !value.includes(" ") && !dismissed && isActive;

  useInput(
    (input, key) => {
      if (!shown) return;
      // 无匹配时不拦 Enter / Tab / 方向键，让 PromptInput 继续处理。
      if (filtered.length === 0) {
        if (key.escape) {
          setDismissed(true);
          onDismiss();
        }
        return;
      }
      if (key.upArrow) {
        setCursor((c) => Math.max(0, c - 1));
      } else if (key.downArrow) {
        setCursor((c) => Math.min(filtered.length - 1, c + 1));
      } else if (key.tab) {
        const item = filtered[cursor];
        if (item) {
          lastPickedName.current = item.name;
          setValue(item.name + " ");
        }
      } else if (key.return) {
        // Enter 直接执行选中命令。
        const item = filtered[cursor];
        if (item) {
          lastPickedName.current = item.name;
          setValue("");
          onSelect(item.name);
        }
      } else if (key.escape) {
        setDismissed(true);
        onDismiss();
      }
      void input;
    },
    { isActive: shown },
  );

  // 计算实际渲染高度，提前写回 store 供 MainLayout 扣预算。
  const visibleCount =
    shown && filtered.length > 0
      ? Math.min(filtered.length, maxItems)
      : 0;
  const windowStart = Math.max(
    0,
    Math.min(
      cursor - Math.floor(maxItems / 2),
      filtered.length - maxItems,
    ),
  );
  const hasAbove = shown && filtered.length > 0 && windowStart > 0;
  const hasBelow =
    shown && filtered.length > 0 && windowStart + maxItems < filtered.length;
  const renderedRows =
    visibleCount === 0
      ? 0
      : 2 /*border*/ + visibleCount + (hasAbove ? 1 : 0) + (hasBelow ? 1 : 0) + 1 /*hint*/;
  // render 阶段调 setState 会被 React 警告，放到 effect 里。
  useEffect(() => {
    setSlashSuggestRows(renderedRows);
    return () => setSlashSuggestRows(0);
  }, [renderedRows, setSlashSuggestRows]);

  if (!shown) return null;
  if (filtered.length === 0) return null;

  const visible = filtered.slice(windowStart, windowStart + maxItems);

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor={theme.colors.borderActive}
      paddingX={1}
      flexShrink={0}
    >
      {hasAbove && (
        <Text color={theme.colors.textDim}>
          ↑ 还有 {windowStart} 条
        </Text>
      )}
      {visible.map((item, i) => (
        <SuggestionRow
          key={item.name}
          item={item}
          active={windowStart + i === cursor}
        />
      ))}
      {hasBelow && (
        <Text color={theme.colors.textDim}>
          ↓ 还有 {filtered.length - windowStart - maxItems} 条
        </Text>
      )}
      <Text color={theme.colors.textDim}>
        ↑/↓ 选择 · Tab 补全 · Enter 执行 · Esc 关闭
      </Text>
    </Box>
  );
}

function SuggestionRow({
  item,
  active,
}: {
  item: SlashSuggestion;
  active: boolean;
}): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  return (
    <Box flexDirection="row">
      <Text color={active ? theme.colors.primary : theme.colors.textDim}>
        {active ? "▶ " : "  "}
      </Text>
      <Text color={active ? theme.colors.text : theme.colors.textDim} bold={active}>
        {item.name}
      </Text>
      <Text color={theme.colors.textDim}>
        {"  "}
        {item.source === "server" ? "[server]" : "[ui]"}{" "}
        {item.description}
      </Text>
    </Box>
  );
}
