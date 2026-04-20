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
  /** 是否激活（拥有键盘）。外层决定何时 true。 */
  isActive: boolean;
  /** 用户主动 Esc 关闭时触发；用于外层决定下次是否重新展示。 */
  onDismiss(): void;
  /** 最多显示多少行候选（默认 6）。 */
  maxItems?: number;
}

/**
 * 斜杠命令行内补全：
 *   - 从 `store.promptDraft` 读取当前输入；draft 以 `/` 开头且不含空格才展示
 *   - 服务端命令通过 `list_commands` 请求拉取并缓存
 *   - ↑/↓ 选择；Tab 把命令名（带空格）写回 store；Esc 关闭本次补全
 *   - 列表长度 > maxItems 时，窗口以 cursor 为中心滑动
 *
 * 和 PromptInput 完全解耦：它只读 store.promptDraft，写回用 setPromptDraft。
 * 这样可以从 MainLayout 顶层渲染，不推挤其他 flex 块。
 */
export function SlashSuggestions({
  isActive,
  onDismiss,
  maxItems = 6,
}: SlashSuggestionsProps): React.ReactElement | null {
  const theme = resolveTheme(useStore((s) => s.theme));
  const value = useStore((s) => s.promptDraft);
  const setValue = useStore((s) => s.setPromptDraft);
  const [cursor, setCursor] = useState(0);
  const [serverItems, setServerItems] = useState<SlashSuggestion[]>([]);
  const [dismissed, setDismissed] = useState(false);

  // 只在从"非 /"切到"/"时重置 dismissed；typing 过程中不乱动
  const prevWasSlash = React.useRef(false);
  useEffect(() => {
    const isSlash = value.startsWith("/");
    if (isSlash && !prevWasSlash.current) {
      setDismissed(false);
      setCursor(0);
    }
    prevWasSlash.current = isSlash;
  }, [value]);

  // 首次进入 `/` 模式拉一次服务端命令
  useEffect(() => {
    if (!value.startsWith("/") || dismissed) return;
    if (serverItems.length === 0) {
      sendRequest({ kind: "list_commands" });
    }
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

  // cursor 越界 clamp：filtered 收缩后，把 cursor 拉回最后一项
  useEffect(() => {
    if (filtered.length === 0) {
      if (cursor !== 0) setCursor(0);
      return;
    }
    if (cursor >= filtered.length) {
      setCursor(filtered.length - 1);
    }
  }, [filtered.length, cursor]);

  const shown = value.startsWith("/") && !value.includes(" ") && !dismissed && isActive;

  useInput(
    (input, key) => {
      if (!shown) return;
      if (filtered.length === 0) return;
      if (key.upArrow) {
        setCursor((c) => Math.max(0, c - 1));
      } else if (key.downArrow) {
        setCursor((c) => Math.min(filtered.length - 1, c + 1));
      } else if (key.tab) {
        const item = filtered[cursor];
        if (item) setValue(item.name + " ");
      } else if (key.escape) {
        setDismissed(true);
        onDismiss();
      }
      void input;
    },
    { isActive: shown },
  );

  if (!shown) return null;
  if (filtered.length === 0) return null;

  // 以 cursor 为中心滑动窗口
  const windowStart = Math.max(
    0,
    Math.min(
      cursor - Math.floor(maxItems / 2),
      filtered.length - maxItems,
    ),
  );
  const visible = filtered.slice(windowStart, windowStart + maxItems);
  const hasAbove = windowStart > 0;
  const hasBelow = windowStart + maxItems < filtered.length;

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
