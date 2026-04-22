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
  /**
   * Enter 时提交完整命令（例如 `/clear`）。
   * 由 MainLayout 传入，走和普通消息一致的 submit 链路
   * （`tryRunClientCommand` → client/server 命令或 agent）。
   */
  onSelect(text: string): void;
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
  // 记忆上次用户选中的命令名；再次打开 / 且没有更具体的匹配时，把 cursor 恢复到它。
  // 不跨 App 生命周期持久化（没必要，ref 足够）。
  const lastPickedName = React.useRef<string | null>(null);

  // 只在从"非 /"切到"/"时重置 dismissed；typing 过程中不乱动
  const prevWasSlash = React.useRef(false);
  useEffect(() => {
    const isSlash = value.startsWith("/");
    if (isSlash && !prevWasSlash.current) {
      setDismissed(false);
      // 不无条件清 cursor：先不动，等 filtered 算出来后在另一 effect 里定位到 lastPicked
    }
    prevWasSlash.current = isSlash;
  }, [value]);

  // 订阅 transport commands 事件：组件生命周期内只挂一次，避免重复订阅。
  // 首次进入 `/` 模式时（serverItems 为空）触发 list_commands 请求。
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

  // 当面板刚打开且查询为空时，把 cursor 定位到上次用户选择过的命令（如果仍在列表里）。
  // 仅在"从非 / 切到 /"时触发一次，避免输入过程被反复重置。
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
    // filtered 与 value 都在依赖里，但 filtered 依赖链里已经包含 value，所以只需依赖 filtered
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filtered]);

  const shown = value.startsWith("/") && !value.includes(" ") && !dismissed && isActive;

  useInput(
    (input, key) => {
      if (!shown) return;
      // 无匹配命令时，不消费 Enter / Tab / 方向键，交给 PromptInput 的 TextInput
      // 正常处理（用户可能想把 `/abc` 这种未知命令当作普通消息发出去）。
      // Esc 仍然可以 dismiss 面板。
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
        // Enter：直接执行选中命令（不带参数）。
        // 同一事件里 TextInput 也会触发 onSubmit，但 PromptInput 已约定对
        // slash-only draft 走 no-op，所以这里独占 submit 链路。
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
  //   - 不显示时 rows = 0，Messages 区可占满
  //   - 显示时 = border 2 + 可能的上/下提示 + items + hint 1
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
