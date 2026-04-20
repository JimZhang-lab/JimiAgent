import React, { useEffect, useMemo, useState } from "react";
import { Box, Text, useInput } from "ink";
import TextInput from "ink-text-input";
import Fuse from "fuse.js";
import { useStore } from "../state/store.js";
import { resolveTheme } from "../themes/index.js";
import { CLIENT_COMMANDS } from "../utils/slashCommands.js";
import { sendRequest, subscribeTransport } from "../state/wiring.js";
import { useAgent } from "../hooks/useAgent.js";

export interface CommandPaletteProps {
  onClose(): void;
  maxRows: number;
  onExit(): void;
  onOpenHistory(): void;
  onOpenMemory(): void;
}

interface PaletteItem {
  name: string;
  description: string;
  source: "client" | "server";
}

/**
 * 命令面板：fuse.js 模糊匹配 + 上下导航 + Enter 执行。
 *
 * 内容来源：
 *   - 客户端命令（slashCommands.ts）
 *   - 服务端命令（由 worker 的 list_commands 返回）
 */
export function CommandPalette({
  onClose,
  maxRows,
  onExit,
  onOpenHistory,
  onOpenMemory,
}: CommandPaletteProps): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const { newSession, refreshSessions } = useAgent();
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const [serverItems, setServerItems] = useState<PaletteItem[]>([]);

  // 打开面板立刻拉一次服务端命令
  useEffect(() => {
    sendRequest({ kind: "list_commands" });
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

  const allItems = useMemo<PaletteItem[]>(() => {
    const clientItems: PaletteItem[] = CLIENT_COMMANDS.map((c) => ({
      name: c.name,
      description: c.description,
      source: "client",
    }));
    // 去重（服务端不覆盖客户端同名）
    const seen = new Set(clientItems.map((x) => x.name));
    const remote = serverItems.filter((x) => !seen.has(x.name));
    return [...clientItems, ...remote];
  }, [serverItems]);

  const fuse = useMemo(
    () =>
      new Fuse(allItems, {
        keys: ["name", "description"],
        threshold: 0.4,
        ignoreLocation: true,
      }),
    [allItems],
  );

  const filtered = useMemo<PaletteItem[]>(() => {
    if (!query.trim()) return allItems;
    return fuse.search(query).map((r) => r.item);
  }, [query, fuse, allItems]);

  // query 变时，cursor 回零
  useEffect(() => {
    setCursor(0);
  }, [query]);

  useInput(
    (input, key) => {
      if (key.upArrow) {
        setCursor((c) => Math.max(0, c - 1));
      } else if (key.downArrow) {
        setCursor((c) => Math.min(filtered.length - 1, c + 1));
      } else if (key.tab) {
        // 快速全选下一项（wrap）
        setCursor((c) => (c + 1) % Math.max(1, filtered.length));
      } else if (key.return) {
        const item = filtered[cursor];
        if (!item) return;
        runItem(item);
      } else if (input === "\u0003" /* Ctrl+C */) {
        // 由 App 层统一处理；此处忽略
      }
    },
    { isActive: true },
  );

  function runItem(item: PaletteItem): void {
    onClose();
    if (item.source === "client") {
      const c = CLIENT_COMMANDS.find((x) => x.name === item.name);
      if (!c) return;
      c.run("", {
        exit: onExit,
        newSession,
        refreshSessions,
        openPalette: () => {},
        openHistory: onOpenHistory,
        openMemory: onOpenMemory,
      });
    } else {
      // 服务端命令：当作 chat 消息发出（agent 会识别 /xxx）
      const sid = useStore.getState().currentSessionId;
      if (!sid) return;
      sendRequest({ kind: "chat", session_id: sid, message: item.name });
    }
  }

  const headerRows = 2;
  const listRows = Math.max(1, maxRows - headerRows - 2);
  const visible = filtered.slice(0, listRows);

  return (
    <Box flexDirection="column" paddingX={1}>
      <Box
        borderStyle="round"
        borderColor={theme.colors.borderActive}
        paddingX={1}
      >
        <Text color={theme.colors.primary} bold>
          ❯{" "}
        </Text>
        <TextInput
          value={query}
          onChange={setQuery}
          placeholder="输入关键字过滤命令…（Esc 关闭）"
          showCursor
        />
      </Box>
      <Box marginTop={1} flexDirection="column">
        {visible.length === 0 ? (
          <Text color={theme.colors.textDim}>无匹配命令</Text>
        ) : (
          visible.map((it, i) => (
            <CommandRow
              key={it.name}
              item={it}
              active={i === cursor}
            />
          ))
        )}
        {filtered.length > listRows && (
          <Text color={theme.colors.textDim}>
            …还有 {filtered.length - listRows} 条（向下滚动）
          </Text>
        )}
      </Box>
      <Box marginTop={1}>
        <Text color={theme.colors.textDim}>
          ↑/↓ 选择 · Enter 执行 · Esc 关闭 · Tab 下一项
        </Text>
      </Box>
    </Box>
  );
}

function CommandRow({
  item,
  active,
}: {
  item: PaletteItem;
  active: boolean;
}): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const nameColor = active ? theme.colors.primary : theme.colors.text;
  const descColor = active ? theme.colors.text : theme.colors.textDim;
  const marker = active ? "▶" : " ";
  return (
    <Box flexDirection="row">
      <Text color={theme.colors.primary}>{marker} </Text>
      <Text color={nameColor} bold={active}>
        {item.name}
      </Text>
      <Text color={theme.colors.textDim}>
        {"  "}
        {item.source === "server" ? "[server]" : "[ui]"}{" "}
      </Text>
      <Text color={descColor}>{item.description}</Text>
    </Box>
  );
}
