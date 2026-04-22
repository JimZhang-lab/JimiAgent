import React from "react";
import { Box, Text } from "ink";
import { useStore } from "../state/store.js";
import { resolveTheme } from "../themes/index.js";
import { MessageItem } from "./MessageItem.js";

/**
 * print-above 架构里的动态层。
 *
 * 历史消息交给 `HistoryStatic`，这里只渲染 Splash 和流式中的尾部消息。
 */
export function Messages(): React.ReactElement {
  const messages = useStore((s) => s.messages);
  const connected = useStore((s) => s.connected);

  // 从尾部向前找连续的 streaming 消息，作为 pending 区。
  let stableEnd = messages.length;
  while (stableEnd > 0 && messages[stableEnd - 1]?.streaming) stableEnd--;
  const pending = messages.slice(stableEnd);

  if (messages.length === 0) {
    return <SplashScreen connected={connected} />;
  }

  return (
    <Box flexDirection="column" paddingX={1}>
      {pending.map((m) => (
        <MessageItem key={m.id} message={m} />
      ))}
    </Box>
  );
}

/** 空消息态的开场画面。 */
function SplashScreen({
  connected,
}: {
  connected: boolean;
}): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const cwd = useStore((s) => s.workspaceCwd);
  const agentInfo = useStore((s) => s.agentInfo);
  const session = useStore((s) => {
    const id = s.currentSessionId;
    return id ? s.sessions.find((x) => x.id === id) ?? null : null;
  });

  if (!connected) {
    return (
      <Box flexDirection="column" paddingX={1}>
        <Text color={theme.colors.warning}>正在连接 tui_worker…</Text>
      </Box>
    );
  }

  const shortCwd = cwd ? shortenHome(cwd) : "";

  return (
    <Box flexDirection="column" paddingX={1}>
      <Text color={theme.colors.primary} bold>
        JimiAgent
      </Text>
      <Text color={theme.colors.textDim}>
        React TUI · 实时记忆 · 任务代理
      </Text>
      <Box marginTop={1} flexDirection="column">
        {session && (
          <Text color={theme.colors.text}>
            <Text color={theme.colors.textDim}>会话 </Text>
            <Text color={theme.colors.accent}>{session.title}</Text>
            <Text color={theme.colors.textDim}> [{session.id}]</Text>
          </Text>
        )}
        {shortCwd && (
          <Text color={theme.colors.text}>
            <Text color={theme.colors.textDim}>目录 </Text>
            <Text color={theme.colors.info}>{shortCwd}</Text>
          </Text>
        )}
        {agentInfo && (
          <Text color={theme.colors.text}>
            <Text color={theme.colors.textDim}>模型 </Text>
            <Text color={theme.colors.info}>{agentInfo.model}</Text>
            <Text color={theme.colors.textDim}>
              {" "}· {agentInfo.skills.length} skills · memory={agentInfo.memory}
            </Text>
          </Text>
        )}
      </Box>
      <Box marginTop={1} flexDirection="column">
        <Text color={theme.colors.textDim}>快捷键：</Text>
        <Text color={theme.colors.textDim}>
          · <Text color={theme.colors.accent}>Ctrl+P</Text> 命令面板
          {" · "}<Text color={theme.colors.accent}>Ctrl+S</Text> 会话列表
          {" · "}<Text color={theme.colors.accent}>Ctrl+M</Text> 记忆管理
        </Text>
        <Text color={theme.colors.textDim}>
          · 历史回顾：直接在**终端原生**向上滚（鼠标滚轮 / PgUp / Cmd+↑）
        </Text>
        <Text color={theme.colors.textDim}>
          · 输入 <Text color={theme.colors.accent}>/help</Text> 看更多，
          {" "}<Text color={theme.colors.accent}>/quit</Text> 退出
        </Text>
      </Box>
    </Box>
  );
}

function shortenHome(p: string): string {
  const home = process.env.HOME;
  if (home && p.startsWith(home)) return "~" + p.slice(home.length);
  return p;
}
