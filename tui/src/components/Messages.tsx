import React from "react";
import { Box, Static, Text } from "ink";
import { useStore } from "../state/store.js";
import { resolveTheme } from "../themes/index.js";
import { MessageItem } from "./MessageItem.js";

/**
 * 消息列表容器 —— **print-above** 架构。
 *
 * 目的：不再在 TUI 内部做虚拟滚动窗口，把历史消息一条条"打印"到终端 stdout，
 * 让终端自身的 scrollback 成为事实上的历史视图（用户鼠标滚轮 / PgUp 即看完整历史）。
 *
 * 实现：
 *   - 稳定消息（非流式）走 `<Static>`：Ink 只在追加时渲染新增项，打印后脱离重绘区，
 *     留在 terminal scrollback 里不会被覆盖
 *   - 正在流式中的消息走普通 `<Box>`：每个 chunk 触发 rerender，位置在固定动态区
 *   - 一旦流式完成（streaming=false），该消息从 pending 区"升华"到 Static items，
 *     Ink diff 发现 items 纯追加，打印到 scrollback，完成一次状态迁移
 *
 * **前置条件**：Ink 5 **不**进入 alt-screen（默认即不进）。若将来引入
 * `fullscreen` 模式，这套架构会失效。
 *
 * 空消息态渲染 Splash（连接状态 + 帮助提示），等待第一条消息出现后
 * Splash 被 Ink 擦除并被 Static 接管。
 */
export function Messages(): React.ReactElement {
  const messages = useStore((s) => s.messages);
  const connected = useStore((s) => s.connected);

  if (messages.length === 0) {
    return <SplashScreen connected={connected} />;
  }

  // 把 messages 切成 [stable 前缀, pending 尾部]：
  //   - pending 尾部：从末尾向前找连续的 streaming 消息
  //   - stable：剩余前缀，作为 Static items
  // 保证 Static items 永远是"上次的 items + 尾部新增"的纯追加序列，
  // Ink Static 才能识别并只打印新项。
  let stableEnd = messages.length;
  while (stableEnd > 0 && messages[stableEnd - 1]?.streaming) stableEnd--;
  const stable = messages.slice(0, stableEnd);
  const pending = messages.slice(stableEnd);

  return (
    <Box flexDirection="column" paddingX={1}>
      {stable.length > 0 && (
        <Static items={stable}>
          {(m) => <MessageItem key={m.id} message={m} />}
        </Static>
      )}
      {pending.map((m) => (
        <MessageItem key={m.id} message={m} />
      ))}
    </Box>
  );
}

/**
 * 空消息态的"开场画面"。
 *
 * 展示内容：
 *   - ASCII logo / 欢迎语
 *   - 当前会话 / 工作目录 / 模型等上下文，帮助用户确认"自己在哪"
 *   - 常用快捷键三条，避免第一次进来完全不知道怎么开始
 *
 * print-above 架构下：Splash 位于动态区，用户发第一条消息时会被 Ink 自动擦除，
 * 不会滞留在 scrollback 里造成噪声。
 */
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
