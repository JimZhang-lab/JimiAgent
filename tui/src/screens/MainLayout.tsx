import React from "react";
import { Box, Text } from "ink";
import { useStore } from "../state/store.js";
import { resolveTheme } from "../themes/index.js";
import { StatusBar } from "./StatusBar.js";
import { CommandPalette } from "./CommandPalette.js";
import { HistoryPanel } from "./HistoryPanel.js";
import { MemoryPanel } from "./MemoryPanel.js";
import { Messages } from "../components/Messages.js";
import { PromptInput } from "../components/PromptInput.js";
import { ConfirmBanner } from "../components/ConfirmBanner.js";
import { ActivityBar } from "../components/ActivityBar.js";
import { SlashSuggestions } from "../components/SlashSuggestions.js";
import { useAgent } from "../hooks/useAgent.js";
import { tryRunClientCommand } from "../utils/slashCommands.js";

export interface MainLayoutProps {
  /** 由 App 提供的全局退出；供客户端命令 /quit 使用。 */
  onExit(): void;
  /** 打开命令面板。 */
  onOpenPalette(): void;
  /** 打开历史面板。 */
  onOpenHistory(): void;
  /** 打开记忆面板。 */
  onOpenMemory(): void;
  /** 关闭任意弹层。 */
  onCloseOverlays(): void;
}

/**
 * 主布局：header + messages 区 + ConfirmBanner + prompt 区 + statusbar。
 */
export function MainLayout({
  onExit,
  onOpenPalette,
  onOpenHistory,
  onOpenMemory,
  onCloseOverlays,
}: MainLayoutProps): React.ReactElement {
  // print-above 架构下，外层容器不再硬设 height/overflow：
  //   - 消息区通过 <Static> 把每条消息逐条打印到 terminal stdout，
  //     打印后内容进入终端 scrollback，用户靠原生终端滚动就能回看完整历史
  //   - 只有动态 UI（Header / ActivityBar / SlashSuggestions / PromptInput /
  //     StatusBar / ConfirmBanner）保留在屏幕底部，被 Ink 按需重绘
  // 不再计算 contentRows / confirmRows / activityRows 预算，交给 Ink 自然 layout。
  const theme = resolveTheme(useStore((s) => s.theme));
  const connected = useStore((s) => s.connected);
  const bootError = useStore((s) => s.bootError);
  const pendingConfirm = useStore((s) => s.pendingConfirm);
  const paletteOpen = useStore((s) => s.paletteOpen);
  const historyOpen = useStore((s) => s.historyOpen);
  const memoryOpen = useStore((s) => s.memoryOpen);
  const { sendMessage, cancel, newSession, refreshSessions } = useAgent();

  const handleSubmit = (text: string) => {
    const handled = tryRunClientCommand(text, {
      exit: onExit,
      newSession,
      refreshSessions,
      openPalette: onOpenPalette,
      openHistory: onOpenHistory,
      openMemory: onOpenMemory,
    });
    if (!handled) sendMessage(text);
  };

  // Overlay 类面板仍然在"屏幕上"弹出；此时隐藏 print-above 的消息区，
  // 避免流式消息被面板遮挡产生错位。关闭面板后 Messages 重新挂载，
  // Ink <Static> 会把已打印的 stable items 视为"已发射"不再重打，新的 pending
  // 尾部继续流式。
  let overlay: React.ReactElement | null = null;
  if (paletteOpen) {
    overlay = (
      <CommandPalette
        onClose={onCloseOverlays}
        onExit={onExit}
        onOpenHistory={onOpenHistory}
        onOpenMemory={onOpenMemory}
      />
    );
  } else if (historyOpen) {
    overlay = <HistoryPanel onClose={onCloseOverlays} />;
  } else if (memoryOpen) {
    overlay = <MemoryPanel onClose={onCloseOverlays} />;
  }

  // 输入焦点和 SlashSuggestions 的激活条件保持与 PromptInput 一致
  const focus = useStore((s) => s.focus);
  const vim = useStore((s) => s.vim);
  const slashActive = focus === "prompt" && !pendingConfirm &&
    !paletteOpen && !historyOpen && !memoryOpen &&
    connected && !bootError && (!vim.enabled || vim.mode === "insert");

  return (
    <Box flexDirection="column">
      <Header theme={theme} />
      {overlay ? (
        overlay
      ) : connected ? (
        <Messages />
      ) : (
        <ConnectingIndicator />
      )}
      {pendingConfirm && <ConfirmBanner />}
      <ActivityBar />
      <SlashSuggestions
        isActive={slashActive}
        onDismiss={() => {
          /* 由 store.promptDraft 变更自动重置 */
        }}
        onSelect={handleSubmit}
      />
      <PromptInput
        onSubmit={handleSubmit}
        onCancel={cancel}
        disabled={!connected || !!bootError}
      />
      <StatusBar />
    </Box>
  );
}

function Header({
  theme,
}: {
  theme: ReturnType<typeof resolveTheme>;
}): React.ReactElement {
  const agentInfo = useStore((s) => s.agentInfo);
  const cwd = useStore((s) => s.workspaceCwd);
  const short = cwd ? shortenPath(cwd) : "";
  return (
    <Box
      paddingX={1}
      borderStyle="single"
      borderColor={theme.colors.border}
      borderTop={false}
      borderLeft={false}
      borderRight={false}
      borderBottom={true}
      justifyContent="space-between"
    >
      <Box>
        <Text color={theme.colors.primary} bold>
          JimiAgent
        </Text>
        <Text color={theme.colors.textDim}> · React TUI</Text>
        {short && (
          <>
            <Text color={theme.colors.textDim}>  · </Text>
            <Text color={theme.colors.accent}>{short}</Text>
          </>
        )}
      </Box>
      <Box>
        {agentInfo && (
          <Text color={theme.colors.textDim}>
            {agentInfo.model} · {agentInfo.skills.length} skills
          </Text>
        )}
      </Box>
    </Box>
  );
}

/**
 * 把绝对路径简写：
 *   - /Users/<user>/... → ~/...
 *   - 其他保留原样
 *   - 超过 48 字符时中间用 `…/` 截断，保留 tail
 */
function shortenPath(p: string): string {
  const home = process.env.HOME;
  let s = p;
  if (home && s.startsWith(home)) s = "~" + s.slice(home.length);
  if (s.length <= 48) return s;
  const parts = s.split("/");
  // 保留首段（~ 或 ""）和末 2 段
  const head = parts[0] ?? "";
  const tail = parts.slice(-2).join("/");
  return `${head}/…/${tail}`;
}

function ConnectingIndicator(): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const bootError = useStore((s) => s.bootError);
  if (bootError) {
    return (
      <Box flexDirection="column" paddingX={1}>
        <Text color={theme.colors.error} bold>
          启动失败：
        </Text>
        <Text color={theme.colors.error}>{bootError}</Text>
        <Box marginTop={1} flexDirection="column">
          <Text color={theme.colors.textDim}>
            · 日志：<Text color={theme.colors.info}>~/.jimiagent/tui-worker.log</Text>
          </Text>
          <Text color={theme.colors.textDim}>
            · 重试：按 <Text color={theme.colors.warning} bold>Ctrl+C</Text> 两次退出后重跑
            {" "}<Text color={theme.colors.accent}>jimi chat</Text>
          </Text>
          <Text color={theme.colors.textDim}>
            · 如果是 Python 环境问题，先 <Text color={theme.colors.info}>conda activate jimiAgent312</Text>
          </Text>
        </Box>
      </Box>
    );
  }
  return (
    <Box>
      <Text color={theme.colors.warning}>正在连接 tui_worker…</Text>
    </Box>
  );
}

