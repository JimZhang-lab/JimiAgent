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
  // 从 store 读反应式 dims，由 App.tsx 的 resize listener 更新
  const dims = useStore((s) => s.dims);
  // 安全边距 1 行：macOS IME 候选窗、Terminal.app 的 status line 都可能蚕食底部；
  // 留一行余量可避免总渲染高度贴着 stdout.rows 时触发终端原生 scroll。
  const rows = Math.max(dims.rows - 1, 10);
  const cols = Math.max(dims.cols, 20);
  const theme = resolveTheme(useStore((s) => s.theme));
  const connected = useStore((s) => s.connected);
  const bootError = useStore((s) => s.bootError);
  const pendingConfirm = useStore((s) => s.pendingConfirm);
  const paletteOpen = useStore((s) => s.paletteOpen);
  const historyOpen = useStore((s) => s.historyOpen);
  const memoryOpen = useStore((s) => s.memoryOpen);
  const activity = useStore((s) => s.activity);
  const streaming = useStore((s) => s.streaming);
  const promptDraft = useStore((s) => s.promptDraft);
  const { sendMessage, cancel, newSession, refreshSessions } = useAgent();

  // 动态 overlay 的占列预算，隔离在这里统一扣掉给 content
  const confirmRows = pendingConfirm ? 5 : 0;                    // border 2 + 标题 1 + 摘要 1 + 提示 1
  const activityRows = activity || streaming ? 3 : 0;            // border 2 + 内容 1
  const slashOpen = promptDraft.startsWith("/") && !promptDraft.includes(" ");
  const slashRows = slashOpen ? 9 : 0;                           // border 2 + 6 项 + hint 1
  const HEADER_ROWS = 2;
  const PROMPT_ROWS = 3;
  const STATUS_ROWS = 2;
  const contentRows = Math.max(
    rows - HEADER_ROWS - PROMPT_ROWS - STATUS_ROWS - confirmRows - activityRows - slashRows,
    3,
  );

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

  let contentArea: React.ReactElement;
  if (paletteOpen) {
    contentArea = (
      <CommandPalette
        onClose={onCloseOverlays}
        maxRows={contentRows}
        onExit={onExit}
        onOpenHistory={onOpenHistory}
        onOpenMemory={onOpenMemory}
      />
    );
  } else if (historyOpen) {
    contentArea = <HistoryPanel onClose={onCloseOverlays} maxRows={contentRows} />;
  } else if (memoryOpen) {
    contentArea = <MemoryPanel onClose={onCloseOverlays} maxRows={contentRows} />;
  } else if (connected) {
    contentArea = <Messages maxRows={contentRows - 1} />;
  } else {
    contentArea = <ConnectingIndicator />;
  }

  // 输入焦点和 SlashSuggestions 的激活条件保持与 PromptInput 一致
  const focus = useStore((s) => s.focus);
  const vim = useStore((s) => s.vim);
  const slashActive = focus === "prompt" && !pendingConfirm &&
    !paletteOpen && !historyOpen && !memoryOpen &&
    connected && !bootError && (!vim.enabled || vim.mode === "insert");

  return (
    <Box
      flexDirection="column"
      width={cols}
      height={rows}
      overflow="hidden"
    >
      <Header theme={theme} />
      <Box
        flexDirection="column"
        flexGrow={1}
        flexShrink={1}
        minHeight={0}
        overflow="hidden"
      >
        {contentArea}
      </Box>
      {pendingConfirm && <ConfirmBanner />}
      <ActivityBar />
      <SlashSuggestions isActive={slashActive} onDismiss={() => { /* 由 store.promptDraft 变更自动重置 */ }} />
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

function ConnectingIndicator(): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const bootError = useStore((s) => s.bootError);
  if (bootError) {
    return (
      <Box flexDirection="column">
        <Text color={theme.colors.error} bold>
          启动失败：
        </Text>
        <Text color={theme.colors.error}>{bootError}</Text>
        <Box marginTop={1}>
          <Text color={theme.colors.textDim}>
            日志见 ~/.jimiagent/tui-worker.log
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

