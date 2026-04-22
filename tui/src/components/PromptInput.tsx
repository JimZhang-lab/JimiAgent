import React from "react";
import { Box, Text, useInput } from "ink";
import { SmartTextInput } from "./SmartTextInput.js";
import Spinner from "ink-spinner";
import { useStore } from "../state/store.js";
import { resolveTheme } from "../themes/index.js";
import { useVim, type VimIntent } from "../global/Vim.js";

export interface PromptInputProps {
  /** 真正的提交 handler；外层决定走 agent 还是客户端命令。 */
  onSubmit(text: string): void;
  /** 是否处于 placeholder 模式（连接中/不可用）。 */
  disabled?: boolean;
  /** 取消当前生成。 */
  onCancel?: () => void;
}

/** 输入框本体；value 来自 store.promptDraft，方便 MainLayout 统一做布局预算。 */
export function PromptInput({
  onSubmit,
  disabled,
  onCancel,
}: PromptInputProps): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const streaming = useStore((s) => s.streaming);
  const activity = useStore((s) => s.activity);
  const focus = useStore((s) => s.focus);
  const pendingConfirm = useStore((s) => s.pendingConfirm);
  const vim = useStore((s) => s.vim);
  const value = useStore((s) => s.promptDraft);
  const setValue = useStore((s) => s.setPromptDraft);
  /** streaming 或 thinking/tool 都算忙碌，整段期间都显示 spinner。 */
  const busy =
    streaming ||
    activity?.kind === "thinking" ||
    activity?.kind === "tool";
  const [lastSubmitted, setLastSubmitted] = React.useState<string>("");
  const [history, setHistory] = React.useState<string[]>([]);
  const [historyIdx, setHistoryIdx] = React.useState<number | null>(null);

  const isActive = focus === "prompt" && !pendingConfirm && !disabled;
  // Vim 开启时只有 insert 模式接收输入
  const textInputActive =
    isActive && (!vim.enabled || vim.mode === "insert");

  // ↑/↓ 只在非 vim.normal 且非 slash 补全模式下切换历史
  useInput(
    (input, key) => {
      if (!isActive) return;
      // Vim normal 下 Enter：有 draft 就提交；空 draft 就切回 insert。
      if (vim.enabled && vim.mode === "normal" && key.return) {
        if (value.trim()) {
          handleSubmit(value);
        } else {
          // 空 draft 按 Enter 进入 insert
          useStore.getState().setVim({ mode: "insert" });
        }
        return;
      }
      if (vim.enabled && vim.mode === "normal") return;
      if (value.startsWith("/") && !value.includes(" ")) return;
      if (key.upArrow) {
        if (history.length === 0) return;
        const next = historyIdx === null ? history.length - 1 : Math.max(0, historyIdx - 1);
        setHistoryIdx(next);
        setValue(history[next] ?? "");
      } else if (key.downArrow) {
        if (historyIdx === null) return;
        const next = historyIdx + 1;
        if (next >= history.length) {
          setHistoryIdx(null);
          setValue("");
        } else {
          setHistoryIdx(next);
          setValue(history[next] ?? "");
        }
      } else if (input === "\u0003" /* Ctrl+C */ && busy) {
        // busy 状态下第一个 Ctrl+C 先取消 agent，双按退出交给 App.tsx 的空闲态。
        onCancel?.();
      }
    },
    { isActive },
  );

  // Vim 意图处理
  useVim(
    (intent: VimIntent) => {
      switch (intent.kind) {
        case "clear-buffer":
          setValue("");
          break;
        case "undo":
          setValue(lastSubmitted);
          break;
        case "enter-insert":
          break;
        case "session-prev":
        case "session-next":
        case "open-cmdline":
          break;
      }
    },
    { isActive },
  );

  const handleSubmit = (text: string) => {
    const t = text.trim();
    if (!t) return;
    // 若 SlashSuggestions 本帧已消费 Enter，会先把 draft 清空；这里顺手跳过重复提交。
    if (useStore.getState().promptDraft === "" && text.length > 0) return;
    setHistory((h) => [...h.slice(-50), text]);
    setHistoryIdx(null);
    setLastSubmitted(text);
    setValue("");
    onSubmit(text);
  };

  return (
    <Box
      paddingX={1}
      borderStyle="round"
      borderColor={isActive ? theme.colors.borderActive : theme.colors.border}
      flexDirection="row"
      flexShrink={0}
    >
      <Box marginRight={1}>
        {busy ? (
          <Text color={theme.colors.primary}>
            <Spinner type="dots" />
          </Text>
        ) : (
          <Text color={isActive ? theme.colors.primary : theme.colors.textDim} bold>
            &gt;
          </Text>
        )}
      </Box>
      <Box flexGrow={1}>
        {textInputActive ? (
          <SmartTextInput
            value={value}
            onChange={setValue}
            onSubmit={handleSubmit}
            placeholder={placeholderFor(busy, activity?.kind, streaming)}
            showCursor
          />
        ) : (
          <Text color={isActive ? theme.colors.text : theme.colors.textDim}>
            {pendingConfirm
              ? "等待确认 (y/n)…"
              : busy
                ? placeholderFor(true, activity?.kind, streaming)
                : vim.enabled && vim.mode === "normal"
                  ? value ? `${value}█` : "█  （NORMAL 模式，按 i 进入 INSERT）"
                  : disabled
                    ? "（未就绪）"
                    : `（按 Tab 或 i 回到输入）`}
          </Text>
        )}
      </Box>
    </Box>
  );
}

/** 根据当前状态挑一条合适的 placeholder。 */
function placeholderFor(
  busy: boolean,
  kind: string | undefined,
  streaming: boolean,
): string {
  if (!busy) return "输入消息，Enter 发送；/help 查看命令";
  if (streaming) return "生成中…Ctrl+C 取消";
  if (kind === "tool") return "调用工具中…Ctrl+C 取消";
  return "等待模型响应…Ctrl+C 取消";
}
