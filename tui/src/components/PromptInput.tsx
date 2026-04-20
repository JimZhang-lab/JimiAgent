import React from "react";
import { Box, Text, useInput } from "ink";
import TextInput from "ink-text-input";
import Spinner from "ink-spinner";
import { useStore } from "../state/store.js";
import { resolveTheme } from "../themes/index.js";
import { useVim, type VimIntent } from "../global/Vim.js";

export interface PromptInputProps {
  /** 真正的提交 handler；外层决定是走 agent 还是客户端命令。 */
  onSubmit(text: string): void;
  /** 是否处于 placeholder 模式（连接中/不可用）。 */
  disabled?: boolean;
  /** 取消当前生成。 */
  onCancel?: () => void;
}

/**
 * 输入框：value 来自 store.promptDraft（便于 MainLayout 做 layout 预算）。
 * SlashSuggestions 由 MainLayout 顶层渲染，不再在此处拼接。
 */
export function PromptInput({
  onSubmit,
  disabled,
  onCancel,
}: PromptInputProps): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const streaming = useStore((s) => s.streaming);
  const focus = useStore((s) => s.focus);
  const pendingConfirm = useStore((s) => s.pendingConfirm);
  const vim = useStore((s) => s.vim);
  const value = useStore((s) => s.promptDraft);
  const setValue = useStore((s) => s.setPromptDraft);
  const [lastSubmitted, setLastSubmitted] = React.useState<string>("");
  const [history, setHistory] = React.useState<string[]>([]);
  const [historyIdx, setHistoryIdx] = React.useState<number | null>(null);

  const isActive = focus === "prompt" && !pendingConfirm && !disabled;
  // Vim 启用时，insert 模式接受输入；normal 模式 TextInput 失活
  const textInputActive =
    isActive && (!vim.enabled || vim.mode === "insert");

  // ↑/↓ 历史切换（仅非 vim.normal 模式激活；slash 模式让给补全面板）
  useInput(
    (input, key) => {
      if (!isActive) return;
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
      } else if (input === "\u0003" /* Ctrl+C */ && streaming) {
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
        {streaming ? (
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
          <TextInput
            value={value}
            onChange={setValue}
            onSubmit={handleSubmit}
            placeholder={
              streaming ? "生成中…Ctrl+C 取消" : "输入消息，Enter 发送；/help 查看命令"
            }
            showCursor
          />
        ) : (
          <Text color={isActive ? theme.colors.text : theme.colors.textDim}>
            {pendingConfirm
              ? "等待确认 (y/n)…"
              : streaming
                ? "生成中…Ctrl+C 取消"
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
