import React from "react";
import { Box, Text, useInput } from "ink";
import { SmartTextInput } from "./SmartTextInput.js";
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
  const activity = useStore((s) => s.activity);
  const focus = useStore((s) => s.focus);
  const pendingConfirm = useStore((s) => s.pendingConfirm);
  const vim = useStore((s) => s.vim);
  const value = useStore((s) => s.promptDraft);
  const setValue = useStore((s) => s.setPromptDraft);
  /**
   * "忙碌"广义状态：streaming=true（已有 chunk 到达）或 activity=thinking/tool
   * （已发送但 chunk 还没来 / 正在调工具）。
   *
   * 首次 chunk 到达之前的 TTFB 时段原先只显示静态 `>`，体感为"没反应"，
   * 这里把 spinner 的显示条件扩大到整段忙碌期。
   */
  const busy =
    streaming ||
    activity?.kind === "thinking" ||
    activity?.kind === "tool";
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
      // Vim normal 下 Enter：TextInput 未 active，onSubmit 永远不触发。
      // 若 draft 非空就当作"用户确认提交"走同一 handleSubmit；否则切回 insert。
      if (vim.enabled && vim.mode === "normal" && key.return) {
        if (value.trim()) {
          handleSubmit(value);
        } else {
          // 空 draft 按 Enter 进入 insert，等同于 `i` / `a`
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
        // 第一个 Ctrl+C 取消当前在跑的 agent，不等 chunk。
        // 上层 App.tsx 的 双按 Ctrl+C 退出 仅在 !busy 时生效。
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
    // 若 SlashSuggestions 本帧已消费 Enter（它会 setValue("") 清 draft），
    // store.promptDraft 会先于 TextInput.onSubmit 被更新为 ""。
    // 此时跳过重复提交。反之（SlashSuggestions 未消费——无匹配 / dismissed），
    // 即使 text 是 "/abc"，也放行让外层 handleSubmit 去 tryRunClientCommand
    // 或发给 agent，避免"Enter 黑洞"（旧版会直接静默 return）。
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

/**
 * 根据当前忙碌状态选一条合适的 placeholder。
 *
 * 优先级：
 *   1. streaming=true → 模型正在吐 token。
 *   2. activity=tool   → 正在调用工具。
 *   3. activity=thinking → 已发送但还没 chunk 返回（TTFB）。
 *   4. 空闲→常规提示。
 */
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
