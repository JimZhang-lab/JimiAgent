import React from "react";
import { useInput } from "ink";
import { useStore } from "../state/store.js";

/**
 * Vim 状态机（最小实现）。
 *
 * 设计目标：让 Vim 用户有 "modal" 体感，但不强求精确语义。
 *
 * Normal 模式支持：
 *   - **i** 进入 insert（光标在原位）
 *   - **a** 进入 insert（等同 i，不模拟 append）
 *   - **A** 进入 insert（同上）
 *   - **o** 进入 insert 并清空 buffer（相当于换行输入新内容）
 *   - **x** 清空 buffer
 *   - **dd** 清空 buffer（两次 d 触发）
 *   - **u** 撤销（最多 1 步，回到最近一次 submit 前的草稿）
 *   - **:** 进入 command-line（未实现细节，只显示提示）
 *   - **j/k** 会话历史上下切换（j 下一条会话、k 上一条；仅切换当前高亮，不 submit）
 *
 * Insert 模式：正常打字；**Esc** 回到 Normal。
 *
 * 实现：此 hook 只维护模式/意图；真正的 buffer mutation 由 PromptInput 在
 * `onVimIntent` 回调里执行。这样 VIM 逻辑和 TextInput UI 解耦。
 */
export interface VimIntent {
  kind:
    | "enter-insert"
    | "clear-buffer"
    | "undo"
    | "session-prev"
    | "session-next"
    | "open-cmdline";
}

export function useVim(
  onIntent: (intent: VimIntent) => void,
  opts?: { isActive?: boolean },
): void {
  const vim = useStore((s) => s.vim);
  const setVim = useStore((s) => s.setVim);
  const pendingRef = React.useRef<string>("");

  const active = !!vim.enabled && (opts?.isActive ?? true);

  useInput(
    (input, key) => {
      if (!active) return;

      // Normal → Insert 转换
      if (vim.mode === "normal") {
        if (key.escape) {
          // 重置 pending
          pendingRef.current = "";
          return;
        }
        if (input === "i" || input === "a" || input === "A") {
          setVim({ mode: "insert" });
          onIntent({ kind: "enter-insert" });
          pendingRef.current = "";
          return;
        }
        if (input === "o") {
          setVim({ mode: "insert" });
          onIntent({ kind: "clear-buffer" });
          onIntent({ kind: "enter-insert" });
          pendingRef.current = "";
          return;
        }
        if (input === "x") {
          onIntent({ kind: "clear-buffer" });
          pendingRef.current = "";
          return;
        }
        if (input === "d") {
          if (pendingRef.current === "d") {
            onIntent({ kind: "clear-buffer" });
            pendingRef.current = "";
          } else {
            pendingRef.current = "d";
          }
          return;
        }
        if (input === "u") {
          onIntent({ kind: "undo" });
          pendingRef.current = "";
          return;
        }
        if (input === "j") {
          onIntent({ kind: "session-next" });
          pendingRef.current = "";
          return;
        }
        if (input === "k") {
          onIntent({ kind: "session-prev" });
          pendingRef.current = "";
          return;
        }
        if (input === ":") {
          onIntent({ kind: "open-cmdline" });
          pendingRef.current = "";
          return;
        }
        // 任意其他键：清 pending
        pendingRef.current = "";
        return;
      }

      // Insert → Normal 转换
      if (vim.mode === "insert" && key.escape) {
        setVim({ mode: "normal" });
        return;
      }
    },
    { isActive: active },
  );
}
