import React from "react";
import { useInput } from "ink";
import { useStore } from "../state/store.js";

/**
 * 极简 Vim 状态机。
 *
 * 只维护模式和意图，真正的 buffer 改动仍由 PromptInput 执行。
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

      // Normal -> Insert
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
        // 其他键会清 pending
        pendingRef.current = "";
        return;
      }

      // Insert -> Normal
      if (vim.mode === "insert" && key.escape) {
        setVim({ mode: "normal" });
        return;
      }
    },
    { isActive: active },
  );
}
