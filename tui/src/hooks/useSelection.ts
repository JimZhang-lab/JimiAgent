import { useInput } from "ink";
import clipboardy from "clipboardy";
import { useStore, nextMessageId, type Message } from "../state/store.js";

/**
 * 消息区键盘选区（类 Vim visual mode）。
 *
 * 激活条件：
 *   - 焦点在 messages，或
 *   - Vim 已启用且在 normal 模式
 *
 * 按键行为：
 *   - **v** → 进入选区，锚点落在"最新一条消息"
 *   - **↑/k** / **↓/j** → 扩/缩 head
 *   - **y** → 拷 [min..max] 范围的消息文本到系统剪贴板，追加一条 system 提示
 *   - **Esc** → 退出选区（由全局 Esc 或本 hook 均可触发）
 *
 * 未激活时彻底不处理按键，避免与 prompt / overlay 冲突。
 */
export function useSelection(opts: { isActive: boolean }): void {
  const messages = useStore((s) => s.messages);
  const selection = useStore((s) => s.selection);
  const enterSelection = useStore((s) => s.enterSelection);
  const moveSelectionHead = useStore((s) => s.moveSelectionHead);
  const clearSelection = useStore((s) => s.clearSelection);
  const appendMessage = useStore((s) => s.appendMessage);
  const vim = useStore((s) => s.vim);
  const focus = useStore((s) => s.focus);

  const canEnter =
    focus === "messages" || (vim.enabled && vim.mode === "normal");

  useInput(
    (input, key) => {
      if (!opts.isActive) return;

      // 未开启选区：只有 `v` 可以进入（且仅在允许的上下文）
      if (!selection) {
        if (input === "v" && canEnter) {
          const n = messages.length;
          if (n === 0) return;
          enterSelection(n - 1);
        }
        return;
      }

      // 已在选区中：统一用 arrow keys 或 j/k
      if (key.escape) {
        clearSelection();
        return;
      }
      if (key.upArrow || input === "k") {
        moveSelectionHead(-1);
        return;
      }
      if (key.downArrow || input === "j") {
        moveSelectionHead(+1);
        return;
      }
      // 快速跳转：到上/下一条指定角色的消息
      //   g / G — 跳到最早 / 最新
      //   u / U — 上/下一条 user
      //   a / A — 上/下一条 assistant
      // head 的绝对定位由 moveSelectionHead(delta) 承接，
      // delta = target - current；跳过不在可见窗口时会自动滚动到可见位置。
      const jumpTo = (targetIdx: number | null) => {
        if (targetIdx === null) return;
        moveSelectionHead(targetIdx - selection.head);
      };
      if (input === "g") {
        jumpTo(0);
        return;
      }
      if (input === "G") {
        jumpTo(messages.length - 1);
        return;
      }
      if (input === "u") {
        jumpTo(findRoleFrom(messages, selection.head, "user", -1));
        return;
      }
      if (input === "U") {
        jumpTo(findRoleFrom(messages, selection.head, "user", +1));
        return;
      }
      if (input === "a") {
        jumpTo(findRoleFrom(messages, selection.head, "assistant", -1));
        return;
      }
      if (input === "A") {
        jumpTo(findRoleFrom(messages, selection.head, "assistant", +1));
        return;
      }
      if (input === "y") {
        const lo = Math.min(selection.anchor, selection.head);
        const hi = Math.max(selection.anchor, selection.head);
        const slice = messages.slice(lo, hi + 1);
        const text = stringifyMessages(slice);
        void clipboardy
          .write(text)
          .then(() => {
            appendMessage({
              id: nextMessageId(),
              role: "system",
              content: `✓ 已复制 ${slice.length} 条到剪贴板（${text.length} 字符）`,
              createdAt: Date.now(),
            });
          })
          .catch((e: Error) => {
            appendMessage({
              id: nextMessageId(),
              role: "error",
              content: `复制失败：${e.message}`,
              createdAt: Date.now(),
            });
          });
        clearSelection();
        return;
      }
      // 任意非 selection 相关按键：不消费，交给其他 input handler
    },
    { isActive: opts.isActive },
  );
}

/**
 * 从 `fromIdx` 出发按方向 `dir` 寻找下一条指定 role 的消息下标。
 * 找不到返回 null（jumpTo 里会 no-op，不移动 head）。
 */
function findRoleFrom(
  messages: readonly Message[],
  fromIdx: number,
  role: Message["role"],
  dir: 1 | -1,
): number | null {
  const n = messages.length;
  if (n === 0) return null;
  let i = fromIdx + dir;
  while (i >= 0 && i < n) {
    if (messages[i]!.role === role) return i;
    i += dir;
  }
  return null;
}

/** 把选区内的消息拼成一段纯文本。role 用中文短标签，避免终端解码问题。 */
export function stringifyMessages(msgs: readonly Message[]): string {
  const labelOf: Record<string, string> = {
    user: "用户",
    assistant: "助手",
    tool: "工具",
    error: "错误",
    system: "系统",
    confirm: "确认",
  };
  return msgs
    .map((m) => `[${labelOf[m.role] ?? m.role}] ${m.content}`)
    .join("\n\n");
}
