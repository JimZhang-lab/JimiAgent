import { useCallback } from "react";
import { nextMessageId, useStore } from "../state/store.js";
import { sendRequest } from "../state/wiring.js";

/**
 * 对话层 hook。暴露高层 action：sendMessage / cancel / approve / deny /
 * newSession / switchSession / deleteSession / clear。
 *
 * 所有 action 对 store 和 transport 的改动都在此处集中，组件只调 hook 返回的
 * 函数，不直接碰 transport 或 store 的具体 action。
 */
export function useAgent() {
  const currentSessionId = useStore((s) => s.currentSessionId);
  const streaming = useStore((s) => s.streaming);
  const pendingConfirm = useStore((s) => s.pendingConfirm);
  const appendMessage = useStore((s) => s.appendMessage);
  const clearMessages = useStore((s) => s.clearMessages);
  const setPendingConfirm = useStore((s) => s.setPendingConfirm);
  const beginActivity = useStore((s) => s.beginActivity);
  const endActivity = useStore((s) => s.endActivity);

  const sendMessage = useCallback(
    (text: string) => {
      if (!text.trim()) return;
      const st = useStore.getState();
      const sid = st.currentSessionId;
      if (!sid) return;
      // 防御"幽灵会话"：currentSessionId 可能指向一个已被 delete_session 移除
      // 的 id（session_deleted 事件到达与 wiring 本地切换之间存在窗口；或外部
      // HTTP 脚本删了某会话而本端状态未同步）。若 sessions 列表非空且 sid 不
      // 在列表里，说明进入了这种 race → 拒绝发送。
      //
      // sessions 列表为空时放行：启动期 list_sessions 响应可能还未到达，但
      // ready/bootstrap 已设置 currentSessionId，这段窗口要允许发送。
      if (st.sessions.length > 0 && !st.sessions.some((x) => x.id === sid)) {
        return;
      }
      appendMessage({
        id: nextMessageId(),
        role: "user",
        content: text,
        createdAt: Date.now(),
      });
      // 立即进入 thinking 状态，ActivityBar 就能显示 spinner。
      // 首个 text / tool 事件会把 kind 切成具体 label（wiring.ts 已处理）。
      beginActivity("thinking", "正在思考…");
      sendRequest({
        kind: "chat",
        session_id: sid,
        message: text,
      });
    },
    [appendMessage, beginActivity],
  );

  const cancel = useCallback(() => {
    const sid = useStore.getState().currentSessionId;
    if (!sid) return;
    // 本地立刻清 activity，避免 UI 残留；Python 端 cancel 也会发 done
    endActivity("done");
    sendRequest({ kind: "cancel", session_id: sid });
  }, [endActivity]);

  const approve = useCallback(() => {
    const sid = useStore.getState().currentSessionId;
    if (!sid) return;
    setPendingConfirm(null);
    // 恢复后同样有空白期（工具继续执行或模型继续生成），先置 thinking
    beginActivity("thinking", "已允许，继续执行…");
    sendRequest({ kind: "resume", session_id: sid, approve: true });
  }, [setPendingConfirm, beginActivity]);

  const deny = useCallback(() => {
    const sid = useStore.getState().currentSessionId;
    if (!sid) return;
    setPendingConfirm(null);
    beginActivity("thinking", "已拒绝，继续…");
    sendRequest({ kind: "resume", session_id: sid, approve: false });
  }, [setPendingConfirm, beginActivity]);

  const newSession = useCallback(
    (title?: string) => {
      // 本地立刻清，避免 Python 还在走 session 事件时 user 先看到残留的旧消息。
      // Python 回发 session 事件时 id 改变会再清一次，幂等。
      clearMessages();
      sendRequest({ kind: "new_session", title });
    },
    [clearMessages],
  );

  const switchSession = useCallback(
    (sessionId: string) => {
      sendRequest({ kind: "switch_session", session_id: sessionId });
      clearMessages();
    },
    [clearMessages],
  );

  const deleteSession = useCallback(
    (sessionId: string) => {
      // 若删的是当前会话，本地先清一下；Python 之后会回发新的 session 事件。
      const cur = useStore.getState().currentSessionId;
      if (cur === sessionId) clearMessages();
      sendRequest({ kind: "delete_session", session_id: sessionId });
    },
    [clearMessages],
  );

  const refreshSessions = useCallback(() => {
    sendRequest({ kind: "list_sessions" });
  }, []);

  const clear = useCallback(() => {
    clearMessages();
  }, [clearMessages]);

  return {
    currentSessionId,
    streaming,
    pendingConfirm,
    sendMessage,
    cancel,
    approve,
    deny,
    newSession,
    switchSession,
    deleteSession,
    refreshSessions,
    clear,
  };
}
