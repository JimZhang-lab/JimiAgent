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

  const sendMessage = useCallback(
    (text: string) => {
      if (!text.trim()) return;
      const sid = useStore.getState().currentSessionId;
      if (!sid) return;
      appendMessage({
        id: nextMessageId(),
        role: "user",
        content: text,
        createdAt: Date.now(),
      });
      sendRequest({
        kind: "chat",
        session_id: sid,
        message: text,
      });
    },
    [appendMessage],
  );

  const cancel = useCallback(() => {
    const sid = useStore.getState().currentSessionId;
    if (!sid) return;
    sendRequest({ kind: "cancel", session_id: sid });
  }, []);

  const approve = useCallback(() => {
    const sid = useStore.getState().currentSessionId;
    if (!sid) return;
    setPendingConfirm(null);
    sendRequest({ kind: "resume", session_id: sid, approve: true });
  }, [setPendingConfirm]);

  const deny = useCallback(() => {
    const sid = useStore.getState().currentSessionId;
    if (!sid) return;
    setPendingConfirm(null);
    sendRequest({ kind: "resume", session_id: sid, approve: false });
  }, [setPendingConfirm]);

  const newSession = useCallback((title?: string) => {
    sendRequest({ kind: "new_session", title });
  }, []);

  const switchSession = useCallback((sessionId: string) => {
    sendRequest({ kind: "switch_session", session_id: sessionId });
    clearMessages();
  }, [clearMessages]);

  const deleteSession = useCallback((sessionId: string) => {
    sendRequest({ kind: "delete_session", session_id: sessionId });
  }, []);

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
