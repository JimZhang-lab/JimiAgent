import type { AgentTransport } from "../transport/AgentTransport.js";
import type { AgentEvent, NodeRequest } from "../protocol/events.js";
import { nextMessageId, useStore } from "./store.js";

/**
 * 把 transport 的事件流接到 store actions。
 *
 * 这是唯一一处"协议 ⇄ 视图"的胶水层；屏幕/组件只读 store、只调 useAgent()
 * 返回的 helper，不直接碰 transport。
 *
 * 返回 unbind()，组件卸载时调用。
 */
export function wireTransport(transport: AgentTransport): () => void {
  const s = useStore.getState();
  s.setTransport(transport);

  // 如果 transport.start() 已经先于 React 挂载完成，ready 事件无法再重放，
  // 此处同步读一次 bootstrap 快照，显式点亮 connected。
  const boot = transport.bootstrap;
  if (boot) {
    s.setAgentInfo(boot.agent_info);
    s.setConnected(true);
    if (!s.currentSessionId) {
      s.setCurrentSession(boot.default_session_id);
    }
  }

  const offEvent = transport.onEvent((ev) => dispatchEvent(ev));
  const offErr = transport.onError((err) => {
    useStore.getState().setBootError(err.message);
  });
  const offClose = transport.onClose((code) => {
    useStore.getState().setConnected(false);
    if (!useStore.getState().bootError && code !== 0) {
      useStore
        .getState()
        .setBootError(`tui_worker exited unexpectedly (code=${code ?? "null"})`);
    }
  });

  return () => {
    offEvent();
    offErr();
    offClose();
  };
}

function dispatchEvent(ev: AgentEvent): void {
  const s = useStore.getState();
  switch (ev.type) {
    case "ready":
      s.setAgentInfo(ev.agent_info);
      s.setConnected(true);
      if (!s.currentSessionId) {
        s.setCurrentSession(ev.default_session_id);
      }
      return;

    case "text":
      // 首个 text 切换到 thinking 活动（如果还没开启）
      if (!s.activity || s.activity.kind !== "thinking") {
        s.beginActivity("thinking", "生成回复中");
      }
      s.appendAssistantChunk(ev.content);
      return;

    case "tool":
      s.beginActivity("tool", ev.name);
      s.appendMessage({
        id: nextMessageId(),
        role: "tool",
        content: `调用工具: ${ev.name}`,
        toolName: ev.name,
        createdAt: Date.now(),
      });
      return;

    case "session":
      s.setCurrentSession(ev.session_id);
      return;

    case "error":
      // 结束 streaming，防止 spinner 永转
      s.finishAssistantStreaming();
      s.endActivity("error", ev.content);
      s.appendMessage({
        id: nextMessageId(),
        role: "error",
        content: ev.content,
        createdAt: Date.now(),
      });
      return;

    case "confirm_required":
      s.finishAssistantStreaming();
      s.beginActivity(
        "confirm",
        typeof ev.payload?.summary === "string" ? ev.payload.summary : "等待确认",
      );
      s.setPendingConfirm({
        interrupt_id: ev.interrupt_id,
        resumable: ev.resumable,
        payload: ev.payload,
      });
      s.appendMessage({
        id: nextMessageId(),
        role: "confirm",
        content:
          typeof ev.payload?.summary === "string"
            ? ev.payload.summary
            : JSON.stringify(ev.payload),
        confirm: ev,
        createdAt: Date.now(),
      });
      return;

    case "done":
      s.finishAssistantStreaming();
      s.endActivity("done");
      return;

    case "sessions":
      s.setSessions(ev.items);
      return;

    case "memories":
      s.setMemories(ev.items, ev.query);
      return;

    case "memory_deleted":
      if (ev.ok) s.removeMemoryLocal(ev.memory_id);
      return;

    case "status":
      if (ev.data.agent) s.setAgentInfo(ev.data.agent);
      return;

    case "trace":
    case "log":
    case "commands":
    case "pong":
    case "session_deleted":
      // 这些事件交给具体屏幕自己订阅（见 useTransportEvent）
      return;

    default:
      // 穷举检查兜底（未知事件已在 isAgentEvent 过滤）
      return;
  }
}

/** 发送请求；自动读当前 transport，若未连就报错。 */
export function sendRequest(req: NodeRequest): void {
  const t = useStore.getState().transport;
  if (!t) throw new Error("transport 未初始化");
  t.send(req);
}

/**
 * 针对某些 meta 事件（如 sessions / commands / pong）的订阅 hook。
 * 不走 store，避免大量一次性 meta 进 store。
 */
export function subscribeTransport(
  cb: (ev: AgentEvent) => void,
): () => void {
  const t = useStore.getState().transport;
  if (!t) return () => {};
  return t.onEvent(cb);
}
