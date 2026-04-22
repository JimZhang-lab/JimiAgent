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
    // 启动后主动请求一次 switch_session，触发服务端回放 default session 的历史
    // （print-above 架构下，这些历史消息会直接打印到 terminal scrollback）。
    transport.send({
      kind: "switch_session",
      session_id: boot.default_session_id,
    });
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
    case "ready": {
      s.setAgentInfo(ev.agent_info);
      s.setConnected(true);
      const isFirstSession = !s.currentSessionId;
      if (isFirstSession) {
        s.setCurrentSession(ev.default_session_id);
      }
      // bootstrap 走 wireTransport 里同步路径；这里是事件流路径的补全：
      // 仅在首次进入时请求 switch_session，触发服务端回放 default session 的历史
      if (isFirstSession) {
        try {
          sendRequest({
            kind: "switch_session",
            session_id: ev.default_session_id,
          });
        } catch {
          // transport 未绑定时静默跳过
        }
      }
      return;
    }

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

    case "session": {
      // 切换 / 新建会话后 Python 会发 session 事件确认当前会话。
      // 切换语义下 Python 紧接着会发一条 `history` 事件回放 checkpointer 里
      // 的历史消息（见 tui_worker._emit_history_for）。所以这里一律先
      // clearMessages，给 history 事件一个干净的舞台。
      //   - 启动阶段首次 session（prev 为 null）也要清一下（理论上已空），
      //     保证幂等
      //   - 同时刷新 sessions 列表，让 HistoryPanel 看到新会话条目
      s.setCurrentSession(ev.session_id);
      s.clearMessages();
      sendRequest({ kind: "list_sessions" });
      return;
    }

    case "history": {
      // 批量回放会话历史：按 items 顺序追加到 store.messages。
      // 每条都非 streaming，Messages 组件会把它们放进 Static items —— 一次 flush
      // 里 Ink 会把整段历史打印到 stdout（进入 terminal scrollback）。
      // 注意：若回放期间用户已有未完成的流式请求，理论上不会发生（switch_session
      // 在发 history 前已 cancel current_stream_task）。
      if (ev.session_id !== s.currentSessionId) {
        // 防御：session 事件与 history 事件之间可能穿插异步，忽略不匹配的回放
        return;
      }
      for (const it of ev.items) {
        s.appendMessage({
          id: nextMessageId(),
          role: it.role,
          content: it.content,
          toolName: it.tool_name,
          createdAt: Date.now(),
        });
      }
      return;
    }

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

    case "session_deleted":
      // Python 删完会话后一定要刷新列表，否则 HistoryPanel 仍显示已删记录。
      // 若删的是当前会话，Python 会紧跟一条 session 事件切到 fallback 会话，
      // 那条事件会走 id 变化分支顺带 list_sessions；这里是无论如何都刷一次。
      if (ev.ok) sendRequest({ kind: "list_sessions" });
      return;

    case "trace":
    case "log":
    case "commands":
    case "pong":
      // 这些事件交给具体屏幕自己订阅（见 subscribeTransport）
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
