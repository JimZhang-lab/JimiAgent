import type { AgentTransport } from "../transport/AgentTransport.js";
import type { AgentEvent, NodeRequest } from "../protocol/events.js";
import { nextMessageId, useStore } from "./store.js";

/**
 * 把 transport 事件流接到 store。
 *
 * 这是协议层和视图层之间唯一的胶水点。
 */
export function wireTransport(transport: AgentTransport): () => void {
  const s = useStore.getState();
  s.setTransport(transport);

  // transport 先于 React 完成启动时，直接消费 bootstrap 快照。
  const boot = transport.bootstrap;
  if (boot) {
    s.setAgentInfo(boot.agent_info);
    s.setConnected(true);
    if (!s.currentSessionId) {
      s.setCurrentSession(boot.default_session_id);
    }
    // 主动请求一次 switch_session，把默认会话历史回放到 scrollback。
    transport.send({
      kind: "switch_session",
      session_id: boot.default_session_id,
    });
    // 再主动拉一次 sessions，避免启动早期列表为空。
    transport.send({ kind: "list_sessions" });
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
      // bootstrap 走上面的同步路径；这里补事件流路径。
      if (isFirstSession) {
        try {
          sendRequest({
            kind: "switch_session",
            session_id: ev.default_session_id,
          });
          // 同步拉一次 sessions 列表
          sendRequest({ kind: "list_sessions" });
        } catch {
          // transport 未绑定时静默跳过
        }
      }
      return;
    }

    case "text":
      // 首个 text 把活动切到 thinking
      if (!s.activity || s.activity.kind !== "thinking") {
        s.beginActivity("thinking", "生成回复中");
      }
      s.appendAssistantChunk(ev.content);
      return;

    case "tool":
      // tool 先以 streaming 形式挂起，等 tool_result 补完整后再进 scrollback。
      s.beginActivity("tool", ev.name);
      s.appendMessage({
        id: nextMessageId(),
        role: "tool",
        content: `调用工具: ${ev.name}`,
        toolName: ev.name,
        streaming: true,
        createdAt: Date.now(),
      });
      return;

    case "tool_result":
      // tool_result 会补全对应 tool 消息，并把它从 pending 升到 Static。
      s.appendToolOutput(ev.name, ev.output);
      return;

    case "session": {
      // session 事件先清空当前消息，再给后续 history 回放让路。
      s.setCurrentSession(ev.session_id);
      s.clearMessages();
      sendRequest({ kind: "list_sessions" });
      return;
    }

    case "history": {
      // 按顺序回放整段会话历史，让它们直接进入 scrollback。
      if (ev.session_id !== s.currentSessionId) {
        // 忽略穿插进来的过期回放
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
      // 结束 streaming，避免 spinner 一直转
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
      // 删完会话后刷新列表，并同步清掉本地选择状态。
      //       thread → 消息写进孤儿 session（session_mgr 里无元数据，
      //       HistoryPanel 看不到）。这是"无会话也能对话"bug 的根因之一。
      //
      //       策略：
      //       - 若 sessions 里（去掉被删的）还有剩余 → 自动 switch 到第一条
      //       - 若列表会变空 → 本地置 null + clearMessages，等后端 L4 的
      //         ensure_default_session 回发 session 事件切过去
      if (ev.ok) {
        const sel = s.sessionSelection;
        if (sel.includes(ev.session_id)) {
          s.setSessionSelection(sel.filter((id) => id !== ev.session_id));
        }
        sendRequest({ kind: "list_sessions" });
        if (ev.session_id === s.currentSessionId) {
          s.setCurrentSession(null);
          s.clearMessages();
          const remaining = s.sessions.filter(
            (x) => x.id !== ev.session_id,
          );
          if (remaining.length > 0 && remaining[0]) {
            sendRequest({
              kind: "switch_session",
              session_id: remaining[0].id,
            });
          }
          // remaining 为空：依赖后端 handle_delete_session 的 ensure_default
          // 兜底发 session 事件切到新默认会话；前端短暂处于 null 期，
          // sendMessage 的 `!sid` 防御会拦住这段窗口内的用户输入。
        }
      }
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
