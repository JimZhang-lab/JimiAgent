import { afterEach, describe, expect, it } from "vitest";
import type {
  AgentEvent,
  AgentInfo,
  NodeRequest,
} from "../protocol/events.js";
import type { AgentTransport } from "../transport/AgentTransport.js";
import { wireTransport } from "../state/wiring.js";
import { useStore } from "../state/store.js";

/**
 * 构造一个假的 transport：capture 发出的请求用于断言；
 * emit(ev) 模拟 Python 发送事件。
 */
function makeFakeTransport(): {
  transport: AgentTransport;
  emit: (ev: AgentEvent) => void;
  sent: NodeRequest[];
} {
  const listeners: Array<(ev: AgentEvent) => void> = [];
  const sent: NodeRequest[] = [];
  const info: AgentInfo = {
    model: "gpt-test",
    embedding: "text-emb",
    skills: [],
    gateway: { host: "127.0.0.1", port: 0 },
    memory: "disabled",
  };
  const transport: AgentTransport = {
    async start() {},
    send(req) {
      sent.push(req);
    },
    onEvent(cb) {
      listeners.push(cb);
      return () => {
        const i = listeners.indexOf(cb);
        if (i >= 0) listeners.splice(i, 1);
      };
    },
    onError() {
      return () => {};
    },
    onClose() {
      return () => {};
    },
    async close() {},
    isAlive: true,
    bootstrap: {
      agent_info: info,
      default_session_id: "s-initial",
      worker_version: "0",
    },
  };
  const emit = (ev: AgentEvent) => {
    for (const cb of [...listeners]) cb(ev);
  };
  return { transport, emit, sent };
}

function reset() {
  useStore.setState((s) => ({
    ...s,
    messages: [],
    currentSessionId: null,
    sessions: [],
    streaming: false,
    pendingConfirm: null,
    activity: null,
    activityLog: [],
    transport: null,
    connected: false,
    agentInfo: null,
  }));
}

describe("state/wiring.dispatchEvent", () => {
  afterEach(reset);

  it("bootstrap 时设置 connected + currentSessionId 并触发 switch_session", () => {
    const { transport, sent } = makeFakeTransport();
    wireTransport(transport);
    expect(useStore.getState().connected).toBe(true);
    expect(useStore.getState().currentSessionId).toBe("s-initial");
    // 启动时应主动请求一次 switch_session，让服务端回放 default session 的历史
    expect(
      sent.some(
        (r) => r.kind === "switch_session" && r.session_id === "s-initial",
      ),
    ).toBe(true);
  });

  it("session 事件总是清 messages + 请求 list_sessions（为 history 回放做准备）", () => {
    const { transport, emit, sent } = makeFakeTransport();
    wireTransport(transport);
    useStore.getState().appendMessage({
      id: "m1",
      role: "user",
      content: "old",
      createdAt: 0,
    });
    expect(useStore.getState().messages).toHaveLength(1);

    emit({ type: "session", session_id: "s-new" } as AgentEvent);

    expect(useStore.getState().currentSessionId).toBe("s-new");
    expect(useStore.getState().messages).toHaveLength(0);
    expect(sent.some((r) => r.kind === "list_sessions")).toBe(true);
  });

  it("history 事件批量 append messages 到当前会话", () => {
    const { transport, emit } = makeFakeTransport();
    wireTransport(transport);
    // 先模拟切换到 s-new（清空 + currentSessionId=s-new）
    emit({ type: "session", session_id: "s-new" } as AgentEvent);
    emit({
      type: "history",
      session_id: "s-new",
      items: [
        { role: "user", content: "你好" },
        { role: "assistant", content: "你好！" },
        { role: "tool", content: "调用工具: shell_exec\n...", tool_name: "shell_exec" },
      ],
    } as AgentEvent);
    const msgs = useStore.getState().messages;
    expect(msgs).toHaveLength(3);
    expect(msgs[0]?.role).toBe("user");
    expect(msgs[1]?.role).toBe("assistant");
    expect(msgs[2]?.role).toBe("tool");
    expect(msgs[2]?.toolName).toBe("shell_exec");
  });

  it("history 事件 session_id 与当前不匹配时被丢弃", () => {
    const { transport, emit } = makeFakeTransport();
    wireTransport(transport);
    emit({ type: "session", session_id: "s-new" } as AgentEvent);
    emit({
      type: "history",
      session_id: "s-OTHER",
      items: [{ role: "user", content: "应该被忽略" }],
    } as AgentEvent);
    expect(useStore.getState().messages).toHaveLength(0);
  });

  it("session_deleted 事件 ok=true 触发 list_sessions", () => {
    const { transport, emit, sent } = makeFakeTransport();
    wireTransport(transport);
    emit({ type: "session_deleted", session_id: "s-x", ok: true } as AgentEvent);
    expect(sent.some((r) => r.kind === "list_sessions")).toBe(true);
  });

  it("session_deleted 事件 ok=false 不刷新", () => {
    const { transport, emit, sent } = makeFakeTransport();
    wireTransport(transport);
    emit({ type: "session_deleted", session_id: "s-x", ok: false } as AgentEvent);
    expect(sent.some((r) => r.kind === "list_sessions")).toBe(false);
  });

  it("done 事件清 streaming + activity", () => {
    const { transport, emit } = makeFakeTransport();
    wireTransport(transport);
    useStore.getState().beginActivity("thinking", "t");
    useStore.getState().setStreaming(true);
    emit({ type: "done" } as AgentEvent);
    expect(useStore.getState().streaming).toBe(false);
    expect(useStore.getState().activity).toBeNull();
  });

  it("error 事件追加 error 消息 + 清 streaming", () => {
    const { transport, emit } = makeFakeTransport();
    wireTransport(transport);
    useStore.getState().setStreaming(true);
    emit({ type: "error", content: "oops" } as AgentEvent);
    const last = useStore.getState().messages.at(-1);
    expect(last?.role).toBe("error");
    expect(last?.content).toBe("oops");
    expect(useStore.getState().streaming).toBe(false);
  });
});
