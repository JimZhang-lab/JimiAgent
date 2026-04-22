import { afterEach, describe, expect, it } from "vitest";
import type {
  AgentEvent,
  AgentInfo,
  NodeRequest,
} from "../protocol/events.js";
import type { AgentTransport } from "../transport/AgentTransport.js";
import { wireTransport } from "../state/wiring.js";
import { useStore } from "../state/store.js";

/** 构造一个假的 transport；sent 记请求，emit(ev) 模拟 Python 发事件。 */
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

function mkSession(id: string, title = id) {
  return {
    id,
    title,
    created_at: 1,
    updated_at: 1,
    message_count: 0,
    metadata: {},
  };
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
    // 启动时应主动请求一次 switch_session，回放 default session 历史
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
    // 先模拟切到 s-new
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

  it("tool + tool_result 事件：工具输出追加到同名 tool 消息 content，并清 streaming", () => {
    const { transport, emit } = makeFakeTransport();
    wireTransport(transport);
    emit({ type: "session", session_id: "s-new" } as AgentEvent);
    // on_tool_start
    emit({ type: "tool", name: "list_dir" } as AgentEvent);
    let tool = useStore
      .getState()
      .messages.find((m) => m.role === "tool" && m.toolName === "list_dir");
    expect(tool).toBeDefined();
    expect(tool?.content).toBe("调用工具: list_dir");
    expect(tool?.streaming).toBe(true);

    // on_tool_end：output 追加、streaming 清掉
    emit({
      type: "tool_result",
      name: "list_dir",
      output: "a.txt\nb.txt\nc.txt",
    } as AgentEvent);
    tool = useStore
      .getState()
      .messages.find((m) => m.role === "tool" && m.toolName === "list_dir");
    expect(tool?.content).toBe("调用工具: list_dir\na.txt\nb.txt\nc.txt");
    expect(tool?.streaming).toBe(false);
  });

  it("tool_result 无匹配 tool 消息时兜底创建新消息", () => {
    const { transport, emit } = makeFakeTransport();
    wireTransport(transport);
    emit({ type: "session", session_id: "s-new" } as AgentEvent);
    // 直接发 tool_result，没有 on_tool_start
    emit({
      type: "tool_result",
      name: "shell_exec",
      output: "exit 0",
    } as AgentEvent);
    const msgs = useStore.getState().messages;
    const tool = msgs.find(
      (m) => m.role === "tool" && m.toolName === "shell_exec",
    );
    expect(tool).toBeDefined();
    expect(tool?.content).toBe("调用工具: shell_exec\nexit 0");
    expect(tool?.streaming).toBeFalsy();
  });

  it("bootstrap 也会主动发 list_sessions（让 sessions 尽快可用，配合 sendMessage 的 sid 校验）", () => {
    const { transport, sent } = makeFakeTransport();
    wireTransport(transport);
    expect(sent.some((r) => r.kind === "list_sessions")).toBe(true);
  });

  it("session_deleted 事件 ok=true 触发 list_sessions", () => {
    const { transport, emit, sent } = makeFakeTransport();
    wireTransport(transport);
    emit({ type: "session_deleted", session_id: "s-x", ok: true } as AgentEvent);
    expect(sent.some((r) => r.kind === "list_sessions")).toBe(true);
  });

  it("session_deleted 删的是当前会话 + 还有剩余 → 自动 switch 到剩余首条（防幽灵会话 bug）", () => {
    const { transport, emit, sent } = makeFakeTransport();
    wireTransport(transport);
    // 先铺一个包含 current + 两条其他会话的 sessions 快照
    useStore.getState().setSessions([
      mkSession("s-current", "当前"),
      mkSession("s-b", "B"),
      mkSession("s-c", "C"),
    ]);
    useStore.getState().setCurrentSession("s-current");
    useStore.getState().appendMessage({
      id: "m1",
      role: "user",
      content: "旧消息",
      createdAt: 0,
    });

    const before = sent.length;
    emit({
      type: "session_deleted",
      session_id: "s-current",
      ok: true,
    } as AgentEvent);

    // 本地状态：currentSessionId 清空 + messages 清空
    expect(useStore.getState().currentSessionId).toBeNull();
    expect(useStore.getState().messages).toHaveLength(0);
    // 发出 switch_session 切到剩余首条（s-b）
    const fresh = sent.slice(before);
    expect(
      fresh.some(
        (r) => r.kind === "switch_session" && r.session_id === "s-b",
      ),
    ).toBe(true);
  });

  it("session_deleted 删的是当前 + 无剩余 → 只本地置 null，不乱发 new_session（等后端 L4 ensure）", () => {
    const { transport, emit, sent } = makeFakeTransport();
    wireTransport(transport);
    useStore.getState().setSessions([mkSession("s-only", "仅此一条")]);
    useStore.getState().setCurrentSession("s-only");

    const before = sent.length;
    emit({
      type: "session_deleted",
      session_id: "s-only",
      ok: true,
    } as AgentEvent);

    expect(useStore.getState().currentSessionId).toBeNull();
    const fresh = sent.slice(before);
    // 不该主动发 switch_session / new_session，等后端 ensure_default 回发 session
    expect(fresh.some((r) => r.kind === "switch_session")).toBe(false);
    expect(fresh.some((r) => r.kind === "new_session")).toBe(false);
  });

  it("session_deleted 删的不是当前会话 → 不动 currentSessionId / messages", () => {
    const { transport, emit } = makeFakeTransport();
    wireTransport(transport);
    useStore.getState().setSessions([
      mkSession("s-cur", "A"),
      mkSession("s-other", "B"),
    ]);
    useStore.getState().setCurrentSession("s-cur");
    useStore.getState().appendMessage({
      id: "m1",
      role: "user",
      content: "保留",
      createdAt: 0,
    });
    emit({
      type: "session_deleted",
      session_id: "s-other",
      ok: true,
    } as AgentEvent);
    expect(useStore.getState().currentSessionId).toBe("s-cur");
    expect(useStore.getState().messages).toHaveLength(1);
  });

  it("session_deleted 事件 ok=true 时把该 id 从 sessionSelection 剔除", () => {
    const { transport, emit } = makeFakeTransport();
    wireTransport(transport);
    useStore.getState().setSessionSelection(["s-a", "s-b", "s-c"]);
    emit({ type: "session_deleted", session_id: "s-b", ok: true } as AgentEvent);
    expect(useStore.getState().sessionSelection).toEqual(["s-a", "s-c"]);
  });

  it("session_deleted ok=false 不改动 sessionSelection / 不刷新", () => {
    const { transport, emit, sent } = makeFakeTransport();
    wireTransport(transport);
    useStore.getState().setSessionSelection(["s-a"]);
    const lenBefore = sent.length;
    emit({ type: "session_deleted", session_id: "s-a", ok: false } as AgentEvent);
    expect(useStore.getState().sessionSelection).toEqual(["s-a"]);
    expect(sent.length).toBe(lenBefore);
  });

  it("session_deleted 事件 ok=false 不追加 list_sessions（bootstrap 已发过的不算）", () => {
    const { transport, emit, sent } = makeFakeTransport();
    wireTransport(transport);
    // bootstrap 已发过一次 list_sessions；这里在那个基线之上断言
    const before = sent.length;
    emit({ type: "session_deleted", session_id: "s-x", ok: false } as AgentEvent);
    const fresh = sent.slice(before);
    expect(fresh.some((r) => r.kind === "list_sessions")).toBe(false);
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
