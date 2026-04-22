import { afterEach, describe, expect, it } from "vitest";
import { nextMessageId, useStore } from "../state/store.js";

function reset() {
  useStore.setState((s) => ({
    ...s,
    messages: [],
    streaming: false,
    pendingConfirm: null,
    currentSessionId: null,
    sessions: [],
    activity: null,
    activityLog: [],
    promptDraft: "",
    slashSuggestRows: 0,
    workspaceCwd: "",
    memories: [],
    memoryQuery: "",
    memoryLoading: false,
    memoryOpen: false,
    memorySelection: [],
    selection: null,
    overlayOwnsEscape: false,
  }));
}

describe("state/store", () => {
  afterEach(reset);

  it("nextMessageId 单调递增", () => {
    const a = nextMessageId();
    const b = nextMessageId();
    expect(a).not.toBe(b);
  });

  it("appendAssistantChunk 合并到同一条流式消息", () => {
    const s = useStore.getState();
    s.appendAssistantChunk("Hello ");
    s.appendAssistantChunk("world");
    const msgs = useStore.getState().messages;
    expect(msgs).toHaveLength(1);
    expect(msgs[0]?.content).toBe("Hello world");
    expect(msgs[0]?.streaming).toBe(true);
    expect(useStore.getState().streaming).toBe(true);
  });

  it("finishAssistantStreaming 关闭流式标记", () => {
    const s = useStore.getState();
    s.appendAssistantChunk("hi");
    s.finishAssistantStreaming();
    const last = useStore.getState().messages.at(-1);
    expect(last?.streaming).toBe(false);
    expect(useStore.getState().streaming).toBe(false);
  });

  it("appendAssistantChunk 在非流式消息后新开一条", () => {
    const s = useStore.getState();
    s.appendMessage({
      id: nextMessageId(),
      role: "user",
      content: "hi",
      createdAt: Date.now(),
    });
    s.appendAssistantChunk("A");
    s.appendAssistantChunk("B");
    const msgs = useStore.getState().messages;
    expect(msgs).toHaveLength(2);
    expect(msgs[1]?.content).toBe("AB");
  });

  it("clearMessages 清空并重置 streaming", () => {
    const s = useStore.getState();
    s.appendAssistantChunk("x");
    s.setPendingConfirm({ interrupt_id: "i", resumable: true, payload: {} });
    s.clearMessages();
    const st = useStore.getState();
    expect(st.messages).toHaveLength(0);
    expect(st.streaming).toBe(false);
    expect(st.pendingConfirm).toBeNull();
  });

  it("setVim 合并 enabled 与 mode", () => {
    const s = useStore.getState();
    s.setVim({ enabled: true });
    expect(useStore.getState().vim.enabled).toBe(true);
    s.setVim({ mode: "normal" });
    expect(useStore.getState().vim).toMatchObject({
      enabled: true,
      mode: "normal",
    });
  });

  it("beginActivity 设置当前 activity", () => {
    useStore.getState().beginActivity("tool", "web_search");
    const a = useStore.getState().activity;
    expect(a).not.toBeNull();
    expect(a?.kind).toBe("tool");
    expect(a?.label).toBe("web_search");
  });

  it("beginActivity 连续调用：前一个进 log，新的为当前", () => {
    const s = useStore.getState();
    s.beginActivity("tool", "first");
    s.beginActivity("tool", "second");
    const st = useStore.getState();
    expect(st.activity?.label).toBe("second");
    expect(st.activityLog).toHaveLength(1);
    expect(st.activityLog[0]?.label).toBe("first");
    expect(st.activityLog[0]?.durationMs).toBeGreaterThanOrEqual(0);
  });

  it("endActivity 清空当前并入 log", () => {
    const s = useStore.getState();
    s.beginActivity("thinking", "generating");
    s.endActivity("done");
    const st = useStore.getState();
    expect(st.activity).toBeNull();
    expect(st.activityLog).toHaveLength(1);
    expect(st.activityLog[0]?.kind).toBe("text");
  });

  it("endActivity 传 error 时 kind=error", () => {
    const s = useStore.getState();
    s.beginActivity("tool", "shell_exec");
    s.endActivity("error", "timeout");
    expect(useStore.getState().activityLog[0]?.kind).toBe("error");
    expect(useStore.getState().activityLog[0]?.label).toContain("timeout");
  });

  it("setPromptDraft 更新 draft", () => {
    const s = useStore.getState();
    expect(useStore.getState().promptDraft).toBe("");
    s.setPromptDraft("/hel");
    expect(useStore.getState().promptDraft).toBe("/hel");
    s.setPromptDraft("");
    expect(useStore.getState().promptDraft).toBe("");
  });

  it("setMemories 写入列表且关掉 loading", () => {
    const s = useStore.getState();
    s.setMemoryLoading(true);
    s.setMemories(
      [
        {
          id: 1,
          kind: "semantic",
          subject: "user",
          text: "hello",
          created_at: "",
          updated_at: "",
          source_session: "",
          hits: 0,
        },
      ],
      "hi",
    );
    const st = useStore.getState();
    expect(st.memories).toHaveLength(1);
    expect(st.memoryQuery).toBe("hi");
    expect(st.memoryLoading).toBe(false);
  });

  it("removeMemoryLocal 移除指定 id", () => {
    const s = useStore.getState();
    s.setMemories(
      [1, 2, 3].map((id) => ({
        id,
        kind: "semantic",
        subject: "user",
        text: `t${id}`,
        created_at: "",
        updated_at: "",
        source_session: "",
        hits: 0,
      })),
    );
    s.removeMemoryLocal(2);
    expect(useStore.getState().memories.map((m) => m.id)).toEqual([1, 3]);
  });

  it("toggleMemorySelection add/remove + 排序", () => {
    const s = useStore.getState();
    s.toggleMemorySelection(3);
    s.toggleMemorySelection(1);
    s.toggleMemorySelection(2);
    expect(useStore.getState().memorySelection).toEqual([1, 2, 3]);
    s.toggleMemorySelection(2);
    expect(useStore.getState().memorySelection).toEqual([1, 3]);
  });

  it("setMemorySelection 覆盖 + 清空", () => {
    const s = useStore.getState();
    s.setMemorySelection([5, 1, 3]);
    expect(useStore.getState().memorySelection).toEqual([1, 3, 5]);
    s.clearMemorySelection();
    expect(useStore.getState().memorySelection).toEqual([]);
  });

  it("setMemories 修剪失效的选中 id", () => {
    const s = useStore.getState();
    s.setMemorySelection([1, 2, 3]);
    s.setMemories(
      [1, 3].map((id) => ({
        id,
        kind: "semantic",
        subject: "user",
        text: "",
        created_at: "",
        updated_at: "",
        source_session: "",
        hits: 0,
      })),
    );
    // id=2 已不存在 → 从 selection 里剔除
    expect(useStore.getState().memorySelection).toEqual([1, 3]);
  });

  it("removeMemoryLocal 同步剔除 selection", () => {
    const s = useStore.getState();
    s.setMemories(
      [1, 2].map((id) => ({
        id,
        kind: "semantic",
        subject: "user",
        text: "",
        created_at: "",
        updated_at: "",
        source_session: "",
        hits: 0,
      })),
    );
    s.setMemorySelection([1, 2]);
    s.removeMemoryLocal(1);
    expect(useStore.getState().memorySelection).toEqual([2]);
  });

  it("setMemoryOpen(false) 清空 selection", () => {
    const s = useStore.getState();
    s.setMemorySelection([1, 2]);
    s.setMemoryOpen(true);
    expect(useStore.getState().memorySelection).toEqual([1, 2]);
    s.setMemoryOpen(false);
    expect(useStore.getState().memorySelection).toEqual([]);
  });

  it("clearMessages 同时重置 activity", () => {
    const s = useStore.getState();
    s.appendMessage({ id: "a", role: "user", content: "x", createdAt: 0 });
    s.beginActivity("tool", "t");
    s.clearMessages();
    const st = useStore.getState();
    expect(st.messages).toHaveLength(0);
    expect(st.activity).toBeNull();
    expect(st.activityLog).toHaveLength(0);
  });

  it("enterSelection 锚点落在合法范围", () => {
    const s = useStore.getState();
    for (let i = 0; i < 3; i++) {
      s.appendMessage({
        id: `m${i}`,
        role: "user",
        content: `t${i}`,
        createdAt: 0,
      });
    }
    s.enterSelection(99); // 超界 → 裁剪到末尾
    const sel = useStore.getState().selection;
    expect(sel).not.toBeNull();
    expect(sel!.anchor).toBe(2);
    expect(sel!.head).toBe(2);
  });

  it("enterSelection 空消息不创建 selection", () => {
    const s = useStore.getState();
    s.enterSelection(0);
    expect(useStore.getState().selection).toBeNull();
  });

  it("moveSelectionHead 扩/缩 + 边界裁剪", () => {
    const s = useStore.getState();
    for (let i = 0; i < 5; i++) {
      s.appendMessage({
        id: `m${i}`,
        role: "user",
        content: "x",
        createdAt: 0,
      });
    }
    s.enterSelection(2);
    s.moveSelectionHead(+5); // 4
    expect(useStore.getState().selection?.head).toBe(4);
    s.moveSelectionHead(-10); // 0
    expect(useStore.getState().selection?.head).toBe(0);
    // anchor 保持
    expect(useStore.getState().selection?.anchor).toBe(2);
  });

  it("clearSelection 把 selection 置 null", () => {
    const s = useStore.getState();
    s.appendMessage({ id: "a", role: "user", content: "x", createdAt: 0 });
    s.enterSelection(0);
    expect(useStore.getState().selection).not.toBeNull();
    s.clearSelection();
    expect(useStore.getState().selection).toBeNull();
  });

  it("moveSelectionHead 在未激活时无副作用", () => {
    const s = useStore.getState();
    s.moveSelectionHead(+3);
    expect(useStore.getState().selection).toBeNull();
  });
});
