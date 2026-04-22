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
    sessionSelection: [],
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

  it("appendAssistantChunk 超长内容触发 rollover：切首段为 head + 剩余 streaming body", () => {
    // 模拟 24 行终端 -> maxLines=10, threshold=20
    useStore.setState((s) => ({ ...s, dims: { cols: 80, rows: 24 } }));
    const s = useStore.getState();
    // 30 行内容：前 20 行切到 head，后 10 行留在 streaming body
    const lines = Array.from({ length: 30 }, (_, i) => `line-${i + 1}`);
    s.appendAssistantChunk(lines.join("\n"));
    const msgs = useStore.getState().messages;
    expect(msgs).toHaveLength(2);
    expect(msgs[0]?.fragment).toBe("head");
    expect(msgs[0]?.streaming).toBeFalsy();
    expect(msgs[0]?.content.split("\n")).toHaveLength(20);
    expect(msgs[0]?.content.split("\n")[0]).toBe("line-1");
    expect(msgs[1]?.fragment).toBe("body");
    expect(msgs[1]?.streaming).toBe(true);
    expect(msgs[1]?.content.split("\n")).toHaveLength(10);
    expect(msgs[1]?.content.split("\n")[0]).toBe("line-21");
  });

  it("连续 rollover：第二次切出的 fragment=body，streaming 仍 body", () => {
    useStore.setState((s) => ({ ...s, dims: { cols: 80, rows: 24 } }));
    const s = useStore.getState();
    // 30 行触发首次 rollover（head + body streaming）
    s.appendAssistantChunk(Array.from({ length: 30 }, (_, i) => `a${i}`).join("\n"));
    // 再追加 20 行（streaming 从 10 -> 30 行，再次超阈值）
    s.appendAssistantChunk("\n" + Array.from({ length: 20 }, (_, i) => `b${i}`).join("\n"));
    const msgs = useStore.getState().messages;
    expect(msgs).toHaveLength(3);
    expect(msgs[0]?.fragment).toBe("head"); // 首次 rollover 的 head
    expect(msgs[1]?.fragment).toBe("body"); // 第二次 rollover 的 body
    expect(msgs[1]?.streaming).toBeFalsy();
    expect(msgs[2]?.fragment).toBe("body"); // streaming 仍是 body
    expect(msgs[2]?.streaming).toBe(true);
  });

  it("finishAssistantStreaming 把尾部 body 改为 tail 并加 marginBottom 语义", () => {
    useStore.setState((s) => ({ ...s, dims: { cols: 80, rows: 24 } }));
    const s = useStore.getState();
    s.appendAssistantChunk(Array.from({ length: 30 }, (_, i) => `x${i}`).join("\n"));
    s.finishAssistantStreaming();
    const msgs = useStore.getState().messages;
    expect(msgs.at(-1)?.fragment).toBe("tail");
    expect(msgs.at(-1)?.streaming).toBeFalsy();
  });

  it("rollover 代码块保护：切点处于未闭合代码块内时跳过 rollover", () => {
    useStore.setState((s) => ({ ...s, dims: { cols: 80, rows: 24 } }));
    const s = useStore.getState();
    // 前 15 行纯文本 + "```python" + 10 行未闭合代码，共 26 行。
    // 初始 cut 落在代码块里，且后面找不到闭合 fence，所以本次 rollover 会放弃。
    const parts = [
      ...Array.from({ length: 15 }, (_, i) => `p${i}`),
      "```python",
      ...Array.from({ length: 10 }, (_, i) => `c${i}`),
    ];
    s.appendAssistantChunk(parts.join("\n"));
    const msgs = useStore.getState().messages;
    // 放弃本次 rollover，整段仍在单条 streaming message
    expect(msgs).toHaveLength(1);
    expect(msgs[0]?.streaming).toBe(true);
    expect(msgs[0]?.fragment).toBeUndefined();
  });

  it("rollover 代码块保护：切点向后推到闭合 fence 后", () => {
    useStore.setState((s) => ({ ...s, dims: { cols: 80, rows: 24 } }));
    const s = useStore.getState();
    // 前 15 行文本 + ```py + 3 行代码 + ``` + 后 15 行文本，共 35 行。
    // 初始 cut 前缀里 fence 数为偶数，所以无需后推。
    const parts = [
      ...Array.from({ length: 15 }, (_, i) => `p${i}`),
      "```py",
      "c0",
      "c1",
      "c2",
      "```",
      ...Array.from({ length: 15 }, (_, i) => `q${i}`),
    ];
    s.appendAssistantChunk(parts.join("\n"));
    const msgs = useStore.getState().messages;
    expect(msgs).toHaveLength(2);
    // 代码块完整落在 head 片段里
    expect(msgs[0]?.content).toContain("```py");
    expect(msgs[0]?.content).toContain("```");
    expect(msgs[1]?.content.startsWith("q")).toBe(true);
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

  it("toggleSessionSelection add/remove", () => {
    const s = useStore.getState();
    s.toggleSessionSelection("a");
    s.toggleSessionSelection("b");
    s.toggleSessionSelection("c");
    expect(useStore.getState().sessionSelection).toEqual(["a", "b", "c"]);
    s.toggleSessionSelection("b");
    expect(useStore.getState().sessionSelection).toEqual(["a", "c"]);
  });

  it("setSessionSelection 覆盖 + 清空", () => {
    const s = useStore.getState();
    s.setSessionSelection(["x", "y", "z"]);
    expect(useStore.getState().sessionSelection).toEqual(["x", "y", "z"]);
    s.clearSessionSelection();
    expect(useStore.getState().sessionSelection).toEqual([]);
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
