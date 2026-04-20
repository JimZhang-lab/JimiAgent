import { afterEach, describe, expect, it, vi } from "vitest";
import { tryRunClientCommand } from "../utils/slashCommands.js";
import { useStore } from "../state/store.js";

function makeCtx() {
  return {
    exit: vi.fn(),
    newSession: vi.fn(),
    refreshSessions: vi.fn(),
    openPalette: vi.fn(),
    openHistory: vi.fn(),
    openMemory: vi.fn(),
  };
}

function reset() {
  useStore.setState((s) => ({
    ...s,
    messages: [],
    theme: "auto",
    vim: { enabled: false, mode: "insert" },
  }));
}

describe("utils/slashCommands", () => {
  afterEach(reset);

  it("非命令返回 false", () => {
    expect(tryRunClientCommand("hello", makeCtx())).toBe(false);
    expect(tryRunClientCommand("", makeCtx())).toBe(false);
  });

  it("/quit 触发 exit", () => {
    const ctx = makeCtx();
    expect(tryRunClientCommand("/quit", ctx)).toBe(true);
    expect(ctx.exit).toHaveBeenCalledOnce();
  });

  it("/exit 与 /quit 别名", () => {
    const ctx = makeCtx();
    expect(tryRunClientCommand("/exit", ctx)).toBe(true);
    expect(ctx.exit).toHaveBeenCalledOnce();
  });

  it("/new 带/不带标题", () => {
    const ctx1 = makeCtx();
    tryRunClientCommand("/new", ctx1);
    expect(ctx1.newSession).toHaveBeenCalledWith(undefined);

    const ctx2 = makeCtx();
    tryRunClientCommand("/new 测试标题", ctx2);
    expect(ctx2.newSession).toHaveBeenCalledWith("测试标题");
  });

  it("/clear 清空消息", () => {
    useStore.getState().appendMessage({
      id: "m",
      role: "user",
      content: "x",
      createdAt: Date.now(),
    });
    expect(useStore.getState().messages).toHaveLength(1);
    tryRunClientCommand("/clear", makeCtx());
    expect(useStore.getState().messages).toHaveLength(0);
  });

  it("/help 追加一条 system 说明消息", () => {
    tryRunClientCommand("/help", makeCtx());
    const msgs = useStore.getState().messages;
    expect(msgs).toHaveLength(1);
    expect(msgs[0]?.role).toBe("system");
    expect(msgs[0]?.content).toContain("快捷键");
  });

  it("/theme dark 切主题", () => {
    tryRunClientCommand("/theme dark", makeCtx());
    expect(useStore.getState().theme).toBe("dark");
  });

  it("/theme 非法参数输出帮助", () => {
    tryRunClientCommand("/theme xyz", makeCtx());
    const msg = useStore.getState().messages.at(-1);
    expect(msg?.role).toBe("system");
    expect(msg?.content).toContain("/theme");
  });

  it("/vim toggle", () => {
    tryRunClientCommand("/vim", makeCtx());
    expect(useStore.getState().vim.enabled).toBe(true);
    expect(useStore.getState().vim.mode).toBe("normal");
    tryRunClientCommand("/vim", makeCtx());
    expect(useStore.getState().vim.enabled).toBe(false);
  });

  it("/vim on/off 显式开关", () => {
    tryRunClientCommand("/vim on", makeCtx());
    expect(useStore.getState().vim.enabled).toBe(true);
    tryRunClientCommand("/vim off", makeCtx());
    expect(useStore.getState().vim.enabled).toBe(false);
  });

  it("/memory 打开记忆面板", () => {
    const ctx = makeCtx();
    expect(tryRunClientCommand("/memory", ctx)).toBe(true);
    expect(ctx.openMemory).toHaveBeenCalledOnce();
  });

  it("/mem 作为 /memory 别名", () => {
    const ctx = makeCtx();
    expect(tryRunClientCommand("/mem", ctx)).toBe(true);
    expect(ctx.openMemory).toHaveBeenCalledOnce();
  });
});
