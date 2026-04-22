import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
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

  it("/cwd 输出 workspace 与 user cwd", () => {
    useStore.getState().setWorkspaceCwd("/tmp/workspace");
    tryRunClientCommand("/cwd", makeCtx());
    const msg = useStore.getState().messages.at(-1);
    expect(msg?.role).toBe("system");
    expect(msg?.content).toContain("/tmp/workspace");
    expect(msg?.content).toContain("工作目录");
  });

  it("/open 无匹配时提示", () => {
    tryRunClientCommand("/open", makeCtx());
    const msg = useStore.getState().messages.at(-1);
    expect(msg?.content).toContain("没有可识别的 URL");
  });

  it("/open 无参数时列出所有 URL", () => {
    useStore.getState().appendMessage({
      id: "m1",
      role: "assistant",
      content: "看下 https://example.com 和 https://foo.bar/baz 这两个",
      createdAt: Date.now(),
    });
    tryRunClientCommand("/open", makeCtx());
    const msg = useStore.getState().messages.at(-1);
    expect(msg?.content).toMatch(/\[1\] https:\/\/example\.com/);
    expect(msg?.content).toMatch(/\[2\] https:\/\/foo\.bar\/baz/);
  });

  it("/export 写入到指定路径", () => {
    useStore.getState().appendMessage({
      id: "u",
      role: "user",
      content: "你好",
      createdAt: Date.now(),
    });
    useStore.getState().appendMessage({
      id: "a",
      role: "assistant",
      content: "你好呀",
      createdAt: Date.now(),
    });
    const target = path.join(os.tmpdir(), `jimi-export-test-${Date.now()}.md`);
    tryRunClientCommand(`/export ${target}`, makeCtx());
    expect(fs.existsSync(target)).toBe(true);
    const content = fs.readFileSync(target, "utf8");
    expect(content).toContain("你好");
    expect(content).toContain("你好呀");
    fs.unlinkSync(target);
  });

  it("/save 作为 /export 别名", () => {
    useStore.getState().appendMessage({
      id: "u",
      role: "user",
      content: "hi",
      createdAt: Date.now(),
    });
    const target = path.join(os.tmpdir(), `jimi-save-test-${Date.now()}.md`);
    tryRunClientCommand(`/save ${target}`, makeCtx());
    expect(fs.existsSync(target)).toBe(true);
    fs.unlinkSync(target);
  });
});
