import { afterEach, describe, expect, it } from "vitest";
import {
  registerKeybinding,
  unregisterKeybinding,
  listKeybindings,
  groupByScope,
  __resetKeybindingsForTest,
} from "../keybindings/registry.js";

describe("keybindings/registry", () => {
  afterEach(() => {
    __resetKeybindingsForTest();
  });

  it("register 返回 id；list 按 scope/key 排序", () => {
    const id1 = registerKeybinding("global", "ctrl+s", "保存");
    const id2 = registerKeybinding("global", "ctrl+c", "退出");
    expect(id1).not.toBe(id2);
    const all = listKeybindings();
    expect(all.map((b) => b.key)).toEqual(["ctrl+c", "ctrl+s"]);
  });

  it("冲突时不抛错，复用既有 id", () => {
    const a = registerKeybinding("global", "ctrl+s", "保存");
    const b = registerKeybinding("global", "ctrl+s", "save-alias");
    expect(a).toBe(b);
    expect(listKeybindings()).toHaveLength(1);
  });

  it("不同 scope 同 key 不冲突", () => {
    registerKeybinding("global", "escape", "关弹层");
    registerKeybinding("overlay-palette", "escape", "关面板");
    expect(listKeybindings()).toHaveLength(2);
  });

  it("unregister 后消失；再注册不冲突", () => {
    const id = registerKeybinding("prompt", "tab", "补全");
    unregisterKeybinding(id);
    expect(listKeybindings()).toHaveLength(0);
    const id2 = registerKeybinding("prompt", "tab", "again");
    expect(id2).not.toBe(id);
  });

  it("list(scope) 过滤正确", () => {
    registerKeybinding("global", "ctrl+c", "1");
    registerKeybinding("prompt", "tab", "2");
    registerKeybinding("overlay-palette", "up", "3");
    expect(listKeybindings("prompt")).toHaveLength(1);
    expect(listKeybindings("global")).toHaveLength(1);
  });

  it("groupByScope 按 scope 分桶", () => {
    registerKeybinding("global", "ctrl+c", "1");
    registerKeybinding("global", "ctrl+p", "2");
    registerKeybinding("selection", "v", "3");
    const g = groupByScope();
    expect(Object.keys(g).sort()).toEqual(["global", "selection"]);
    expect(g.global).toHaveLength(2);
    expect(g.selection).toHaveLength(1);
  });
});
