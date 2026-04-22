import { describe, expect, it } from "vitest";
import { wrapToWidth, sliceWrappedLines } from "../utils/textWidth.js";

/**
 * 文本宽度工具单测。
 *
 * 注：历史上这些函数是 Messages 虚拟滚动窗口的核心（`pickVisibleMessages` /
 * `measureMessageRows` 依赖它们做行级切片）。print-above 架构改造后虚拟滚动
 * 逻辑全部删除，但 `wrapToWidth` / `sliceWrappedLines` 仍被 Markdown 代码块
 * 渲染复用，所以纯函数测试保留下来。
 */
describe("utils/textWidth wrapToWidth/sliceWrappedLines", () => {
  it("短行原样保留", () => {
    expect(wrapToWidth("hello", 10)).toEqual(["hello"]);
  });

  it("按 `\\n` 拆行", () => {
    expect(wrapToWidth("a\nb\n\nc", 10)).toEqual(["a", "b", "", "c"]);
  });

  it("超宽硬换行（ASCII）", () => {
    expect(wrapToWidth("abcdefghij", 4)).toEqual(["abcd", "efgh", "ij"]);
  });

  it("超宽硬换行（CJK，每字符 2 列）", () => {
    expect(wrapToWidth("你好世界", 4)).toEqual(["你好", "世界"]);
  });

  it("CJK + ASCII 混排", () => {
    expect(wrapToWidth("a你b好c", 3)).toEqual(["a你", "b好", "c"]);
  });

  it("sliceWrappedLines 取中间段", () => {
    expect(sliceWrappedLines("a\nb\nc\nd", 10, 1, 3)).toBe("b\nc");
  });

  it("sliceWrappedLines 范围越界自动裁剪", () => {
    expect(sliceWrappedLines("a\nb", 10, -1, 100)).toBe("a\nb");
    expect(sliceWrappedLines("a\nb", 10, 5, 10)).toBe("");
  });
});
