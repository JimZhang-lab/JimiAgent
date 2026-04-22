import { describe, expect, it } from "vitest";
import chalk from "chalk";
// vitest 默认非 TTY，chalk.level 会退到 0 导致 inverse 不输出 ANSI 序列。
// 强制到 ANSI 16-color 级别，保证 \u001b[7m...\u001b[27m 能出现在 lastFrame() 里
chalk.level = 1;
import { render } from "ink-testing-library";
import {
  SmartTextInput,
  prevWordStart,
  nextWordEnd,
} from "../components/SmartTextInput.js";

const ESC = String.fromCharCode(27);
const ANSI_SGR_RE = new RegExp(`${ESC}\\[[0-9;]*m`, "g");
const INVERSE_SEGMENT_RE = new RegExp(`${ESC}\\[7m([^${ESC}]*)${ESC}\\[27m`);

/**
 * ink-testing-library 的 MockStdin 只 emit 'data'，Ink 5 内部读取 stdin 用
 * readline/keypress，不会订阅 mock stdin 的 'data' 事件触发 useInput。
 * 所以这里**不**尝试模拟按键；改为测试：
 *   - 纯函数 prevWordStart / nextWordEnd
 *   - 外部 props 变化时的渲染结果（cursor 同步到末尾的 bug 修复）
 */

function stripAnsi(s: string | undefined): string {
  if (!s) return "";
  return s.replace(ANSI_SGR_RE, "");
}

/** 提取第一段反色字符（光标块覆盖的字符）。 */
function extractCursorChar(s: string | undefined): string {
  if (!s) return "";
  const m = s.match(INVERSE_SEGMENT_RE);
  return m?.[1] ?? "";
}

describe("SmartTextInput 纯渲染", () => {
  it("空 value 渲染反色空格作为光标块", () => {
    const { lastFrame } = render(
      <SmartTextInput value="" onChange={() => {}} />,
    );
    expect(extractCursorChar(lastFrame())).toBe(" ");
  });

  it("空 value + placeholder：第一字符反色", () => {
    const { lastFrame } = render(
      <SmartTextInput
        value=""
        onChange={() => {}}
        placeholder="请输入…"
      />,
    );
    expect(extractCursorChar(lastFrame())).toBe("请");
  });

  it("非空 value：光标默认在末尾，显示为追加的反色空格，不覆盖任何字符", () => {
    const { lastFrame } = render(
      <SmartTextInput value="hello" onChange={() => {}} />,
    );
    const plain = stripAnsi(lastFrame());
    expect(plain).toContain("hello");
    // 光标应是末尾空格（反色），不应吞末尾字符
    expect(extractCursorChar(lastFrame())).toBe(" ");
  });

  it("外部 setValue 扩长 value（如 Tab 补全）后 cursor 跟到末尾（修 /quit 光标在 q 上的老 bug）", () => {
    const { rerender, lastFrame } = render(
      <SmartTextInput value="/" onChange={() => {}} />,
    );
    // 第一次渲染：cursor 在末尾（位置 1），展示 `/` + 反色空格
    expect(stripAnsi(lastFrame())).toContain("/");
    // 模拟 Tab 补全把 value 改成 `/quit `
    rerender(<SmartTextInput value="/quit " onChange={() => {}} />);
    const plain = stripAnsi(lastFrame());
    expect(plain).toContain("/quit");
    // 光标应落在末尾（反色空格），不在 `q` 上
    expect(extractCursorChar(lastFrame())).toBe(" ");
  });

  it("focus=false 时不渲染光标块", () => {
    const { lastFrame } = render(
      <SmartTextInput value="hello" onChange={() => {}} focus={false} />,
    );
    expect(extractCursorChar(lastFrame())).toBe("");
    expect(stripAnsi(lastFrame())).toBe("hello");
  });

  it("showCursor=false 时不渲染光标块", () => {
    const { lastFrame } = render(
      <SmartTextInput
        value="hello"
        onChange={() => {}}
        showCursor={false}
      />,
    );
    expect(extractCursorChar(lastFrame())).toBe("");
  });
});

describe("SmartTextInput 词跳工具", () => {
  it("prevWordStart 跳过连续空白到前一个词起点", () => {
    expect(prevWordStart("hello world", 11)).toBe(6); // 从末尾到 world 起点
    expect(prevWordStart("hello world", 6)).toBe(0); // 从 world 起点到 hello 起点
    // 从末尾到 abc 起点（跳非空白；不会一路退到纯空白前缀之前）
    expect(prevWordStart("   abc", 6)).toBe(3);
    // 光标在首部空白正中：跳空白后 i=0 停止
    expect(prevWordStart("   abc", 2)).toBe(0);
    expect(prevWordStart("", 0)).toBe(0);
  });

  it("nextWordEnd 跳过空白到下一个词终点", () => {
    expect(nextWordEnd("hello world", 0)).toBe(5);
    expect(nextWordEnd("hello world", 5)).toBe(11);
    expect(nextWordEnd("hello   world", 5)).toBe(13);
    expect(nextWordEnd("abc", 3)).toBe(3);
  });

  it("prevWordStart / nextWordEnd 在 CJK 上也工作（非空白即当作词）", () => {
    expect(prevWordStart("你好 世界", 5)).toBe(3);
    expect(nextWordEnd("你好 世界", 0)).toBe(2);
  });
});
