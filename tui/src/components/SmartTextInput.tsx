import React, { useReducer, useRef } from "react";
import { Text, useInput } from "ink";
import chalk from "chalk";

/**
 * 受控文本输入框，替换 `ink-text-input`。
 *
 * 为什么自造轮子：
 *   1. ink-text-input@6 的 useInput 只在 key.return / Tab / 箭头 / Ctrl+C 有特判，
 *      其它所有组合（含 Ctrl+U / Ctrl+W / Ctrl+A / Ctrl+E / Ctrl+K）都走 else
 *      分支把字母当字符插入。macOS 终端会把 Cmd+Backspace 映射成 \x15（Ctrl+U），
 *      结果就出现"按 Cmd+Backspace 输出了 u"的 bug。
 *   2. 其 useEffect 只在 `cursorOffset > value.length - 1` 时才 reset，外部
 *      setValue 扩长 value（如 Tab 补全 `/` → `/quit `）时光标会保持在
 *      老位置（字符 `q` 处），造成 "光标在 /quit 中间" 的视觉错位。
 *
 * 修复后支持的行为：
 *   - 光标移动：←/→，Ctrl+B / Ctrl+F，Home / End（Ctrl+A / Ctrl+E）
 *   - 删除：Backspace / Delete
 *   - 行编辑：Ctrl+U 删到行首、Ctrl+K 删到行尾、Ctrl+W 删上一个词
 *   - 词跳：Option+B / Option+F（Ink 暴露为 key.meta）
 *   - 外部 setValue：若用户上次正好在末尾（或外部替换了整个 value），
 *     光标跟到新末尾；否则 clamp 到 [0, value.length] 保持用户位置。
 *   - 忽略 Ctrl+C / Tab / ↑↓：把按键让给上层 useInput 处理（取消 / 补全 / 历史）。
 */

export interface SmartTextInputProps {
  value: string;
  onChange: (next: string) => void;
  onSubmit?: (value: string) => void;
  placeholder?: string;
  /** 是否展示光标块；为 false 时纯渲染 value。 */
  showCursor?: boolean;
  /** 是否处理输入；为 false 时整个 useInput 不激活。 */
  focus?: boolean;
}

export function SmartTextInput({
  value,
  onChange,
  onSubmit,
  placeholder = "",
  showCursor = true,
  focus = true,
}: SmartTextInputProps): React.ReactElement {
  // cursor 存 ref 而不是 state，目的是 render-time 能够同步跟随 props.value 的
  // 变化（Tab 补全 / history 切换等外部改写），避免 useEffect 延迟一帧导致的
  // 光标错位 bug（/ 按 Tab → /quit，光标卡在 q 上）。配合 forceRender 触发 UI 更新。
  const cursorRef = useRef<number>(value.length);
  const lastValueRef = useRef<string>(value);
  const [, forceRender] = useReducer((x: number) => x + 1, 0);

  // —— render-time 同步 ——
  // 注：render body 里修改 ref 在 React 18 strict mode 下会被 double-invoke，
  // 但幂等（把 cursor 设成 value.length 或 clamp），结果不受影响。
  if (value !== lastValueRef.current) {
    // 外部改 value：把 cursor 拉到新末尾
    cursorRef.current = value.length;
    lastValueRef.current = value;
  } else {
    // 保险：value 长度变了仍能 clamp 在合法区间
    if (cursorRef.current > value.length) cursorRef.current = value.length;
    if (cursorRef.current < 0) cursorRef.current = 0;
  }

  const safeCursor = cursorRef.current;

  const setCursor = (n: number) => {
    cursorRef.current = Math.max(0, Math.min(n, value.length));
    forceRender();
  };

  const emit = (next: string, nextCursor: number) => {
    // 先把 ref 提前同步为新 value/cursor，避免 onChange 回来触发的 rerender
    // 时被 render-time 同步逻辑误判为"外部改写"再把 cursor 重置到末尾。
    lastValueRef.current = next;
    cursorRef.current = Math.max(0, Math.min(nextCursor, next.length));
    if (next !== value) onChange(next);
    forceRender();
  };

  useInput(
    (input, key) => {
      if (!focus) return;

      // 把以下按键让给上层处理（取消 / 补全 / 历史等）
      if (key.ctrl && input === "c") return;
      if (key.tab) return;
      if (key.upArrow || key.downArrow) return;

      if (key.return) {
        onSubmit?.(value);
        return;
      }

      // —— 光标移动 ——
      if (key.leftArrow || (key.ctrl && input === "b")) {
        setCursor(Math.max(0, safeCursor - 1));
        return;
      }
      if (key.rightArrow || (key.ctrl && input === "f")) {
        setCursor(Math.min(value.length, safeCursor + 1));
        return;
      }
      if (key.ctrl && input === "a") {
        setCursor(0);
        return;
      }
      if (key.ctrl && input === "e") {
        setCursor(value.length);
        return;
      }

      // —— Option/Meta 词跳 ——
      if (key.meta && (input === "b" || input === "f")) {
        setCursor(
          input === "b"
            ? prevWordStart(value, safeCursor)
            : nextWordEnd(value, safeCursor),
        );
        return;
      }

      // —— 删除 ——
      if (key.backspace || key.delete) {
        if (safeCursor === 0) return;
        const next = value.slice(0, safeCursor - 1) + value.slice(safeCursor);
        emit(next, safeCursor - 1);
        return;
      }
      if (key.ctrl && input === "u") {
        // 删光标到行首（Cmd+Backspace 的 macOS 默认绑定也走到这里）
        emit(value.slice(safeCursor), 0);
        return;
      }
      if (key.ctrl && input === "k") {
        // 删光标到行尾
        emit(value.slice(0, safeCursor), safeCursor);
        return;
      }
      if (key.ctrl && input === "w") {
        // 删上一个词（Option+Backspace 的 macOS 默认绑定走到这里）
        const start = prevWordStart(value, safeCursor);
        emit(value.slice(0, start) + value.slice(safeCursor), start);
        return;
      }

      // —— 其它 Ctrl / Meta 组合：吞掉，不当字符插入 ——
      if (key.ctrl || key.meta) return;

      // —— 普通字符插入 ——
      if (input && input.length > 0) {
        const next =
          value.slice(0, safeCursor) + input + value.slice(safeCursor);
        emit(next, safeCursor + input.length);
      }
    },
    { isActive: focus },
  );

  // —— 渲染 ——
  if (!showCursor || !focus) {
    if (value.length === 0 && placeholder) {
      return <Text>{chalk.grey(placeholder)}</Text>;
    }
    return <Text>{value}</Text>;
  }

  if (value.length === 0) {
    if (placeholder.length > 0) {
      return (
        <Text>
          {chalk.inverse(placeholder[0]!) + chalk.grey(placeholder.slice(1))}
        </Text>
      );
    }
    return <Text>{chalk.inverse(" ")}</Text>;
  }

  // cursor 落在某字符上时反色那个字符；cursor === length 时末尾追加反色空格
  let rendered = "";
  for (let i = 0; i < value.length; i++) {
    rendered += i === safeCursor ? chalk.inverse(value[i]!) : value[i];
  }
  if (safeCursor >= value.length) {
    rendered += chalk.inverse(" ");
  }
  return <Text>{rendered}</Text>;
}

/** 找到 `pos` 之前的词起点（跳过空白再跳过非空白）。导出供测试。 */
export function prevWordStart(value: string, pos: number): number {
  let i = pos;
  while (i > 0 && /\s/.test(value[i - 1]!)) i--;
  while (i > 0 && !/\s/.test(value[i - 1]!)) i--;
  return i;
}

/** 找到 `pos` 之后的词终点（跳过空白再跳过非空白）。导出供测试。 */
export function nextWordEnd(value: string, pos: number): number {
  let i = pos;
  while (i < value.length && /\s/.test(value[i]!)) i++;
  while (i < value.length && !/\s/.test(value[i]!)) i++;
  return i;
}
