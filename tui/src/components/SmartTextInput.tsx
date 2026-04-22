import React, { useReducer, useRef } from "react";
import { Text, useInput } from "ink";
import chalk from "chalk";

/**
 * 受控文本输入框，用来替代 `ink-text-input`。
 *
 * 重点修掉组合键被误插入字符和外部 setValue 后光标错位的问题。
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
  // cursor 放 ref 里，保证 render 阶段就能跟上外部 value 变化。
  const cursorRef = useRef<number>(value.length);
  const lastValueRef = useRef<string>(value);
  const [, forceRender] = useReducer((x: number) => x + 1, 0);

  // render-time 同步：外部改 value 时直接修正 cursor。
  if (value !== lastValueRef.current) {
    // 外部改 value：把 cursor 拉到新末尾
    cursorRef.current = value.length;
    lastValueRef.current = value;
  } else {
    // 保险：始终把 cursor clamp 在合法区间
    if (cursorRef.current > value.length) cursorRef.current = value.length;
    if (cursorRef.current < 0) cursorRef.current = 0;
  }

  const safeCursor = cursorRef.current;

  const setCursor = (n: number) => {
    cursorRef.current = Math.max(0, Math.min(n, value.length));
    forceRender();
  };

  const emit = (next: string, nextCursor: number) => {
    // 先同步 ref，避免 rerender 时被误判成“外部改写”。
    lastValueRef.current = next;
    cursorRef.current = Math.max(0, Math.min(nextCursor, next.length));
    if (next !== value) onChange(next);
    forceRender();
  };

  useInput(
    (input, key) => {
      if (!focus) return;

      // 这些按键让给上层处理（取消 / 补全 / 历史等）
      if (key.ctrl && input === "c") return;
      if (key.tab) return;
      if (key.upArrow || key.downArrow) return;

      if (key.return) {
        onSubmit?.(value);
        return;
      }

      // 光标移动
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

      // Option/Meta 词跳
      if (key.meta && (input === "b" || input === "f")) {
        setCursor(
          input === "b"
            ? prevWordStart(value, safeCursor)
            : nextWordEnd(value, safeCursor),
        );
        return;
      }

      // 删除
      if (key.backspace || key.delete) {
        if (safeCursor === 0) return;
        const next = value.slice(0, safeCursor - 1) + value.slice(safeCursor);
        emit(next, safeCursor - 1);
        return;
      }
      if (key.ctrl && input === "u") {
        // 删到行首（macOS 的 Cmd+Backspace 也会走到这里）
        emit(value.slice(safeCursor), 0);
        return;
      }
      if (key.ctrl && input === "k") {
        // 删到行尾
        emit(value.slice(0, safeCursor), safeCursor);
        return;
      }
      if (key.ctrl && input === "w") {
        // 删上一个词（macOS 的 Option+Backspace 也会走到这里）
        const start = prevWordStart(value, safeCursor);
        emit(value.slice(0, start) + value.slice(safeCursor), start);
        return;
      }

      // 其它 Ctrl / Meta 组合直接吞掉，不当字符插入
      if (key.ctrl || key.meta) return;

      // 普通字符插入
      if (input && input.length > 0) {
        const next =
          value.slice(0, safeCursor) + input + value.slice(safeCursor);
        emit(next, safeCursor + input.length);
      }
    },
    { isActive: focus },
  );

  // 渲染
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

  // cursor 落在字符上就反色；落在末尾就补一个反色空格。
  let rendered = "";
  for (let i = 0; i < value.length; i++) {
    rendered += i === safeCursor ? chalk.inverse(value[i]!) : value[i];
  }
  if (safeCursor >= value.length) {
    rendered += chalk.inverse(" ");
  }
  return <Text>{rendered}</Text>;
}

/** 找到 `pos` 之前的词起点。导出供测试。 */
export function prevWordStart(value: string, pos: number): number {
  let i = pos;
  while (i > 0 && /\s/.test(value[i - 1]!)) i--;
  while (i > 0 && !/\s/.test(value[i - 1]!)) i--;
  return i;
}

/** 找到 `pos` 之后的词终点。导出供测试。 */
export function nextWordEnd(value: string, pos: number): number {
  let i = pos;
  while (i < value.length && /\s/.test(value[i]!)) i++;
  while (i < value.length && !/\s/.test(value[i]!)) i++;
  return i;
}
