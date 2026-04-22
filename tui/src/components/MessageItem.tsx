import React from "react";
import { Box, Text } from "ink";
import type { Message } from "../state/store.js";
import { useStore } from "../state/store.js";
import { resolveTheme } from "../themes/index.js";
import { Markdown } from "./Markdown.js";
import { DiffView } from "./DiffView.js";
import { looksLikeUnifiedDiff } from "../utils/diff.js";

export interface MessageItemProps {
  message: Message;
  /** 选区高亮：在消息前加竖条 + 背景提示。 */
  highlighted?: boolean;
}

/**
 * 单条消息的渲染。根据 role 切换颜色、前缀、排版。
 *
 * `highlighted` 表示当前处于 visual 选区内，外层用 Yoga border 绘制色条。
 * 避免使用 backgroundColor：在 16 色 / macOS Terminal 下背景+前景对比度不可控，
 * 改用左侧 `▌` 竖条 + 整段保留原色，视觉可辨识度高且对主题鲁棒。
 */
export function MessageItem({
  message,
  highlighted,
}: MessageItemProps): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const { role, content, streaming, toolName, fragment } = message;
  const gutter = highlighted ? (
    <Text color={theme.colors.accent} bold>
      {"▌"}
    </Text>
  ) : null;

  switch (role) {
    case "user":
      return (
        <Box marginBottom={1} flexDirection="row">
          {gutter}
          <Box marginRight={1}>
            <Text color={theme.colors.user} bold>
              ❯
            </Text>
          </Box>
          <Box flexDirection="column" flexGrow={1}>
            <Text color={theme.colors.user}>{content}</Text>
          </Box>
        </Box>
      );

    case "assistant": {
      // rollover 分段：
      //   fragment=undefined → 完整单段，按原样 Markdown 渲染（含 streaming 光标）
      //   fragment="head"    → 首段（带 ◆），marginBottom=0 紧贴下一段
      //   fragment="body"    → 中间段（无 ◆、用 " " 占位），marginBottom=0
      //   fragment="tail"    → 末段（无 ◆ 占位），marginBottom=1 视觉分隔
      // 带 fragment 的都用纯 Text，避免跨段 Markdown 结构被拆断。
      const isBody = fragment === "body";
      const isTail = fragment === "tail";
      const isHead = fragment === "head";
      const hideGlyph = isBody || isTail;
      const mb = isHead || isBody ? 0 : 1;
      return (
        <Box marginBottom={mb} flexDirection="row">
          {gutter}
          <Box marginRight={1}>
            {hideGlyph ? (
              <Text> </Text>
            ) : (
              <Text color={theme.colors.assistant} bold>
                ◆
              </Text>
            )}
          </Box>
          <Box flexDirection="column" flexGrow={1}>
            {fragment ? (
              <Text color={theme.colors.text}>{content}</Text>
            ) : (
              <Markdown source={content} streaming={streaming} />
            )}
          </Box>
        </Box>
      );
    }

    case "tool": {
      // tool 消息 content 的约定格式：
      //   - 实时流式（on_tool_start）：content = "调用工具: <name>"，无输出
      //   - 历史回放（_emit_history_for）：content = "调用工具: <name>\n<输出>"
      //   - tool_result 事件追加：同历史回放格式
      // 把首行"调用工具: X"剥离，剩余部分作为工具输出预览并折叠展示。
      const output = extractToolOutput(content, toolName);
      const outputLines: string[] = output ? output.split(/\r?\n/) : [];
      // 折叠阈值：预览最多 3 行，超过则显示"…已折叠 N 行"
      const PREVIEW_LINES = 3;
      const truncated = outputLines.length > PREVIEW_LINES;
      const visible = truncated
        ? outputLines.slice(0, PREVIEW_LINES)
        : outputLines;
      const isDiff = looksLikeUnifiedDiff(output);
      return (
        <Box marginBottom={1} flexDirection="row">
          {gutter}
          <Box marginRight={1}>
            <Text color={theme.colors.tool}>⚙</Text>
          </Box>
          <Box flexDirection="column" flexGrow={1}>
            <Text color={theme.colors.tool}>
              {toolName ? `调用工具: ${toolName}` : content}
              {outputLines.length > 0 && (
                <Text color={theme.colors.textDim}>
                  {`  · ${outputLines.length} 行输出`}
                </Text>
              )}
            </Text>
            {isDiff ? (
              <DiffView source={output} maxRows={30} />
            ) : (
              visible.length > 0 && (
                <Box flexDirection="column" marginLeft={2}>
                  {visible.map((line, i) => (
                    <Text key={i} color={theme.colors.textDim}>
                      {line || " "}
                    </Text>
                  ))}
                  {truncated && (
                    <Text color={theme.colors.textDim} italic>
                      {`… 已折叠 ${outputLines.length - PREVIEW_LINES} 行（完整内容在终端滚动记录里）`}
                    </Text>
                  )}
                </Box>
              )
            )}
          </Box>
        </Box>
      );
    }

    case "error":
      return (
        <Box marginBottom={1} flexDirection="row">
          {gutter}
          <Box
            flexDirection="column"
            borderStyle="round"
            borderColor={theme.colors.error}
            paddingX={1}
            flexGrow={1}
          >
            <Text color={theme.colors.error} bold>
              错误
            </Text>
            <Text color={theme.colors.error}>{content}</Text>
          </Box>
        </Box>
      );

    case "confirm":
      return (
        <Box marginBottom={1} flexDirection="row">
          {gutter}
          <Box
            flexDirection="column"
            borderStyle="round"
            borderColor={theme.colors.confirm}
            paddingX={1}
            flexGrow={1}
          >
            <Text color={theme.colors.confirm} bold>
              ⚠ 等待确认
            </Text>
            <Text color={theme.colors.text}>{content}</Text>
            <Text color={theme.colors.textDim}>
              按 [y] 允许 / [n] 拒绝
            </Text>
          </Box>
        </Box>
      );

    case "system":
    default:
      return (
        <Box marginBottom={1} flexDirection="row">
          {gutter}
          <Text color={theme.colors.textDim} italic>
            {content}
          </Text>
        </Box>
      );
  }
}

/**
 * 从 tool message 的 content 里剥离首行"调用工具: <name>"，返回纯工具输出。
 *
 * 约定格式：
 *   - 实时流式（on_tool_start）：`"调用工具: <name>"`，无输出 → 返回 ""
 *   - 历史回放 / tool_result 事件：`"调用工具: <name>\n<输出>"` → 返回 `<输出>`
 *
 * 若 content 不以"调用工具:"开头（比如旧格式 / 异常路径），返回空串，
 * 此时 MessageItem 只渲染 toolName 行，不显示折叠预览。
 */
function extractToolOutput(content: string, toolName?: string): string {
  if (!content) return "";
  // 优先匹配显式前缀 `调用工具: <toolName>`
  if (toolName) {
    const prefix = `调用工具: ${toolName}`;
    if (content.startsWith(prefix)) {
      return content.slice(prefix.length).replace(/^\r?\n/, "").trimEnd();
    }
  }
  // 兜底：任意首行形如 "调用工具: xxx" 都剥离
  const nl = content.indexOf("\n");
  const firstLine = nl >= 0 ? content.slice(0, nl) : content;
  if (/^调用工具:\s*\S+/.test(firstLine)) {
    return nl >= 0 ? content.slice(nl + 1).trimEnd() : "";
  }
  return "";
}
