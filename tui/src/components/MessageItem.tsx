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
}

/**
 * 单条消息的渲染。根据 role 切换颜色、前缀、排版。
 */
export function MessageItem({ message }: MessageItemProps): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const { role, content, streaming, toolName } = message;

  switch (role) {
    case "user":
      return (
        <Box marginBottom={1} flexDirection="row">
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

    case "assistant":
      return (
        <Box marginBottom={1} flexDirection="row">
          <Box marginRight={1}>
            <Text color={theme.colors.assistant} bold>
              ◆
            </Text>
          </Box>
          <Box flexDirection="column" flexGrow={1}>
            <Markdown source={content} streaming={streaming} />
          </Box>
        </Box>
      );

    case "tool":
      return (
        <Box marginBottom={1} flexDirection="row">
          <Box marginRight={1}>
            <Text color={theme.colors.tool}>⚙</Text>
          </Box>
          <Box flexDirection="column" flexGrow={1}>
            <Text color={theme.colors.tool}>
              {toolName ? `调用工具: ${toolName}` : content}
            </Text>
            {looksLikeUnifiedDiff(content) && (
              <DiffView source={content} maxRows={30} />
            )}
          </Box>
        </Box>
      );

    case "error":
      return (
        <Box
          marginBottom={1}
          flexDirection="column"
          borderStyle="round"
          borderColor={theme.colors.error}
          paddingX={1}
        >
          <Text color={theme.colors.error} bold>
            错误
          </Text>
          <Text color={theme.colors.error}>{content}</Text>
        </Box>
      );

    case "confirm":
      return (
        <Box
          marginBottom={1}
          flexDirection="column"
          borderStyle="round"
          borderColor={theme.colors.confirm}
          paddingX={1}
        >
          <Text color={theme.colors.confirm} bold>
            ⚠ 等待确认
          </Text>
          <Text color={theme.colors.text}>{content}</Text>
          <Text color={theme.colors.textDim}>
            按 [y] 允许 / [n] 拒绝
          </Text>
        </Box>
      );

    case "system":
    default:
      return (
        <Box marginBottom={1}>
          <Text color={theme.colors.textDim} italic>
            {content}
          </Text>
        </Box>
      );
  }
}
