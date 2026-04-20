import React from "react";
import { Box, Text } from "ink";
import { useStore } from "../state/store.js";
import { resolveTheme } from "../themes/index.js";

/**
 * 1 行状态栏：左侧显示会话 / 模型 / 计数，右侧显示连接状态 + 模式提示。
 */
export function StatusBar(): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const session = useStore((s) => {
    const id = s.currentSessionId;
    return id ? s.sessions.find((x) => x.id === id) ?? null : null;
  });
  const agentInfo = useStore((s) => s.agentInfo);
  const connected = useStore((s) => s.connected);
  const streaming = useStore((s) => s.streaming);
  const vim = useStore((s) => s.vim);

  const left: React.ReactNode[] = [];
  if (session) {
    left.push(
      <Text key="title" color={theme.colors.secondary} bold>
        {session.title}
      </Text>,
      <Text key="id" color={theme.colors.textDim}>
        {" "}
        [{session.id}]
      </Text>,
    );
  } else if (connected) {
    left.push(
      <Text key="noSess" color={theme.colors.textDim}>
        （无会话）
      </Text>,
    );
  } else {
    left.push(
      <Text key="conn" color={theme.colors.warning}>
        连接中…
      </Text>,
    );
  }

  if (agentInfo) {
    left.push(
      <Text key="sep1" color={theme.colors.textDim}>
        {"  ·  "}
      </Text>,
      <Text key="model" color={theme.colors.info}>
        {agentInfo.model}
      </Text>,
    );
    if (session) {
      left.push(
        <Text key="sep2" color={theme.colors.textDim}>
          {"  ·  "}
        </Text>,
        <Text key="msgs" color={theme.colors.accent}>
          msgs:{session.message_count}
        </Text>,
      );
    }
  }

  const right: React.ReactNode[] = [];
  if (streaming) {
    right.push(
      <Text key="stream" color={theme.colors.primary} bold>
        ⟲ 生成中
      </Text>,
      <Text key="space" color={theme.colors.textDim}>
        {"  "}
      </Text>,
    );
  }
  if (vim.enabled) {
    const color =
      vim.mode === "normal"
        ? theme.colors.primary
        : vim.mode === "visual"
          ? theme.colors.warning
          : theme.colors.success;
    right.push(
      <Text key="vim" color={color} bold>
        {vim.mode.toUpperCase()}
      </Text>,
      <Text key="vimSep" color={theme.colors.textDim}>
        {"  "}
      </Text>,
    );
  }
  right.push(
    <Text key="hint" color={theme.colors.textDim}>
      ? 帮助 · /quit 退出
    </Text>,
  );

  return (
    <Box
      flexDirection="row"
      justifyContent="space-between"
      paddingX={1}
      borderStyle="single"
      borderColor={theme.colors.border}
      borderTop={true}
      borderBottom={false}
      borderLeft={false}
      borderRight={false}
    >
      <Box>{left}</Box>
      <Box>{right}</Box>
    </Box>
  );
}
