import React from "react";
import { Box, Text } from "ink";
import { useStore } from "../state/store.js";
import { resolveTheme } from "../themes/index.js";
import { parseUnifiedDiff, type DiffLine, type ParsedDiff } from "../utils/diff.js";

export interface DiffViewProps {
  /** 原始 unified diff 文本。 */
  source: string;
  /** 最大渲染行数；超过会截断并提示。 */
  maxRows?: number;
}

/**
 * 把 unified diff 文本按 add/del/context 分别上色渲染。不使用 backgroundColor
 * 避免 16-color 终端下对比度差；用 "+" "-" 前缀 + fg 色即可辨认。
 */
export function DiffView({
  source,
  maxRows = 40,
}: DiffViewProps): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const parsed = React.useMemo<ParsedDiff>(
    () => parseUnifiedDiff(source),
    [source],
  );

  const stats = React.useMemo(() => summarize(parsed), [parsed]);

  // 展平 hunk，统一裁剪
  const flat: { hunk: number; line: DiffLine }[] = [];
  parsed.hunks.forEach((h, hi) => {
    flat.push({ hunk: hi, line: { kind: "meta", text: h.header } });
    for (const l of h.lines) flat.push({ hunk: hi, line: l });
  });
  const visible = flat.slice(0, maxRows);
  const truncated = flat.length > maxRows;

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor={theme.colors.border}
      paddingX={1}
    >
      <Box>
        <Text color={theme.colors.textDim}>
          {parsed.fileA} → {parsed.fileB}
        </Text>
        <Text color={theme.colors.success}> +{stats.adds} </Text>
        <Text color={theme.colors.error}>-{stats.dels}</Text>
      </Box>
      {visible.map((row, i) => (
        <DiffLineRow key={i} line={row.line} />
      ))}
      {truncated && (
        <Text color={theme.colors.textDim}>
          …还剩 {flat.length - maxRows} 行，展开需进入消息查看
        </Text>
      )}
    </Box>
  );
}

function summarize(p: ParsedDiff): { adds: number; dels: number } {
  let adds = 0;
  let dels = 0;
  for (const h of p.hunks) {
    for (const l of h.lines) {
      if (l.kind === "add") adds++;
      else if (l.kind === "del") dels++;
    }
  }
  return { adds, dels };
}

function DiffLineRow({ line }: { line: DiffLine }): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  switch (line.kind) {
    case "add":
      return (
        <Text color={theme.colors.success}>
          <Text bold>+</Text>
          {line.text}
        </Text>
      );
    case "del":
      return (
        <Text color={theme.colors.error}>
          <Text bold>-</Text>
          {line.text}
        </Text>
      );
    case "meta":
      return (
        <Text color={theme.colors.info} italic>
          {line.text}
        </Text>
      );
    case "context":
    default:
      return (
        <Text color={theme.colors.textDim}>
          {" "}
          {line.text}
        </Text>
      );
  }
}
