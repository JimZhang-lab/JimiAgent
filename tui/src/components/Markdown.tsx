import React from "react";
import { Box, Text } from "ink";
import { resolveTheme } from "../themes/index.js";
import { useStore } from "../state/store.js";

/**
 * 极简 Markdown → Ink 渲染器。
 *
 * 支持：
 *   - **粗体** / __粗体__
 *   - *斜体* / _斜体_
 *   - `inline code`
 *   - ``` code block ```（支持指定语言，M2 不做语法高亮）
 *   - # ## ### 标题（1-3 级）
 *   - - / * 无序列表项
 *   - 1. 有序列表项
 *   - > 引用
 *   - [text](url) 链接（只显示为 underline + accent 色）
 *
 * 以"行级 block"为单位解析；段内 inline 再独立解析。此实现故意保持
 * ~150 行内，可用但不完备；不适合渲染复杂文档（表格 / 嵌套列表等），
 * 但足以表达常见 LLM 输出。
 */
export interface MarkdownProps {
  source: string;
  /** 真实流式增量时，末尾追加"▎"光标感。 */
  streaming?: boolean;
}

export function Markdown({ source, streaming }: MarkdownProps): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const blocks = React.useMemo(() => parseBlocks(source), [source]);

  return (
    <Box flexDirection="column">
      {blocks.map((b, i) => (
        <BlockNode key={i} block={b} />
      ))}
      {streaming && (
        <Text color={theme.colors.primary} dimColor>
          ▎
        </Text>
      )}
    </Box>
  );
}

// ============================================================================
// Block 解析
// ============================================================================

type Block =
  | { kind: "heading"; level: 1 | 2 | 3; text: string }
  | { kind: "paragraph"; text: string }
  | { kind: "code"; lang: string; text: string }
  | { kind: "ul"; items: string[] }
  | { kind: "ol"; items: string[] }
  | { kind: "quote"; text: string }
  | { kind: "blank" };

function parseBlocks(src: string): Block[] {
  const lines = src.split(/\r?\n/);
  const out: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const raw = lines[i] ?? "";
    const line = raw;
    // 空行
    if (!line.trim()) {
      if (out[out.length - 1]?.kind !== "blank") out.push({ kind: "blank" });
      i++;
      continue;
    }
    // 代码块
    const codeStart = line.match(/^```\s*([\w-]+)?\s*$/);
    if (codeStart) {
      const lang = codeStart[1] ?? "";
      const buf: string[] = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i] ?? "")) {
        buf.push(lines[i] ?? "");
        i++;
      }
      i++; // skip closing ```
      out.push({ kind: "code", lang, text: buf.join("\n") });
      continue;
    }
    // 标题
    const h = line.match(/^(#{1,3})\s+(.*)$/);
    if (h) {
      const level = h[1]!.length as 1 | 2 | 3;
      out.push({ kind: "heading", level, text: h[2] ?? "" });
      i++;
      continue;
    }
    // 引用
    if (line.startsWith("> ") || line === ">") {
      const buf: string[] = [line.replace(/^>\s?/, "")];
      i++;
      while (
        i < lines.length &&
        (lines[i]!.startsWith("> ") || lines[i] === ">")
      ) {
        buf.push(lines[i]!.replace(/^>\s?/, ""));
        i++;
      }
      out.push({ kind: "quote", text: buf.join("\n") });
      continue;
    }
    // 无序列表
    if (/^[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i] ?? "")) {
        items.push((lines[i] ?? "").replace(/^[-*]\s+/, ""));
        i++;
      }
      out.push({ kind: "ul", items });
      continue;
    }
    // 有序列表
    if (/^\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i] ?? "")) {
        items.push((lines[i] ?? "").replace(/^\d+\.\s+/, ""));
        i++;
      }
      out.push({ kind: "ol", items });
      continue;
    }
    // 段落：聚合直到空行 / 块开始
    const buf: string[] = [line];
    i++;
    while (
      i < lines.length &&
      lines[i] &&
      lines[i]!.trim() &&
      !/^```/.test(lines[i] ?? "") &&
      !/^#{1,3}\s/.test(lines[i] ?? "") &&
      !/^[-*]\s+/.test(lines[i] ?? "") &&
      !/^\d+\.\s+/.test(lines[i] ?? "") &&
      !(lines[i] ?? "").startsWith("> ")
    ) {
      buf.push(lines[i] ?? "");
      i++;
    }
    out.push({ kind: "paragraph", text: buf.join("\n") });
  }
  return out;
}

function BlockNode({ block }: { block: Block }): React.ReactElement | null {
  const theme = resolveTheme(useStore((s) => s.theme));
  switch (block.kind) {
    case "blank":
      return <Box height={1} />;
    case "heading": {
      const color =
        block.level === 1
          ? theme.colors.primary
          : block.level === 2
            ? theme.colors.info
            : theme.colors.accent;
      return (
        <Box>
          <Text color={color} bold>
            {block.level === 1 ? "▌ " : block.level === 2 ? "◆ " : "• "}
            <Inline text={block.text} />
          </Text>
        </Box>
      );
    }
    case "paragraph":
      return (
        <Box>
          <Text color={theme.colors.text}>
            <Inline text={block.text} />
          </Text>
        </Box>
      );
    case "code":
      return (
        <Box
          flexDirection="column"
          paddingX={1}
          marginY={0}
          borderStyle="single"
          borderColor={theme.colors.border}
        >
          {block.lang && (
            <Text color={theme.colors.textDim} italic>
              {block.lang}
            </Text>
          )}
          <Text color={theme.colors.info}>{block.text || " "}</Text>
        </Box>
      );
    case "ul":
      return (
        <Box flexDirection="column">
          {block.items.map((t, i) => (
            <Text key={i} color={theme.colors.text}>
              <Text color={theme.colors.primary}>• </Text>
              <Inline text={t} />
            </Text>
          ))}
        </Box>
      );
    case "ol":
      return (
        <Box flexDirection="column">
          {block.items.map((t, i) => (
            <Text key={i} color={theme.colors.text}>
              <Text color={theme.colors.primary}>{i + 1}. </Text>
              <Inline text={t} />
            </Text>
          ))}
        </Box>
      );
    case "quote":
      return (
        <Box>
          <Text color={theme.colors.textDim} italic>
            {"│ "}
            <Inline text={block.text} />
          </Text>
        </Box>
      );
    default:
      return null;
  }
}

// ============================================================================
// Inline 解析
// ============================================================================

type Span =
  | { kind: "text"; text: string }
  | { kind: "bold"; text: string }
  | { kind: "italic"; text: string }
  | { kind: "code"; text: string }
  | { kind: "link"; text: string; url: string };

const INLINE_RE =
  /(\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|_[^_\n]+_|`[^`\n]+`|\[[^\]]+\]\([^)]+\))/g;

function parseInline(src: string): Span[] {
  const out: Span[] = [];
  let lastIdx = 0;
  for (const m of src.matchAll(INLINE_RE)) {
    const idx = m.index ?? 0;
    if (idx > lastIdx) {
      out.push({ kind: "text", text: src.slice(lastIdx, idx) });
    }
    const token = m[0];
    if (token.startsWith("**") || token.startsWith("__")) {
      out.push({ kind: "bold", text: token.slice(2, -2) });
    } else if (token.startsWith("`")) {
      out.push({ kind: "code", text: token.slice(1, -1) });
    } else if (token.startsWith("[")) {
      const mm = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      if (mm) out.push({ kind: "link", text: mm[1] ?? "", url: mm[2] ?? "" });
      else out.push({ kind: "text", text: token });
    } else if (token.startsWith("*") || token.startsWith("_")) {
      out.push({ kind: "italic", text: token.slice(1, -1) });
    } else {
      out.push({ kind: "text", text: token });
    }
    lastIdx = idx + token.length;
  }
  if (lastIdx < src.length) {
    out.push({ kind: "text", text: src.slice(lastIdx) });
  }
  return out;
}

function Inline({ text }: { text: string }): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const spans = React.useMemo(() => parseInline(text), [text]);
  return (
    <>
      {spans.map((s, i) => {
        switch (s.kind) {
          case "text":
            return <Text key={i}>{s.text}</Text>;
          case "bold":
            return (
              <Text key={i} bold>
                {s.text}
              </Text>
            );
          case "italic":
            return (
              <Text key={i} italic>
                {s.text}
              </Text>
            );
          case "code":
            return (
              <Text key={i} color={theme.colors.info} backgroundColor="blackBright">
                {` ${s.text} `}
              </Text>
            );
          case "link":
            return (
              <Text key={i} color={theme.colors.accent} underline>
                {s.text}
              </Text>
            );
        }
      })}
    </>
  );
}
