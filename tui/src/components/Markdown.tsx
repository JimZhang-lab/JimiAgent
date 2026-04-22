import React from "react";
import { Box, Text } from "ink";
import { resolveTheme } from "../themes/index.js";
import { useStore } from "../state/store.js";
import {
  displayWidth,
  padToWidth,
  truncateToWidth,
} from "../utils/textWidth.js";
import { tokenize, normalizeLang, type TokenKind } from "../utils/highlight.js";

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

  // 流式中的消息每次追加 chunk 都会让 source 变化；如果每次都重新 parseBlocks
  // 会退化为 O(n) 每帧（对几 KB 内容累积 O(n²)）。流式是"预览态"，退化为纯
  // 文本 + 末尾光标即可，停止后再回到完整 Markdown parse。
  // useMemo 永远调用（符合 Hooks 规则），条件只写在 factory 里。
  const blocks = React.useMemo(
    () => (streaming ? null : parseBlocks(source)),
    [source, streaming],
  );

  if (!blocks) {
    return (
      <Box flexDirection="column">
        <Text color={theme.colors.text}>{source}</Text>
        <Text color={theme.colors.primary} dimColor>
          ▎
        </Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column">
      {blocks.map((b, i) => (
        <BlockNode key={i} block={b} />
      ))}
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
  | { kind: "table"; header: string[]; align: TableAlign[]; rows: string[][] }
  | { kind: "blank" };

type TableAlign = "left" | "right" | "center";

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
    // 表格：header 行 + 分隔行（必须同时出现，否则按普通段落处理）
    if (line.includes("|") && isTableSeparator(lines[i + 1] ?? "")) {
      const header = splitTableRow(line);
      const align = parseTableAlign(lines[i + 1] ?? "", header.length);
      i += 2;
      const rows: string[][] = [];
      while (
        i < lines.length &&
        lines[i] &&
        lines[i]!.includes("|") &&
        !/^```/.test(lines[i] ?? "")
      ) {
        const row = splitTableRow(lines[i] ?? "");
        // 列数对齐到 header 长度：不足补空、超出截断
        while (row.length < header.length) row.push("");
        rows.push(row.slice(0, header.length));
        i++;
      }
      out.push({ kind: "table", header, align, rows });
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
      !(lines[i] ?? "").startsWith("> ") &&
      // 下一行如果构成 "header | sep" 表格起点，段落也要结束
      !(
        (lines[i] ?? "").includes("|") &&
        isTableSeparator(lines[i + 1] ?? "")
      )
    ) {
      buf.push(lines[i] ?? "");
      i++;
    }
    out.push({ kind: "paragraph", text: buf.join("\n") });
  }
  return out;
}

/** 判断是否 Markdown 表格分隔行（` | :--- | :---: | ---: | ` 之类）。 */
function isTableSeparator(line: string): boolean {
  if (!line || !line.includes("|")) return false;
  const cells = line.split("|").map((c) => c.trim()).filter((c) => c !== "");
  if (cells.length === 0) return false;
  return cells.every((c) => /^:?-{3,}:?$/.test(c));
}

/** 切分一行表格：忽略两端的 `|`，用 `|` 分割剩余。 */
function splitTableRow(line: string): string[] {
  return line
    .replace(/^\s*\|/, "")
    .replace(/\|\s*$/, "")
    .split("|")
    .map((c) => c.trim());
}

/** 从分隔行解析每列对齐。长度不够时补 left。 */
function parseTableAlign(sepLine: string, n: number): TableAlign[] {
  const cells = sepLine
    .replace(/^\s*\|/, "")
    .replace(/\|\s*$/, "")
    .split("|")
    .map((c) => c.trim());
  const out: TableAlign[] = [];
  for (let i = 0; i < n; i++) {
    const c = cells[i] ?? "";
    if (c.startsWith(":") && c.endsWith(":")) out.push("center");
    else if (c.endsWith(":")) out.push("right");
    else out.push("left");
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
      return <CodeBlock lang={block.lang} text={block.text} />;
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
    case "table":
      return <TableBlock block={block} />;
    default:
      return null;
  }
}

/**
 * 代码块渲染：按语言做 token 级着色。
 *
 * 未识别语言或无 fence 信息时回退为整体 info 色（保留原行为）。
 * 顶部显示语言 tag（若有）。
 */
function CodeBlock({
  lang,
  text,
}: {
  lang: string;
  text: string;
}): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const normalized = normalizeLang(lang);
  const tokens = React.useMemo(
    () => (normalized ? tokenize(text, lang) : null),
    [normalized, lang, text],
  );

  const colorFor = (k: TokenKind): string => {
    switch (k) {
      case "keyword":
        return theme.colors.primary;
      case "string":
        return theme.colors.success;
      case "number":
        return theme.colors.accent;
      case "comment":
        return theme.colors.textDim;
      case "builtin":
        return theme.colors.warning;
      case "type":
        return theme.colors.info;
      default:
        return theme.colors.text;
    }
  };

  return (
    <Box
      flexDirection="column"
      paddingX={1}
      marginY={0}
      borderStyle="single"
      borderColor={theme.colors.border}
    >
      {lang && (
        <Text color={theme.colors.textDim} italic>
          {lang}
        </Text>
      )}
      {tokens ? (
        <Text>
          {tokens.map((tok, i) => (
            <Text
              key={i}
              color={colorFor(tok.kind)}
              italic={tok.kind === "comment"}
              bold={tok.kind === "keyword"}
            >
              {tok.text}
            </Text>
          ))}
        </Text>
      ) : (
        <Text color={theme.colors.info}>{text || " "}</Text>
      )}
    </Box>
  );
}

/**
 * 表格渲染：用等宽字符绘制边框。
 *
 * 列宽取 header 与每行对应列的显示宽度最大值；若总宽超过终端，尾部整体
 * 截断（避免自动 wrap 把单格劈开）。
 */
function TableBlock({
  block,
}: {
  block: Extract<Block, { kind: "table" }>;
}): React.ReactElement {
  const theme = resolveTheme(useStore((s) => s.theme));
  const cols = useStore((s) => s.dims.cols);
  const { header, align, rows } = block;

  // 先算每列显示宽度
  const widths: number[] = header.map((h) => displayWidth(h));
  for (const r of rows) {
    for (let i = 0; i < header.length; i++) {
      const w = displayWidth(r[i] ?? "");
      if (w > widths[i]!) widths[i] = w;
    }
  }
  // 限制总宽不超过屏幕 cols - 4（留边距）。
  // 平均收敛：每次最长列 -1，直到总宽（含分隔符）<= 允许宽。
  const maxRow = Math.max(20, cols - 4);
  // 总宽 = 1(左|) + sum(宽+2(两侧空格)+1(分隔|))
  const rowWidth = () => 1 + widths.reduce((s, w) => s + w + 3, 0);
  let guard = 200;
  while (rowWidth() > maxRow && guard-- > 0) {
    // 找最长列削 1 字宽
    let maxI = 0;
    for (let i = 1; i < widths.length; i++) {
      if (widths[i]! > widths[maxI]!) maxI = i;
    }
    if (widths[maxI]! <= 3) break;
    widths[maxI] = widths[maxI]! - 1;
  }

  const sep =
    "+" + widths.map((w) => "-".repeat(w + 2)).join("+") + "+";

  const renderRow = (cells: string[], isHeader: boolean) => (
    <Text color={isHeader ? theme.colors.primary : theme.colors.text} bold={isHeader}>
      {"|"}
      {widths.map((w, i) => {
        const cell = truncateToWidth(cells[i] ?? "", w);
        const a = align[i] ?? "left";
        return ` ${padToWidth(cell, w, a)} |`;
      }).join("")}
    </Text>
  );

  return (
    <Box flexDirection="column">
      <Text color={theme.colors.border}>{sep}</Text>
      {renderRow(header, true)}
      <Text color={theme.colors.border}>{sep}</Text>
      {rows.map((r, i) => (
        <React.Fragment key={i}>{renderRow(r, false)}</React.Fragment>
      ))}
      <Text color={theme.colors.border}>{sep}</Text>
    </Box>
  );
}

// 显示宽度 / 截断 / 对齐工具抽到 utils/textWidth.ts 共用（见文件顶部 import）

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
