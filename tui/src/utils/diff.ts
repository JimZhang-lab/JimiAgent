import { createPatch, diffLines, type Change } from "diff";

/**
 * Diff 工具：
 *   - parseUnifiedDiff(text): 解析 unified diff 文本（--- / +++ / @@ 头部）
 *   - computeInlineDiff(a, b): 基于 diff.diffLines 生成行级 change 数组
 *   - makePatch(oldStr, newStr, fname): 生成 unified diff 文本（工具输出包装用）
 */

export interface DiffHunk {
  header: string;
  lines: DiffLine[];
}

export interface DiffLine {
  kind: "context" | "add" | "del" | "meta";
  text: string;
}

export interface ParsedDiff {
  fileA: string;
  fileB: string;
  hunks: DiffHunk[];
}

/**
 * 宽松解析 unified diff。输入不一定带 index/---/+++ 头部。
 */
export function parseUnifiedDiff(text: string): ParsedDiff {
  const lines = text.split(/\r?\n/);
  let fileA = "a";
  let fileB = "b";
  const hunks: DiffHunk[] = [];
  let cur: DiffHunk | null = null;

  for (const line of lines) {
    if (line.startsWith("--- ")) {
      fileA = line.slice(4).trim();
      continue;
    }
    if (line.startsWith("+++ ")) {
      fileB = line.slice(4).trim();
      continue;
    }
    if (line.startsWith("@@")) {
      if (cur) hunks.push(cur);
      cur = { header: line, lines: [] };
      continue;
    }
    if (!cur) continue;
    if (line.startsWith("+")) {
      cur.lines.push({ kind: "add", text: line.slice(1) });
    } else if (line.startsWith("-")) {
      cur.lines.push({ kind: "del", text: line.slice(1) });
    } else if (line.startsWith(" ")) {
      cur.lines.push({ kind: "context", text: line.slice(1) });
    } else if (line.startsWith("\\")) {
      cur.lines.push({ kind: "meta", text: line });
    } else {
      // 未知前缀当 context
      cur.lines.push({ kind: "context", text: line });
    }
  }
  if (cur) hunks.push(cur);
  return { fileA, fileB, hunks };
}

/**
 * 判断是否像 unified diff。简单启发：含 `@@ ` 和至少一行 +/- 开头。
 */
export function looksLikeUnifiedDiff(text: string): boolean {
  if (!text) return false;
  if (!/(^|\n)@@ /.test(text)) return false;
  return /(^|\n)[+-]/.test(text);
}

export function computeInlineDiff(a: string, b: string): Change[] {
  return diffLines(a, b);
}

export function makePatch(
  oldStr: string,
  newStr: string,
  fname = "file",
): string {
  return createPatch(fname, oldStr, newStr, "", "");
}
