/**
 * 终端显示宽度工具。
 *
 * 是一份偏轻量的 wcwidth 近似实现，优先服务消息裁剪和表格列宽估算。
 */

/** 计算字符串的终端显示宽度。 */
export function displayWidth(s: string): number {
  let w = 0;
  for (const ch of s) {
    w += charWidth(ch);
  }
  return w;
}

/** 单字符显示宽度。 */
export function charWidth(ch: string): number {
  const cp = ch.codePointAt(0) ?? 0;
  if (cp < 0x20 || cp === 0x7f) return 0;
  if (isWide(cp)) return 2;
  return 1;
}

/** 按显示宽度截断字符串；超出部分用 `…` 替换。 */
export function truncateToWidth(s: string, maxWidth: number): string {
  if (maxWidth <= 0) return "";
  if (displayWidth(s) <= maxWidth) return s;
  let out = "";
  let w = 0;
  for (const ch of s) {
    const cw = charWidth(ch);
    if (w + cw > maxWidth - 1) break;
    out += ch;
    w += cw;
  }
  return out + "…";
}

/** 按显示宽度填充到 targetWidth。 */
export function padToWidth(
  s: string,
  targetWidth: number,
  align: "left" | "right" | "center" = "left",
): string {
  const w = displayWidth(s);
  const pad = Math.max(0, targetWidth - w);
  if (pad === 0) return s;
  if (align === "right") return " ".repeat(pad) + s;
  if (align === "center") {
    const left = Math.floor(pad / 2);
    return " ".repeat(left) + s + " ".repeat(pad - left);
  }
  return s + " ".repeat(pad);
}

/** 按 `maxWidth` 对每一行做硬换行，返回扁平行数组。 */
export function wrapToWidth(content: string, maxWidth: number): string[] {
  if (maxWidth <= 0) return [""];
  const rawLines = content.split(/\r?\n/);
  const out: string[] = [];
  for (const ln of rawLines) {
    if (ln === "") {
      out.push("");
      continue;
    }
    if (displayWidth(ln) <= maxWidth) {
      out.push(ln);
      continue;
    }
    let cur = "";
    let curW = 0;
    for (const ch of ln) {
      const cw = charWidth(ch);
      if (cw === 0) {
        // combining / 控制字符不占列
        cur += ch;
        continue;
      }
      if (curW + cw > maxWidth) {
        out.push(cur);
        cur = ch;
        curW = cw;
      } else {
        cur += ch;
        curW += cw;
      }
    }
    out.push(cur);
  }
  return out;
}

/** wrap 后截取 `[from, to)` 范围的行，并重新 join 成字符串。 */
export function sliceWrappedLines(
  content: string,
  maxWidth: number,
  from: number,
  to: number,
): string {
  const lines = wrapToWidth(content, maxWidth);
  const lo = Math.max(0, Math.min(lines.length, from));
  const hi = Math.max(lo, Math.min(lines.length, to));
  return lines.slice(lo, hi).join("\n");
}

function isWide(cp: number): boolean {
  return (
    (cp >= 0x1100 && cp <= 0x115f) || // Hangul Jamo init. consonants
    (cp >= 0x2e80 && cp <= 0x303e) || // CJK Radicals / Ideographic
    (cp >= 0x3041 && cp <= 0x33ff) || // Hiragana / Katakana / Bopomofo
    (cp >= 0x3400 && cp <= 0x4dbf) || // CJK Unified Ext A
    (cp >= 0x4e00 && cp <= 0x9fff) || // CJK Unified Ideographs
    (cp >= 0xa000 && cp <= 0xa4cf) || // Yi
    (cp >= 0xac00 && cp <= 0xd7a3) || // Hangul Syllables
    (cp >= 0xf900 && cp <= 0xfaff) || // CJK Compatibility Ideographs
    (cp >= 0xfe30 && cp <= 0xfe4f) || // CJK Compat Forms
    (cp >= 0xff00 && cp <= 0xff60) || // Fullwidth Forms
    (cp >= 0xffe0 && cp <= 0xffe6) ||
    (cp >= 0x1f300 && cp <= 0x1f64f) || // Misc Symbols + emoji
    (cp >= 0x1f900 && cp <= 0x1f9ff) ||
    (cp >= 0x20000 && cp <= 0x2fffd) || // CJK Unified Ext B-F
    (cp >= 0x30000 && cp <= 0x3fffd)
  );
}
