import { describe, expect, it } from "vitest";
import {
  looksLikeUnifiedDiff,
  parseUnifiedDiff,
} from "../utils/diff.js";

const SAMPLE = `--- a/file.txt
+++ b/file.txt
@@ -1,3 +1,3 @@
 line 1
-old line 2
+new line 2
 line 3`;

describe("utils/diff", () => {
  it("looksLikeUnifiedDiff 识别典型 diff", () => {
    expect(looksLikeUnifiedDiff(SAMPLE)).toBe(true);
  });

  it("looksLikeUnifiedDiff 排除纯代码/文本", () => {
    expect(looksLikeUnifiedDiff("hello world")).toBe(false);
    expect(looksLikeUnifiedDiff("function foo() {}")).toBe(false);
    expect(looksLikeUnifiedDiff("")).toBe(false);
  });

  it("parseUnifiedDiff 解析文件名与 hunk", () => {
    const parsed = parseUnifiedDiff(SAMPLE);
    expect(parsed.fileA).toBe("a/file.txt");
    expect(parsed.fileB).toBe("b/file.txt");
    expect(parsed.hunks).toHaveLength(1);
    const h = parsed.hunks[0]!;
    expect(h.header.startsWith("@@")).toBe(true);
    expect(h.lines.map((l) => l.kind)).toEqual([
      "context",
      "del",
      "add",
      "context",
    ]);
  });

  it("parseUnifiedDiff 处理多 hunk", () => {
    const two = SAMPLE + `\n@@ -10,1 +10,2 @@\n-removed\n+added1\n+added2`;
    const parsed = parseUnifiedDiff(two);
    expect(parsed.hunks).toHaveLength(2);
  });
});
