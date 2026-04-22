/**
 * 极简代码高亮：不引入 highlight.js / prism（体积大 + Ink 不方便上色）。
 *
 * 每种语言用一组正则标记 "关键字 / 字符串 / 数字 / 注释"，返回按起止位置
 * 的 token 序列；渲染器据此上色即可。
 *
 * 支持：
 *   - js/ts/tsx/jsx
 *   - py/python
 *   - sh/bash/zsh
 *   - json
 *   - go / rust（最小关键字集，复用通用规则）
 *
 * 未识别的语言返回单个 `{kind:"text"}` 覆盖全部内容（上层回退到默认色）。
 */

export type TokenKind =
  | "text"
  | "keyword"
  | "string"
  | "number"
  | "comment"
  | "builtin"
  | "type";

export interface Token {
  kind: TokenKind;
  text: string;
}

interface Rule {
  kind: TokenKind;
  /** `g` / `m` 默认添加 */
  pattern: RegExp;
}

const BASIC_NUM = /\b\d+(\.\d+)?(e[+-]?\d+)?\b/gi;
const BASIC_STR_DQ = /"(?:\\.|[^"\\])*"/g;
const BASIC_STR_SQ = /'(?:\\.|[^'\\])*'/g;
const BASIC_STR_BT = /`(?:\\.|[^`\\])*`/g;

const LANG_RULES: Record<string, Rule[]> = {
  js: [
    { kind: "comment", pattern: /\/\/[^\n]*|\/\*[\s\S]*?\*\//g },
    { kind: "string", pattern: BASIC_STR_DQ },
    { kind: "string", pattern: BASIC_STR_SQ },
    { kind: "string", pattern: BASIC_STR_BT },
    {
      kind: "keyword",
      pattern:
        /\b(const|let|var|function|return|if|else|for|while|do|switch|case|break|continue|class|extends|new|this|super|import|export|from|as|default|async|await|try|catch|finally|throw|typeof|instanceof|of|in|yield|void|delete)\b/g,
    },
    { kind: "builtin", pattern: /\b(true|false|null|undefined|NaN|Infinity)\b/g },
    { kind: "number", pattern: BASIC_NUM },
  ],
  ts: [], // 用 js 规则 + type 关键字（下面拼接）
  python: [
    { kind: "comment", pattern: /#[^\n]*/g },
    { kind: "string", pattern: BASIC_STR_DQ },
    { kind: "string", pattern: BASIC_STR_SQ },
    {
      kind: "string",
      pattern: /"""[\s\S]*?"""|'''[\s\S]*?'''/g,
    },
    {
      kind: "keyword",
      pattern:
        /\b(def|class|return|if|elif|else|for|while|break|continue|import|from|as|pass|with|try|except|finally|raise|yield|lambda|global|nonlocal|in|is|not|and|or|async|await)\b/g,
    },
    { kind: "builtin", pattern: /\b(True|False|None|self|cls)\b/g },
    { kind: "number", pattern: BASIC_NUM },
  ],
  bash: [
    { kind: "comment", pattern: /#[^\n]*/g },
    { kind: "string", pattern: BASIC_STR_DQ },
    { kind: "string", pattern: BASIC_STR_SQ },
    {
      kind: "keyword",
      pattern:
        /\b(if|then|else|elif|fi|for|in|do|done|while|until|case|esac|function|return|break|continue|export|source)\b/g,
    },
    { kind: "builtin", pattern: /\$[A-Za-z_][A-Za-z0-9_]*|\$\{[^}]+\}/g },
    { kind: "number", pattern: BASIC_NUM },
  ],
  json: [
    { kind: "string", pattern: BASIC_STR_DQ },
    { kind: "builtin", pattern: /\b(true|false|null)\b/g },
    { kind: "number", pattern: BASIC_NUM },
  ],
  go: [
    { kind: "comment", pattern: /\/\/[^\n]*|\/\*[\s\S]*?\*\//g },
    { kind: "string", pattern: BASIC_STR_DQ },
    { kind: "string", pattern: BASIC_STR_BT },
    {
      kind: "keyword",
      pattern:
        /\b(func|package|import|var|const|type|struct|interface|map|chan|go|defer|return|if|else|for|range|switch|case|default|break|continue|select|fallthrough|goto)\b/g,
    },
    { kind: "builtin", pattern: /\b(true|false|nil|iota)\b/g },
    { kind: "type", pattern: /\b(string|int|int8|int16|int32|int64|uint|uint8|uint16|uint32|uint64|float32|float64|bool|byte|rune|error|any)\b/g },
    { kind: "number", pattern: BASIC_NUM },
  ],
  rust: [
    { kind: "comment", pattern: /\/\/[^\n]*|\/\*[\s\S]*?\*\//g },
    { kind: "string", pattern: BASIC_STR_DQ },
    {
      kind: "keyword",
      pattern:
        /\b(fn|let|mut|const|static|struct|enum|trait|impl|pub|use|mod|crate|self|super|as|if|else|for|in|while|loop|match|return|break|continue|move|ref|where|async|await|unsafe|dyn|box)\b/g,
    },
    { kind: "builtin", pattern: /\b(true|false|None|Some|Ok|Err)\b/g },
    { kind: "type", pattern: /\b(i8|i16|i32|i64|i128|u8|u16|u32|u64|u128|usize|isize|f32|f64|bool|char|str|String|Vec|Option|Result)\b/g },
    { kind: "number", pattern: BASIC_NUM },
  ],
};

// ts = js + type 关键字
LANG_RULES.ts = [
  ...LANG_RULES.js!,
  {
    kind: "keyword",
    pattern: /\b(interface|type|enum|namespace|declare|readonly|keyof|infer|is|satisfies)\b/g,
  },
  { kind: "type", pattern: /\b(string|number|boolean|any|unknown|never|void|object|symbol|bigint|Array|Record|Partial|Readonly|Pick|Omit|Promise)\b/g },
];

/** 把语言名归一到内部表 key。 */
export function normalizeLang(lang: string): keyof typeof LANG_RULES | null {
  const l = lang.toLowerCase().trim();
  switch (l) {
    case "js":
    case "javascript":
    case "jsx":
    case "mjs":
    case "cjs":
      return "js";
    case "ts":
    case "typescript":
    case "tsx":
      return "ts";
    case "py":
    case "python":
      return "python";
    case "sh":
    case "bash":
    case "zsh":
    case "shell":
      return "bash";
    case "json":
      return "json";
    case "go":
    case "golang":
      return "go";
    case "rs":
    case "rust":
      return "rust";
    default:
      return null;
  }
}

/**
 * 对 `source` 按 `lang` 规则做 token 分片。
 *
 * 算法：把每条规则扫一遍得到 [start, end, kind] 区间；按 start 升序合并，
 * 重叠的优先选"先命中"的那条（注释/字符串优先级最高，写在规则数组的前面）。
 * 未覆盖的范围以 `text` 填回。
 */
export function tokenize(source: string, lang: string): Token[] {
  const key = normalizeLang(lang);
  if (!key) return [{ kind: "text", text: source }];
  const rules = LANG_RULES[key]!;

  interface Range {
    start: number;
    end: number;
    kind: TokenKind;
  }
  const ranges: Range[] = [];
  for (const rule of rules) {
    const re = new RegExp(rule.pattern.source, rule.pattern.flags);
    let m: RegExpExecArray | null;
    while ((m = re.exec(source)) !== null) {
      if (m[0].length === 0) {
        re.lastIndex++;
        continue;
      }
      ranges.push({ start: m.index, end: m.index + m[0].length, kind: rule.kind });
    }
  }
  ranges.sort((a, b) => a.start - b.start || b.end - a.end);

  const picked: Range[] = [];
  let lastEnd = 0;
  for (const r of ranges) {
    if (r.start < lastEnd) continue; // 与已选重叠就丢弃
    picked.push(r);
    lastEnd = r.end;
  }

  const out: Token[] = [];
  let cursor = 0;
  for (const r of picked) {
    if (r.start > cursor) out.push({ kind: "text", text: source.slice(cursor, r.start) });
    out.push({ kind: r.kind, text: source.slice(r.start, r.end) });
    cursor = r.end;
  }
  if (cursor < source.length) out.push({ kind: "text", text: source.slice(cursor) });
  return out;
}
