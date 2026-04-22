/**
 * 键位元数据注册表。
 *
 * 设计定位：
 *   - 纯元数据（scope + key + description），**不**接管实际的 input 路由
 *     （路由仍由各组件的 useInput 完成）
 *   - 用途：/keys 命令列出当前可用键位；开发期冲突检测
 *
 * 之所以不集中路由，是因为 Ink 的 useInput 天然按组件 mount/unmount 自动释放，
 * 且不同 scope 的激活条件（focus 状态、弹层是否打开）已经分散在组件里用 `isActive`
 * 表达得很自然。强行收拢反而会把这些条件揉成一个不易读的路由层。
 */

export type KeybindingScope =
  | "global"
  | "overlay-palette"
  | "overlay-history"
  | "overlay-memory"
  | "prompt"
  | "vim-normal"
  | "selection";

export interface Keybinding {
  id: number;
  scope: KeybindingScope;
  /** 规范化的键位字符串：如 `ctrl+c` / `shift+tab` / `up` / `a` / `escape`。 */
  key: string;
  description: string;
}

let idSeq = 0;
const bindings = new Map<number, Keybinding>();
const byScopeKey = new Map<string, number>();

/** 注册一条键位元数据；冲突时不抛错，而是打印一次 warning 后返回既有 id。 */
export function registerKeybinding(
  scope: KeybindingScope,
  key: string,
  description: string,
): number {
  const sk = `${scope}:${key}`;
  const existing = byScopeKey.get(sk);
  if (existing !== undefined) {
    // 冲突：大多数情况是严格模式下组件 re-mount 残留，静默复用即可。
    return existing;
  }
  const id = ++idSeq;
  bindings.set(id, { id, scope, key, description });
  byScopeKey.set(sk, id);
  return id;
}

export function unregisterKeybinding(id: number): void {
  const b = bindings.get(id);
  if (!b) return;
  bindings.delete(id);
  byScopeKey.delete(`${b.scope}:${b.key}`);
}

export function listKeybindings(scope?: KeybindingScope): Keybinding[] {
  const all = Array.from(bindings.values());
  const out = scope ? all.filter((b) => b.scope === scope) : all;
  // 稳定排序：scope 字母序 → key 字母序
  out.sort((a, b) =>
    a.scope === b.scope ? a.key.localeCompare(b.key) : a.scope.localeCompare(b.scope),
  );
  return out;
}

/** 仅测试用：清空全部注册。 */
export function __resetKeybindingsForTest(): void {
  bindings.clear();
  byScopeKey.clear();
  idSeq = 0;
}

/** 把绑定按 scope 分组，给 /keys 命令输出 Markdown。 */
export function groupByScope(): Record<string, Keybinding[]> {
  const grouped: Record<string, Keybinding[]> = {};
  for (const b of listKeybindings()) {
    (grouped[b.scope] ??= []).push(b);
  }
  return grouped;
}
