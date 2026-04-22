import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";
import { spawn } from "node:child_process";
import clipboardy from "clipboardy";
import { nextMessageId, useStore, type Message } from "../state/store.js";
import type { ThemeName } from "../themes/index.js";
import { groupByScope } from "../keybindings/registry.js";

/**
 * 客户端斜杠命令定义。区别于服务端 commands（/vim、/summary 等交给 agent 处理）。
 *
 * 客户端命令只改动 Node 端 UI 状态，不发送给 Python。
 */
export interface SlashCommandCtx {
  exit: () => void;
  newSession: (title?: string) => void;
  refreshSessions: () => void;
  openPalette: () => void;
  openHistory: () => void;
  openMemory: () => void;
}

export interface ClientSlashCommand {
  name: string;
  description: string;
  aliases?: string[];
  run(arg: string, ctx: SlashCommandCtx): void;
}

export const CLIENT_COMMANDS: ClientSlashCommand[] = [
  {
    name: "/quit",
    aliases: ["/exit"],
    description: "退出 TUI",
    run: (_a, ctx) => ctx.exit(),
  },
  {
    name: "/clear",
    description: "清空当前对话的显示（不删除会话）",
    run: () => useStore.getState().clearMessages(),
  },
  {
    name: "/help",
    description: "显示帮助",
    run: () => {
      // 快捷键段由 keybindings registry 实时投影，保证文档与实际绑定同步。
      const grouped = groupByScope();
      const globalKeys = grouped.global ?? [];
      const selectionKeys = grouped.selection ?? [];

      const lines: string[] = ["## 快捷键"];
      if (globalKeys.length) {
        for (const k of globalKeys) {
          lines.push(`- \`${k.key}\` — ${k.description}`);
        }
      }
      if (selectionKeys.length) {
        lines.push("", "### 消息选区（Visual 模式）");
        for (const k of selectionKeys) {
          lines.push(`- \`${k.key}\` — ${k.description}`);
        }
      }
      lines.push(
        "",
        "## 客户端命令",
        ...CLIENT_COMMANDS.map(
          (c) =>
            `- \`${c.name}\`${
              c.aliases?.length ? ` (别名: ${c.aliases.join(", ")})` : ""
            } — ${c.description}`,
        ),
        "",
        "_服务端命令请看 `/palette` 或 `/keys`。_",
      );
      useStore.getState().appendMessage({
        id: nextMessageId(),
        role: "system",
        content: lines.join("\n"),
        createdAt: Date.now(),
      });
    },
  },
  {
    name: "/new",
    description: "新建会话 (/new [标题])",
    run: (arg, ctx) => ctx.newSession(arg.trim() || undefined),
  },
  {
    name: "/sessions",
    aliases: ["/history"],
    description: "打开会话列表（Enter 切换会话后将回放历史到终端 scrollback）",
    run: (_a, ctx) => ctx.openHistory(),
  },
  {
    name: "/memory",
    aliases: ["/mem"],
    description: "打开记忆管理面板（浏览 / 搜索 / 删除）",
    run: (_a, ctx) => ctx.openMemory(),
  },
  {
    name: "/palette",
    description: "打开命令面板",
    run: (_a, ctx) => ctx.openPalette(),
  },
  {
    name: "/theme",
    description: "切换主题 (/theme dark|light|auto)",
    run: (arg) => {
      const t = arg.trim().toLowerCase();
      if (t === "dark" || t === "light" || t === "auto") {
        useStore.getState().setTheme(t as ThemeName);
      } else {
        useStore.getState().appendMessage({
          id: nextMessageId(),
          role: "system",
          content: "用法: `/theme dark|light|auto`",
          createdAt: Date.now(),
        });
      }
    },
  },
  {
    name: "/copy",
    description: "复制最后一条 assistant 消息到系统剪贴板",
    run: () => {
      const state = useStore.getState();
      const msgs = state.messages;
      const last = [...msgs].reverse().find((m) => m.role === "assistant");
      if (!last) {
        state.appendMessage({
          id: nextMessageId(),
          role: "system",
          content: "没有可复制的 assistant 消息。",
          createdAt: Date.now(),
        });
        return;
      }
      // 流式中的消息带 streaming 标记，复制会拿到截断内容。
      // 明确告诉用户现在复制的是不完整版本，让其决定要不要等生成完。
      const partialHint = last.streaming || state.streaming
        ? "（⚠ 当前仍在流式生成，复制为不完整内容）"
        : "";
      clipboardy
        .write(last.content)
        .then(() => {
          state.appendMessage({
            id: nextMessageId(),
            role: "system",
            content: `已复制到剪贴板 (${last.content.length} 字符) ${partialHint}`.trim(),
            createdAt: Date.now(),
          });
        })
        .catch((e: Error) => {
          state.appendMessage({
            id: nextMessageId(),
            role: "error",
            content: `复制失败: ${e.message}`,
            createdAt: Date.now(),
          });
        });
    },
  },
  {
    name: "/open",
    description: "打开消息中的链接 (/open [n]，n=第几个 URL，默认 1)",
    run: (arg) => {
      const state = useStore.getState();
      const msgs = state.messages;
      const urls = collectUrls(msgs);
      if (urls.length === 0) {
        state.appendMessage({
          id: nextMessageId(),
          role: "system",
          content: "当前会话里没有可识别的 URL。",
          createdAt: Date.now(),
        });
        return;
      }
      const rawN = arg.trim();
      // 无参数：URL 只有一个时直接打开；多个时先列出让用户选
      if (rawN === "") {
        if (urls.length > 1) {
          const listing = urls
            .map((u, i) => `- [${i + 1}] ${u}`)
            .slice(0, 20)
            .join("\n");
          state.appendMessage({
            id: nextMessageId(),
            role: "system",
            content: `共发现 ${urls.length} 个链接：\n${listing}\n\n用 \`/open <n>\` 打开第 n 个。`,
            createdAt: Date.now(),
          });
          return;
        }
        // 只有 1 个 → 打开
      }
      const n = rawN === "" ? 1 : Number(rawN);
      if (!Number.isInteger(n) || n < 1 || n > urls.length) {
        state.appendMessage({
          id: nextMessageId(),
          role: "error",
          content: `非法索引 \`${rawN}\`，有效范围 1-${urls.length}。`,
          createdAt: Date.now(),
        });
        return;
      }
      const url = urls[n - 1]!;
      openExternal(url).then(
        () =>
          state.appendMessage({
            id: nextMessageId(),
            role: "system",
            content: `✓ 已请求系统打开：${url}`,
            createdAt: Date.now(),
          }),
        (e: Error) =>
          state.appendMessage({
            id: nextMessageId(),
            role: "error",
            content: `打开失败：${e.message}`,
            createdAt: Date.now(),
          }),
      );
    },
  },
  {
    name: "/cwd",
    description: "显示 TUI 工作目录与用户 cwd（切换需重启 jimi chat）",
    run: () => {
      const state = useStore.getState();
      const workspace = state.workspaceCwd || "(未设置)";
      // 用户 cwd 在启动时通过 JIMI_USER_CWD 环境变量传入（见 cli.py）
      const userCwd =
        process.env.JIMI_USER_CWD ||
        process.env.PWD ||
        process.cwd();
      const lines = [
        "## 工作目录",
        `- **TUI 启动目录 (workspace)**: \`${workspace}\``,
        `- **用户原始 cwd (agent 执行基)**: \`${userCwd}\``,
        "",
        "_切换目录：退出后在目标目录重新执行 `jimi chat`。_",
      ];
      state.appendMessage({
        id: nextMessageId(),
        role: "system",
        content: lines.join("\n"),
        createdAt: Date.now(),
      });
    },
  },
  {
    name: "/export",
    aliases: ["/save"],
    description: "导出当前会话为 Markdown 文件 (/export [path])",
    run: (arg) => {
      const state = useStore.getState();
      const msgs = state.messages;
      if (msgs.length === 0) {
        state.appendMessage({
          id: nextMessageId(),
          role: "system",
          content: "当前会话没有消息可导出。",
          createdAt: Date.now(),
        });
        return;
      }
      const session = state.currentSessionId
        ? state.sessions.find((x) => x.id === state.currentSessionId) ?? null
        : null;
      const target = resolveExportPath(arg.trim(), session?.title ?? "session");
      const md = renderSessionMarkdown(msgs, session);
      try {
        fs.mkdirSync(path.dirname(target), { recursive: true });
        fs.writeFileSync(target, md, "utf8");
        state.appendMessage({
          id: nextMessageId(),
          role: "system",
          content: `✓ 已导出到 \`${shortenHome(target)}\`（${msgs.length} 条 · ${md.length} 字符）`,
          createdAt: Date.now(),
        });
      } catch (e) {
        state.appendMessage({
          id: nextMessageId(),
          role: "error",
          content: `导出失败：${(e as Error).message}`,
          createdAt: Date.now(),
        });
      }
    },
  },
  {
    name: "/keys",
    description: "显示所有已注册键位（按 scope 分组）",
    run: () => {
      const grouped = groupByScope();
      const scopeOrder = [
        "global",
        "prompt",
        "selection",
        "vim-normal",
        "overlay-palette",
        "overlay-history",
        "overlay-memory",
      ];
      const lines: string[] = ["## 键位清单"];
      const seen = new Set<string>();
      for (const s of scopeOrder) {
        const bs = grouped[s];
        if (!bs || bs.length === 0) continue;
        lines.push(`\n### ${s}`);
        for (const b of bs) lines.push(`- \`${b.key}\` — ${b.description}`);
        seen.add(s);
      }
      // 补齐未在已知顺序里的 scope
      for (const [s, bs] of Object.entries(grouped)) {
        if (seen.has(s) || !bs) continue;
        lines.push(`\n### ${s}`);
        for (const b of bs) lines.push(`- \`${b.key}\` — ${b.description}`);
      }
      if (lines.length === 1) lines.push("（暂无已注册键位）");
      useStore.getState().appendMessage({
        id: nextMessageId(),
        role: "system",
        content: lines.join("\n"),
        createdAt: Date.now(),
      });
    },
  },
  {
    name: "/vim",
    description: "切换 Vim 模式（/vim on|off，默认 toggle）",
    run: (arg) => {
      const t = arg.trim().toLowerCase();
      const cur = useStore.getState().vim;
      let nextEnabled: boolean;
      if (t === "on") nextEnabled = true;
      else if (t === "off") nextEnabled = false;
      else nextEnabled = !cur.enabled;
      useStore.getState().setVim({
        enabled: nextEnabled,
        mode: nextEnabled ? "normal" : "insert",
      });
      useStore.getState().appendMessage({
        id: nextMessageId(),
        role: "system",
        content: `Vim 模式已 ${nextEnabled ? "启用 (normal)" : "关闭"}`,
        createdAt: Date.now(),
      });
    },
  },
];

// ============================================================================
// /export 辅助函数
// ============================================================================

/**
 * 解析 /export 的 path 参数。
 *
 * 规则：
 *   - 无参数：导出到 `~/.jimiagent/exports/<slug>-<yyyymmdd-hhmmss>.md`
 *   - 以 `~/` 开头：展开用户目录
 *   - 相对路径：以 process.cwd() 为基
 *   - 以 `/` 开头：绝对路径
 *   - 目标若是已存在的目录：在其下生成 `<slug>-<timestamp>.md`
 */
function resolveExportPath(raw: string, title: string): string {
  let p = raw;
  if (!p) {
    const dir = path.join(os.homedir(), ".jimiagent", "exports");
    return path.join(dir, `${slugify(title)}-${timestamp()}.md`);
  }
  if (p.startsWith("~/")) p = path.join(os.homedir(), p.slice(2));
  else if (!path.isAbsolute(p)) p = path.resolve(process.cwd(), p);
  // 若是已存在的目录，落在其下；若以 / 结尾，也按目录处理
  let isDir = false;
  try {
    isDir = fs.statSync(p).isDirectory();
  } catch {
    isDir = p.endsWith("/");
  }
  if (isDir) {
    return path.join(p, `${slugify(title)}-${timestamp()}.md`);
  }
  return p.endsWith(".md") ? p : `${p}.md`;
}

/** 把 Message[] 渲染成 Markdown。role 用中文标签，时间附在子标题。 */
function renderSessionMarkdown(
  msgs: readonly Message[],
  session: { title: string; id: string } | null,
): string {
  const labelOf: Record<string, string> = {
    user: "用户",
    assistant: "助手",
    tool: "工具",
    error: "错误",
    system: "系统",
    confirm: "确认",
  };
  const now = new Date();
  const header = [
    `# ${session?.title ?? "会话"}`,
    "",
    `- **导出时间**: ${now.toISOString()}`,
    session ? `- **会话 id**: \`${session.id}\`` : "",
    `- **消息数**: ${msgs.length}`,
    "",
    "---",
    "",
  ]
    .filter((x) => x !== "")
    .join("\n");
  const body = msgs
    .map((m) => {
      const label = labelOf[m.role] ?? m.role;
      const ts = m.createdAt ? ` _(${new Date(m.createdAt).toISOString()})_` : "";
      const extra = m.role === "tool" && m.toolName ? ` · \`${m.toolName}\`` : "";
      return `## ${label}${extra}${ts}\n\n${m.content.trim()}\n`;
    })
    .join("\n");
  return `${header}\n${body}\n`;
}

function slugify(s: string): string {
  const base = s
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40);
  return base || "session";
}

function timestamp(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-` +
    `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`
  );
}

function shortenHome(p: string): string {
  const home = os.homedir();
  if (home && p.startsWith(home)) return "~" + p.slice(home.length);
  return p;
}

// ============================================================================
// /open 辅助函数
// ============================================================================

const URL_RE = /https?:\/\/[^\s<>"'`）)\]}，。；,;]+/g;

/** 按出现顺序去重收集会话里所有 http(s) URL。 */
function collectUrls(msgs: readonly Message[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const m of msgs) {
    const matches = m.content.match(URL_RE);
    if (!matches) continue;
    for (const u of matches) {
      // 剥掉 Markdown link 末尾可能残留的括号/标点
      const clean = u.replace(/[.,;:!?)\]]+$/, "");
      if (!seen.has(clean)) {
        seen.add(clean);
        out.push(clean);
      }
    }
  }
  return out;
}

/**
 * 调用系统默认浏览器打开 URL。
 *
 * - macOS: `open <url>`
 * - Linux: `xdg-open <url>`
 * - Windows: `cmd /c start "" <url>`
 *
 * 注意不要 `shell: true` 拼接字符串，避免 URL 里的特殊字符带来命令注入。
 */
function openExternal(url: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const platform = process.platform;
    let cmd: string;
    let args: string[];
    if (platform === "darwin") {
      cmd = "open";
      args = [url];
    } else if (platform === "win32") {
      cmd = "cmd";
      args = ["/c", "start", "", url];
    } else {
      cmd = "xdg-open";
      args = [url];
    }
    try {
      const child = spawn(cmd, args, {
        stdio: "ignore",
        detached: true,
      });
      child.on("error", reject);
      child.unref();
      resolve();
    } catch (e) {
      reject(e as Error);
    }
  });
}

/**
 * 判断一段文本是否为客户端命令；若是，执行并返回 true。
 * 否则返回 false 交给调用方继续发给 agent。
 */
export function tryRunClientCommand(
  text: string,
  ctx: SlashCommandCtx,
): boolean {
  if (!text.startsWith("/")) return false;
  const space = text.indexOf(" ");
  const cmd = (space >= 0 ? text.slice(0, space) : text).toLowerCase();
  const arg = space >= 0 ? text.slice(space + 1) : "";

  for (const c of CLIENT_COMMANDS) {
    if (c.name === cmd || (c.aliases ?? []).includes(cmd)) {
      c.run(arg, ctx);
      return true;
    }
  }
  return false;
}
