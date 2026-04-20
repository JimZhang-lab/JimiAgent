import clipboardy from "clipboardy";
import { nextMessageId, useStore } from "../state/store.js";
import type { ThemeName } from "../themes/index.js";

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
      const help = [
        "**快捷键**",
        "- **Ctrl+C** 取消当前生成 / 再按一次退出",
        "- **Ctrl+P** 打开命令面板",
        "- **Ctrl+H** 打开会话列表",
        "- **Esc** 关闭弹层",
        "",
        "**客户端命令**",
        ...CLIENT_COMMANDS.map((c) => `- \`${c.name}\` — ${c.description}`),
      ].join("\n");
      useStore.getState().appendMessage({
        id: nextMessageId(),
        role: "system",
        content: help,
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
    description: "打开会话列表",
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
      const msgs = useStore.getState().messages;
      const last = [...msgs].reverse().find((m) => m.role === "assistant");
      if (!last) {
        useStore.getState().appendMessage({
          id: nextMessageId(),
          role: "system",
          content: "没有可复制的 assistant 消息。",
          createdAt: Date.now(),
        });
        return;
      }
      clipboardy
        .write(last.content)
        .then(() => {
          useStore.getState().appendMessage({
            id: nextMessageId(),
            role: "system",
            content: `已复制到剪贴板 (${last.content.length} 字符)`,
            createdAt: Date.now(),
          });
        })
        .catch((e: Error) => {
          useStore.getState().appendMessage({
            id: nextMessageId(),
            role: "error",
            content: `复制失败: ${e.message}`,
            createdAt: Date.now(),
          });
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
