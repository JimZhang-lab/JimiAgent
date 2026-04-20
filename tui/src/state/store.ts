import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { subscribeWithSelector } from "zustand/middleware";
import type { AgentTransport } from "../transport/AgentTransport.js";
import type {
  AgentInfo,
  ConfirmPayload,
  MemoryMeta,
  SessionMeta,
} from "../protocol/events.js";
import type { ThemeName } from "../themes/index.js";

// ============================================================================
// Message 领域模型（TUI 内部）
// ============================================================================

export type MessageRole =
  | "user"
  | "assistant"
  | "tool"
  | "system"
  | "error"
  | "confirm";

export interface Message {
  /** 客户端侧递增 id（不走网络） */
  id: string;
  role: MessageRole;
  /** 渲染内容；assistant 用 markdown，其他是纯文本或 JSON */
  content: string;
  /** assistant 专属：是否仍在流式增量中 */
  streaming?: boolean;
  /** tool 专属：工具名 */
  toolName?: string;
  /** confirm 专属：interrupt 元信息 */
  confirm?: ConfirmPayload;
  /** 创建时间戳（毫秒） */
  createdAt: number;
}

// ============================================================================
// Store 状态
// ============================================================================

/**
 * 动作追踪：一次 agent 调用的"当前正在做什么"快照。
 *
 * 由 `tool` / `text` / `done` / `error` 事件推导：
 *   - 收到 `tool` 时 kind=tool，label=工具名
 *   - 收到第一条 `text` 且当前 kind!==thinking 时 kind=thinking
 *   - 收到 `done`/`error` 时清零
 */
export interface Activity {
  kind: "thinking" | "tool" | "confirm";
  label: string;
  since: number;
}

/**
 * 活动历史：顺序记录本轮对话的动作，供 Timeline 展示（右侧面板预留）。
 * 出于内存成本考虑，最多保留最近 200 条。
 */
export interface ActivityEntry {
  kind: "tool" | "text" | "confirm" | "error" | "done";
  label: string;
  at: number;
  durationMs?: number;
}

export interface Store {
  // 启动 / 连接
  transport: AgentTransport | null;
  connected: boolean;
  agentInfo: AgentInfo | null;
  bootError: string | null;

  // 会话
  currentSessionId: string | null;
  sessions: SessionMeta[];

  // 消息
  messages: Message[];
  streaming: boolean;
  pendingConfirm: ConfirmPayload | null;

  // 动作追踪
  activity: Activity | null;
  activityLog: ActivityEntry[];

  // 消息滚动
  viewOffset: number; // 0 = 停在最新；>0 = 向上滚了多少条

  // 输入框当前 draft：lift 到 store 是为了让 MainLayout 能根据 draft 的前缀
  // 同步决定 SlashSuggestions 是否显示、从而把高度纳入 flex 预算。
  promptDraft: string;

  // 记忆面板
  memories: MemoryMeta[];
  /** 最近一次查询字符串；空串 = 浏览最近记忆（list_recent） */
  memoryQuery: string;
  memoryLoading: boolean;
  /** 多选集合（id 数组；保持排序便于稳定比对） */
  memorySelection: number[];

  // UI
  focus: FocusTarget;
  paletteOpen: boolean;
  historyOpen: boolean;
  memoryOpen: boolean;
  theme: ThemeName;
  vim: { enabled: boolean; mode: VimMode };
  dims: { cols: number; rows: number };

  // Actions
  setTransport(t: AgentTransport): void;
  setConnected(v: boolean): void;
  setAgentInfo(info: AgentInfo): void;
  setBootError(msg: string | null): void;

  setCurrentSession(id: string | null): void;
  setSessions(list: SessionMeta[]): void;

  appendMessage(msg: Message): void;
  appendAssistantChunk(chunk: string): void;
  finishAssistantStreaming(): void;
  setPendingConfirm(p: ConfirmPayload | null): void;
  setStreaming(v: boolean): void;
  clearMessages(): void;

  beginActivity(kind: Activity["kind"], label: string): void;
  endActivity(outcome?: "done" | "error", error?: string): void;

  setViewOffset(n: number): void;
  /** 相对调整 viewOffset；会在消息总数范围内裁剪。 */
  scrollBy(delta: number): void;
  scrollToBottom(): void;
  scrollToTop(): void;

  setPromptDraft(v: string): void;

  setMemories(items: MemoryMeta[], query?: string): void;
  removeMemoryLocal(id: number): void;
  setMemoryQuery(v: string): void;
  setMemoryLoading(v: boolean): void;
  setMemoryOpen(v: boolean): void;
  /** 切换单个 id 的选中。 */
  toggleMemorySelection(id: number): void;
  /** 批量写入选择（用于 a 全选）。 */
  setMemorySelection(ids: number[]): void;
  /** 清空选择。 */
  clearMemorySelection(): void;

  setFocus(f: FocusTarget): void;
  setPaletteOpen(v: boolean): void;
  setHistoryOpen(v: boolean): void;
  setTheme(t: ThemeName): void;
  setVim(next: Partial<{ enabled: boolean; mode: VimMode }>): void;
  setDims(cols: number, rows: number): void;
}

export type FocusTarget =
  | "prompt"
  | "history"
  | "palette"
  | "memory"
  | "messages";
export type VimMode = "normal" | "insert" | "visual";

// ============================================================================
// Store 实现
// ============================================================================

let msgIdSeq = 0;
export function nextMessageId(): string {
  msgIdSeq += 1;
  return `m${Date.now()}-${msgIdSeq}`;
}

export const useStore = create<Store>()(
  subscribeWithSelector(
    immer((set) => ({
      transport: null,
      connected: false,
      agentInfo: null,
      bootError: null,

      currentSessionId: null,
      sessions: [],

      messages: [],
      streaming: false,
      pendingConfirm: null,

      activity: null,
      activityLog: [],

      viewOffset: 0,

      promptDraft: "",

      memories: [],
      memoryQuery: "",
      memoryLoading: false,
      memorySelection: [],

      focus: "prompt",
      paletteOpen: false,
      historyOpen: false,
      memoryOpen: false,
      theme: "auto",
      vim: { enabled: false, mode: "insert" },
      dims: { cols: 80, rows: 24 },

      setTransport: (t) =>
        set((s) => {
          s.transport = t;
        }),
      setConnected: (v) =>
        set((s) => {
          s.connected = v;
        }),
      setAgentInfo: (info) =>
        set((s) => {
          s.agentInfo = info;
        }),
      setBootError: (msg) =>
        set((s) => {
          s.bootError = msg;
        }),

      setCurrentSession: (id) =>
        set((s) => {
          s.currentSessionId = id;
        }),
      setSessions: (list) =>
        set((s) => {
          s.sessions = list;
        }),

      appendMessage: (msg) =>
        set((s) => {
          s.messages.push(msg);
        }),

      appendAssistantChunk: (chunk) =>
        set((s) => {
          const last = s.messages[s.messages.length - 1];
          if (last && last.role === "assistant" && last.streaming) {
            last.content += chunk;
          } else {
            s.messages.push({
              id: nextMessageId(),
              role: "assistant",
              content: chunk,
              streaming: true,
              createdAt: Date.now(),
            });
          }
          s.streaming = true;
        }),

      finishAssistantStreaming: () =>
        set((s) => {
          const last = s.messages[s.messages.length - 1];
          if (last && last.role === "assistant" && last.streaming) {
            last.streaming = false;
          }
          s.streaming = false;
        }),

      setPendingConfirm: (p) =>
        set((s) => {
          s.pendingConfirm = p;
        }),

      setStreaming: (v) =>
        set((s) => {
          s.streaming = v;
        }),

      clearMessages: () =>
        set((s) => {
          s.messages = [];
          s.streaming = false;
          s.pendingConfirm = null;
          s.activity = null;
          s.activityLog = [];
          s.viewOffset = 0;
        }),

      beginActivity: (kind, label) =>
        set((s) => {
          // 若已有活动：把上一条推入 log（记录 duration）
          if (s.activity) {
            s.activityLog.push({
              kind: s.activity.kind === "thinking" ? "text" : s.activity.kind,
              label: s.activity.label,
              at: s.activity.since,
              durationMs: Date.now() - s.activity.since,
            });
            if (s.activityLog.length > 200) {
              s.activityLog.splice(0, s.activityLog.length - 200);
            }
          }
          s.activity = { kind, label, since: Date.now() };
        }),

      endActivity: (outcome, error) =>
        set((s) => {
          if (!s.activity) return;
          s.activityLog.push({
            kind: outcome === "error" ? "error" : s.activity.kind === "thinking" ? "text" : s.activity.kind,
            label: error ? `${s.activity.label}: ${error}` : s.activity.label,
            at: s.activity.since,
            durationMs: Date.now() - s.activity.since,
          });
          if (s.activityLog.length > 200) {
            s.activityLog.splice(0, s.activityLog.length - 200);
          }
          s.activity = null;
        }),

      setViewOffset: (n) =>
        set((s) => {
          s.viewOffset = Math.max(0, n);
        }),

      scrollBy: (delta) =>
        set((s) => {
          const max = Math.max(0, s.messages.length - 1);
          s.viewOffset = Math.max(0, Math.min(max, s.viewOffset + delta));
        }),

      scrollToBottom: () =>
        set((s) => {
          s.viewOffset = 0;
        }),

      scrollToTop: () =>
        set((s) => {
          s.viewOffset = Math.max(0, s.messages.length - 1);
        }),

      setPromptDraft: (v) =>
        set((s) => {
          s.promptDraft = v;
        }),

      setMemories: (items, query) =>
        set((s) => {
          s.memories = items;
          if (typeof query === "string") s.memoryQuery = query;
          s.memoryLoading = false;
          // 拉取新列表后修剪已不存在的选中 id
          const existing = new Set(items.map((m) => m.id));
          s.memorySelection = s.memorySelection.filter((id) => existing.has(id));
        }),
      removeMemoryLocal: (id) =>
        set((s) => {
          s.memories = s.memories.filter((m) => m.id !== id);
          s.memorySelection = s.memorySelection.filter((x) => x !== id);
        }),
      setMemoryQuery: (v) =>
        set((s) => {
          s.memoryQuery = v;
        }),
      setMemoryLoading: (v) =>
        set((s) => {
          s.memoryLoading = v;
        }),
      setMemoryOpen: (v) =>
        set((s) => {
          s.memoryOpen = v;
          if (!v) s.memorySelection = []; // 关闭面板清空选择
        }),
      toggleMemorySelection: (id) =>
        set((s) => {
          const i = s.memorySelection.indexOf(id);
          if (i >= 0) s.memorySelection.splice(i, 1);
          else {
            s.memorySelection.push(id);
            s.memorySelection.sort((a, b) => a - b);
          }
        }),
      setMemorySelection: (ids) =>
        set((s) => {
          s.memorySelection = [...ids].sort((a, b) => a - b);
        }),
      clearMemorySelection: () =>
        set((s) => {
          s.memorySelection = [];
        }),

      setFocus: (f) =>
        set((s) => {
          s.focus = f;
        }),
      setPaletteOpen: (v) =>
        set((s) => {
          s.paletteOpen = v;
        }),
      setHistoryOpen: (v) =>
        set((s) => {
          s.historyOpen = v;
        }),
      setTheme: (t) =>
        set((s) => {
          s.theme = t;
        }),
      setVim: (next) =>
        set((s) => {
          if (typeof next.enabled === "boolean") s.vim.enabled = next.enabled;
          if (next.mode) s.vim.mode = next.mode;
        }),
      setDims: (cols, rows) =>
        set((s) => {
          s.dims.cols = cols;
          s.dims.rows = rows;
        }),
    })),
  ),
);
