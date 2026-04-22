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

// Message 领域模型

export type MessageRole =
  | "user"
  | "assistant"
  | "tool"
  | "system"
  | "error"
  | "confirm";

export interface Message {
  /** 客户端侧递增 id。 */
  id: string;
  role: MessageRole;
  /** 展示内容；assistant 走 markdown，其它走纯文本或 JSON。 */
  content: string;
  /** assistant 专属：是否仍在流式中。 */
  streaming?: boolean;
  /** tool 专属：工具名。 */
  toolName?: string;
  /** confirm 专属：interrupt 元信息。 */
  confirm?: ConfirmPayload;
  /** assistant 分段标记，供 rollover 后区分首段/中段/尾段样式。 */
  fragment?: "head" | "body" | "tail";
  /** 创建时间戳（毫秒）。 */
  createdAt: number;
}

// Store 状态

/** 当前 agent 正在做什么，由 tool/text/done/error 事件推导。 */
export interface Activity {
  kind: "thinking" | "tool" | "confirm";
  label: string;
  since: number;
}

/** 最近活动记录，供未来 Timeline 面板使用，最多保留 200 条。 */
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
  /** HistoryPanel 的多选 session id；删除事件会同步清理这里。 */
  sessionSelection: string[];

  // 消息
  messages: Message[];
  streaming: boolean;
  pendingConfirm: ConfirmPayload | null;

  // 动作追踪
  activity: Activity | null;
  activityLog: ActivityEntry[];

  // 输入框 draft 提到 store，便于 MainLayout 统一计算 SlashSuggestions 高度。
  promptDraft: string;

  /** Slash 补全面板当前占用的行数（0 = 不显示）。 */
  slashSuggestRows: number;

  /** agent 启动时的 cwd，只读展示给用户。 */
  workspaceCwd: string;

  // 记忆面板
  memories: MemoryMeta[];
  /** 最近一次记忆查询；空串表示浏览最近记忆。 */
  memoryQuery: string;
  memoryLoading: boolean;
  /** 记忆多选 id；保持排序方便稳定比对。 */
  memorySelection: number[];

  /** 消息区 Visual 选区；anchor/head 都是 messages 下标。 */
  selection: { anchor: number; head: number } | null;

  // UI
  focus: FocusTarget;
  paletteOpen: boolean;
  historyOpen: boolean;
  memoryOpen: boolean;
  /** 弹层是否独占 Esc；为 true 时全局处理器让步。 */
  overlayOwnsEscape: boolean;
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
  /** 切换单个 session id 的选中。 */
  toggleSessionSelection(id: string): void;
  /** 批量写入选择（用于 a 全选）。 */
  setSessionSelection(ids: string[]): void;
  /** 清空选择。 */
  clearSessionSelection(): void;

  appendMessage(msg: Message): void;
  appendAssistantChunk(chunk: string): void;
  /** 把 tool_result 输出追加到最后一条匹配的 tool 消息后面。 */
  appendToolOutput(name: string, output: string): void;
  finishAssistantStreaming(): void;
  setPendingConfirm(p: ConfirmPayload | null): void;
  setStreaming(v: boolean): void;
  clearMessages(): void;

  beginActivity(kind: Activity["kind"], label: string): void;
  endActivity(outcome?: "done" | "error", error?: string): void;

  setPromptDraft(v: string): void;
  setSlashSuggestRows(n: number): void;
  setWorkspaceCwd(cwd: string): void;

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

  /** 进入消息区 visual 选区，锚点落在 index。 */
  enterSelection(index: number): void;
  /** 让选区 head 相对移动 delta，并裁剪到消息范围内。 */
  moveSelectionHead(delta: number): void;
  /** 退出 visual 选区。 */
  clearSelection(): void;

  setFocus(f: FocusTarget): void;
  setPaletteOpen(v: boolean): void;
  setHistoryOpen(v: boolean): void;
  setOverlayOwnsEscape(v: boolean): void;
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

// Store 实现

let msgIdSeq = 0;
export function nextMessageId(): string {
  msgIdSeq += 1;
  return `m${Date.now()}-${msgIdSeq}`;
}

/** 计算 pending 区可保留的最大行数。 */
function pendingMaxLines(rows: number): number {
  return Math.max(3, Math.min(10, rows - 12));
}

/**
 * streaming assistant 太长时，把旧行切成 stable 消息塞到前面。
 *
 * 这样动态区只保留最新几行，避免 Ink fullscreen 降级导致的重绘抖动。
 */
function rolloverStreamingInPlace(
  s: { messages: Message[]; dims: { rows: number } },
  streaming: Message,
): void {
  if (!streaming.streaming || streaming.role !== "assistant") return;
  const maxLines = pendingMaxLines(s.dims.rows);
  const threshold = maxLines * 2;
  const lines = streaming.content.split(/\r?\n/);
  if (lines.length <= threshold) return;

  // 初始切点：尾部保留 maxLines 行
  let cut = lines.length - maxLines;

  // 若切点落在未闭合代码块里，就推到下一个 ``` 之后。
  const countFences = (arr: string[]): number =>
    arr.filter((l) => /^```/.test(l)).length;
  if (countFences(lines.slice(0, cut)) % 2 === 1) {
    let found = -1;
    for (let i = cut; i < lines.length; i++) {
      if (/^```/.test(lines[i]!)) {
        found = i + 1;
        break;
      }
    }
    // 没闭合 fence 或推后已经吃光尾部时，放弃本次 rollover。
    if (found < 0 || lines.length - found < 1) return;
    cut = found;
  }
  // 至少给 streaming 尾部留 1 行
  if (cut >= lines.length) return;

  const rolloverContent = lines.slice(0, cut).join("\n");
  const keptContent = lines.slice(cut).join("\n");

  // 首次切出 head，后续切出 body。
  const isFirst = !streaming.fragment;
  const rolloverFragment: "head" | "body" = isFirst ? "head" : "body";

  // 用 id 查 index，避免 immer draft 和 plain object 的引用比较问题。
  const idx = s.messages.findIndex((m) => m.id === streaming.id);
  if (idx < 0) return;
  s.messages.splice(idx, 0, {
    id: nextMessageId(),
    role: "assistant",
    content: rolloverContent,
    streaming: false,
    fragment: rolloverFragment,
    createdAt: Date.now(),
  });
  streaming.content = keptContent;
  streaming.fragment = "body";
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
      sessionSelection: [],

      messages: [],
      streaming: false,
      pendingConfirm: null,

      activity: null,
      activityLog: [],

      promptDraft: "",
      slashSuggestRows: 0,
      workspaceCwd: "",

      memories: [],
      memoryQuery: "",
      memoryLoading: false,
      memorySelection: [],

      selection: null,

      focus: "prompt",
      paletteOpen: false,
      historyOpen: false,
      memoryOpen: false,
      overlayOwnsEscape: false,
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
      toggleSessionSelection: (id) =>
        set((s) => {
          const i = s.sessionSelection.indexOf(id);
          if (i >= 0) s.sessionSelection.splice(i, 1);
          else s.sessionSelection.push(id);
        }),
      setSessionSelection: (ids) =>
        set((s) => {
          s.sessionSelection = [...ids];
        }),
      clearSessionSelection: () =>
        set((s) => {
          s.sessionSelection = [];
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
          // 从 messages 数组末尾重新拿引用（确保是 immer draft 而不是原 plain
          // object），再做 rollover —— 否则在 push 新消息的分支里 streaming
          // 修改不会被 immer 提交。
          const streamingMsg = s.messages[s.messages.length - 1];
          if (streamingMsg) rolloverStreamingInPlace(s, streamingMsg);
        }),

      appendToolOutput: (name, output) =>
        set((s) => {
          // 倒序找最后一条匹配 name 的 tool 消息（on_tool_start 刚 push 了一条）。
          // 正常情况下它就是 messages 末尾。
          for (let i = s.messages.length - 1; i >= 0; i--) {
            const m = s.messages[i]!;
            if (m.role === "tool" && m.toolName === name) {
              // content 约定首行是 "调用工具: <name>"；把 output 追加在其后，
              // 保持 MessageItem.extractToolOutput 能正确剥离。
              if (!m.content || m.content.endsWith("\n")) {
                m.content = (m.content || `调用工具: ${name}`) + output;
              } else {
                m.content = `${m.content}\n${output}`;
              }
              // 清 streaming 标志 → HistoryStatic 下一帧把它升华到 Static。
              // 这是 tool 消息唯一的"流式结束"信号（tool 消息没有 text chunk）。
              m.streaming = false;
              return;
            }
            // 越过 assistant / text，不越过用户消息（防止把结果挂到上轮 tool 上）
            if (m.role === "user") break;
          }
          // 兜底：未找到匹配 tool 消息，补一条（直接 non-streaming 进 Static）
          s.messages.push({
            id: nextMessageId(),
            role: "tool",
            content: `调用工具: ${name}\n${output}`,
            toolName: name,
            createdAt: Date.now(),
          });
        }),

      finishAssistantStreaming: () =>
        set((s) => {
          // 清理尾部所有还挂着 streaming=true 的消息（assistant 或 tool）。
          // 典型情况：流结束 / 出错时 tool_result 可能因上游异常没到达，这里
          // 兜底把 tool 消息也升华到 stable，避免它永远卡在 pending 区。
          //
          // 另外：若末段 assistant 是 rollover 产生的 body 段，把它标为 tail
          // 以便 MessageItem 给它补 marginBottom 做视觉分隔（避免紧贴下一条
          // 用户消息 / tool 消息）。
          for (let i = s.messages.length - 1; i >= 0; i--) {
            const m = s.messages[i]!;
            if (!m.streaming) break;
            m.streaming = false;
            if (m.role === "assistant" && m.fragment === "body") {
              m.fragment = "tail";
            }
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

      setPromptDraft: (v) =>
        set((s) => {
          s.promptDraft = v;
        }),
      setSlashSuggestRows: (n) =>
        set((s) => {
          s.slashSuggestRows = Math.max(0, Math.floor(n));
        }),
      setWorkspaceCwd: (cwd) =>
        set((s) => {
          s.workspaceCwd = cwd;
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

      enterSelection: (index) =>
        set((s) => {
          const n = s.messages.length;
          if (n === 0) {
            s.selection = null;
            return;
          }
          const i = Math.max(0, Math.min(n - 1, index));
          s.selection = { anchor: i, head: i };
        }),
      moveSelectionHead: (delta) =>
        set((s) => {
          if (!s.selection) return;
          const n = s.messages.length;
          if (n === 0) {
            s.selection = null;
            return;
          }
          const next = Math.max(0, Math.min(n - 1, s.selection.head + delta));
          s.selection.head = next;
          // print-above 架构下，已经打印到 terminal scrollback 的消息无法再高亮；
          // 选区仅对"pending 尾部"的流式消息可见，不再做 viewOffset 跟随。
        }),
      clearSelection: () =>
        set((s) => {
          s.selection = null;
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
      setOverlayOwnsEscape: (v) =>
        set((s) => {
          s.overlayOwnsEscape = v;
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
