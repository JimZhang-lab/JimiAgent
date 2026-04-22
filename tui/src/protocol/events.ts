/**
 * Stdio NDJSON 通信协议：Node TUI ⇄ Python tui_worker。
 *
 * - Node → Python: NodeRequest
 * - Python → Node: AgentEvent
 *
 * 协议设计原则：一行一条 JSON，UTF-8；双向都容忍未知字段（不要 throw）。
 */

// ============================================================================
// Python 端 meta 对象
// ============================================================================

export interface SessionMeta {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
  metadata: Record<string, unknown>;
}

export interface AgentInfo {
  model: string;
  embedding: string;
  skills: string[];
  gateway: { host: string; port: number };
  memory: "sqlite" | "memory" | "disabled";
}

export interface StatusSnapshot {
  session: SessionMeta | null;
  agent: AgentInfo;
  session_count: number;
}

/** 记忆元信息（从 MemoryStore.Memory 投影而来）。 */
export interface MemoryMeta {
  id: number;
  kind: string;
  subject: string;
  text: string;
  created_at: string;
  updated_at: string;
  source_session: string;
  hits: number;
  score?: number;
  predicate?: string;
  object?: string;
  valid_from?: string;
  valid_until?: string;
  namespace?: string;
}

export interface ConfirmPayload {
  interrupt_id: string;
  resumable: boolean;
  payload: {
    summary?: string;
    detail?: Record<string, unknown>;
    [k: string]: unknown;
  };
}

// ============================================================================
// Node → Python (请求)
// ============================================================================

export type NodeRequest =
  | { kind: "init" }
  | {
      kind: "chat";
      session_id: string;
      message: string;
      images?: string[];
      req_id?: string;
    }
  | {
      kind: "resume";
      session_id: string;
      approve: boolean;
      interrupt_id?: string;
      req_id?: string;
    }
  | { kind: "cancel"; session_id: string; req_id?: string }
  | { kind: "list_sessions"; req_id?: string }
  | { kind: "list_commands"; req_id?: string }
  | { kind: "switch_session"; session_id: string; req_id?: string }
  | { kind: "new_session"; title?: string; req_id?: string }
  | { kind: "delete_session"; session_id: string; req_id?: string }
  | { kind: "get_status"; session_id: string; req_id?: string }
  | {
      kind: "list_memories";
      limit?: number;
      kinds?: string[];
      req_id?: string;
    }
  | {
      kind: "search_memories";
      query: string;
      k?: number;
      kinds?: string[];
      req_id?: string;
    }
  | { kind: "delete_memory"; memory_id: number; req_id?: string }
  | { kind: "ping"; req_id?: string }
  | { kind: "shutdown" };

// ============================================================================
// Python → Node (事件)
// ============================================================================

/** 1:1 映射 agent.chat_stream 的原生事件 */
export type AgentStreamEvent =
  | { type: "text"; content: string }
  | { type: "tool"; name: string }
  | { type: "session"; session_id: string }
  | { type: "error"; content: string }
  | {
      type: "trace";
      event: string;
      name: string;
      run_id: string;
    }
  | ({ type: "confirm_required" } & ConfirmPayload);

/** 历史消息条目（history 事件的 items） */
export interface HistoryItem {
  role: "user" | "assistant" | "tool" | "system";
  content: string;
  /** tool role 专属：工具名 */
  tool_name?: string;
}

/** TUI 专用 wrapper 事件 */
export type AgentMetaEvent =
  | {
      type: "ready";
      worker_version: string;
      agent_info: AgentInfo;
      default_session_id: string;
    }
  | { type: "done"; session_id: string; req_id?: string }
  | {
      /**
       * 会话历史批量回放。
       * 由 switch_session / new_session 触发；前端收到后 bulk append，
       * 让历史对话直接写入 terminal scrollback（print-above 架构）。
       */
      type: "history";
      session_id: string;
      items: HistoryItem[];
      req_id?: string;
    }
  | {
      type: "sessions";
      items: SessionMeta[];
      req_id?: string;
    }
  | {
      type: "commands";
      items: { name: string; description: string }[];
      req_id?: string;
    }
  | {
      type: "status";
      data: StatusSnapshot;
      req_id?: string;
    }
  | {
      type: "session_deleted";
      session_id: string;
      ok: boolean;
      req_id?: string;
    }
  | {
      type: "memories";
      items: MemoryMeta[];
      query?: string;
      req_id?: string;
    }
  | {
      type: "memory_deleted";
      memory_id: number;
      ok: boolean;
      req_id?: string;
    }
  | { type: "pong"; req_id?: string }
  | { type: "log"; level: "debug" | "info" | "warn" | "error"; message: string };

export type AgentEvent = AgentStreamEvent | AgentMetaEvent;

// ============================================================================
// Type guards
// ============================================================================

export function isAgentEvent(x: unknown): x is AgentEvent {
  if (!x || typeof x !== "object") return false;
  const t = (x as { type?: unknown }).type;
  return typeof t === "string" && t.length > 0;
}

export function isStreamEvent(ev: AgentEvent): ev is AgentStreamEvent {
  return (
    ev.type === "text" ||
    ev.type === "tool" ||
    ev.type === "session" ||
    ev.type === "error" ||
    ev.type === "trace" ||
    ev.type === "confirm_required"
  );
}
