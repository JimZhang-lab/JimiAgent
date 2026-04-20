import type { AgentEvent, AgentInfo, NodeRequest } from "../protocol/events.js";

/** Bootstrap 快照：ready 事件到来后落在 transport 上，便于晚订阅者同步读取。 */
export interface TransportBootstrap {
  agent_info: AgentInfo;
  default_session_id: string;
  worker_version: string;
}

/**
 * AgentTransport 抽象接口。
 *
 * 所有与 Python agent 通信的实现都遵守此接口。默认实现是 StdioTransport
 * （spawn Python 子进程 + stdin/stdout NDJSON），未来可加 WebSocketTransport
 * 连远程 Gateway 而无需改动上层 React 树。
 */
export interface AgentTransport {
  /** 启动底层连接（spawn 进程 / 建立 socket）。 */
  start(): Promise<void>;

  /** 发送一条请求。非阻塞，失败抛同步 Error。 */
  send(req: NodeRequest): void;

  /** 订阅事件流。返回 unsubscribe 函数。 */
  onEvent(cb: (ev: AgentEvent) => void): () => void;

  /** 订阅连接/进程级错误。 */
  onError(cb: (err: Error) => void): () => void;

  /** 订阅关闭事件（进程退出 / socket 关闭）。 */
  onClose(cb: (code: number | null) => void): () => void;

  /** 主动关闭；调用后 transport 不可再用。 */
  close(): Promise<void>;

  /** 是否已 start 且连接活着。 */
  readonly isAlive: boolean;

  /** 若已 ready，返回 bootstrap 快照；否则 null。 */
  readonly bootstrap: TransportBootstrap | null;
}
