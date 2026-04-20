import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { once } from "node:events";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import * as readline from "node:readline";
import type { AgentTransport } from "./AgentTransport.js";
import {
  isAgentEvent,
  type AgentEvent,
  type AgentInfo,
  type NodeRequest,
} from "../protocol/events.js";

/**
 * Bootstrap 快照：成功 `start()` 后，ready 事件会把以下字段落在 transport 上，
 * 供晚订阅者（例如 React 挂载之后才调 onEvent）同步读取。
 */
export interface TransportBootstrap {
  agent_info: AgentInfo;
  default_session_id: string;
  worker_version: string;
}

export interface StdioTransportOptions {
  /** 自定义启动命令；默认 `python -m server.core.tui_worker`。 */
  command?: string;
  /** 自定义启动参数，会覆盖 command 的默认值。 */
  args?: string[];
  /** worker 的 cwd；默认进程当前 cwd。 */
  cwd?: string;
  /** 附加环境变量；默认继承 process.env。 */
  env?: NodeJS.ProcessEnv;
  /** 初始化完成超时毫秒；默认 15000。 */
  readyTimeoutMs?: number;
  /** stderr 日志路径；默认 ~/.jimiagent/tui-worker.log。 */
  stderrLogPath?: string;
}

type EventCb = (ev: AgentEvent) => void;
type ErrCb = (err: Error) => void;
type CloseCb = (code: number | null) => void;

/**
 * Stdio 子进程 transport 实现。
 *
 * - spawn `python -m server.core.tui_worker`（或 opts.command 覆盖）
 * - stdin/stdout 走 NDJSON；stderr 转发到文件（避免污染 TTY）
 * - 首次收到 `{type:"ready"}` 才算真正 start()
 */
export class StdioTransport implements AgentTransport {
  private child: ChildProcessWithoutNullStreams | null = null;
  private readonly opts: Required<
    Omit<StdioTransportOptions, "args" | "env" | "stderrLogPath">
  > &
    Pick<StdioTransportOptions, "args" | "env" | "stderrLogPath">;
  private readonly eventCbs = new Set<EventCb>();
  private readonly errorCbs = new Set<ErrCb>();
  private readonly closeCbs = new Set<CloseCb>();
  private rl: readline.Interface | null = null;
  private stderrStream: fs.WriteStream | null = null;
  private readyResolved = false;
  private closedByUser = false;
  private _bootstrap: TransportBootstrap | null = null;

  constructor(opts: StdioTransportOptions = {}) {
    this.opts = {
      command: opts.command ?? "python",
      cwd: opts.cwd ?? process.cwd(),
      readyTimeoutMs: opts.readyTimeoutMs ?? 15000,
      args: opts.args,
      env: opts.env,
      stderrLogPath: opts.stderrLogPath,
    };
  }

  get isAlive(): boolean {
    return !!this.child && this.child.exitCode === null && !this.closedByUser;
  }

  /** 已 ready 的 bootstrap 快照；晚订阅者（wireTransport）据此同步恢复状态。 */
  get bootstrap(): TransportBootstrap | null {
    return this._bootstrap;
  }

  async start(): Promise<void> {
    if (this.child) {
      throw new Error("StdioTransport.start() called twice");
    }

    const args = this.opts.args ?? ["-m", "server.core.tui_worker"];
    const env = this.opts.env ?? process.env;

    const stderrPath = this.resolveStderrPath();
    fs.mkdirSync(path.dirname(stderrPath), { recursive: true });
    this.stderrStream = fs.createWriteStream(stderrPath, { flags: "a" });
    this.stderrStream.write(
      `\n===== tui_worker spawned at ${new Date().toISOString()} =====\n`,
    );

    this.child = spawn(this.opts.command, args, {
      cwd: this.opts.cwd,
      env: {
        ...env,
        // 让 Python print 不走行缓冲，确保 NDJSON 每行立即可见
        PYTHONUNBUFFERED: "1",
        PYTHONIOENCODING: "utf-8",
      },
      stdio: ["pipe", "pipe", "pipe"],
    }) as ChildProcessWithoutNullStreams;

    this.child.stderr.on("data", (chunk: Buffer) => {
      this.stderrStream?.write(chunk);
    });

    this.child.on("error", (err) => {
      this.emitError(err);
    });
    this.child.on("exit", (code) => {
      this.emitClose(code);
    });

    this.rl = readline.createInterface({
      input: this.child.stdout,
      crlfDelay: Infinity,
    });
    this.rl.on("line", (line) => this.handleLine(line));

    // 等待 ready 事件或进程退出
    await this.waitForReady();
  }

  send(req: NodeRequest): void {
    if (!this.child || this.child.exitCode !== null) {
      throw new Error("StdioTransport is not alive");
    }
    const line = JSON.stringify(req) + "\n";
    this.child.stdin.write(line, "utf-8");
  }

  onEvent(cb: EventCb): () => void {
    this.eventCbs.add(cb);
    return () => this.eventCbs.delete(cb);
  }

  onError(cb: ErrCb): () => void {
    this.errorCbs.add(cb);
    return () => this.errorCbs.delete(cb);
  }

  onClose(cb: CloseCb): () => void {
    this.closeCbs.add(cb);
    return () => this.closeCbs.delete(cb);
  }

  async close(): Promise<void> {
    this.closedByUser = true;
    const child = this.child;
    if (!child) return;

    // 优雅 shutdown：先发 shutdown 再给 200ms，再 SIGTERM，再 SIGKILL
    try {
      if (child.exitCode === null && child.stdin.writable) {
        child.stdin.write(JSON.stringify({ kind: "shutdown" }) + "\n");
        child.stdin.end();
      }
    } catch {
      /* ignore */
    }

    if (child.exitCode === null) {
      const exited = Promise.race([
        once(child, "exit").then(() => true),
        new Promise<boolean>((resolve) =>
          setTimeout(() => resolve(false), 400),
        ),
      ]);
      if (!(await exited)) {
        child.kill("SIGTERM");
        const killed = Promise.race([
          once(child, "exit").then(() => true),
          new Promise<boolean>((resolve) =>
            setTimeout(() => resolve(false), 400),
          ),
        ]);
        if (!(await killed)) {
          child.kill("SIGKILL");
        }
      }
    }

    this.rl?.close();
    this.stderrStream?.end();
  }

  // ---------- internals ----------

  private async waitForReady(): Promise<void> {
    if (this.readyResolved) return;
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        unsubEv();
        unsubErr();
        unsubClose();
        reject(
          new Error(
            `tui_worker 启动超时 (${this.opts.readyTimeoutMs}ms)，请检查 stderr 日志`,
          ),
        );
      }, this.opts.readyTimeoutMs);

      const unsubEv = this.onEvent((ev) => {
        if (ev.type === "ready") {
          clearTimeout(timeout);
          unsubEv();
          unsubErr();
          unsubClose();
          this.readyResolved = true;
          resolve();
        }
      });
      const unsubErr = this.onError((err) => {
        clearTimeout(timeout);
        unsubEv();
        unsubErr();
        unsubClose();
        reject(err);
      });
      const unsubClose = this.onClose((code) => {
        clearTimeout(timeout);
        unsubEv();
        unsubErr();
        unsubClose();
        reject(
          new Error(
            `tui_worker 在 ready 前退出（code=${code}），请检查 stderr 日志`,
          ),
        );
      });
    });
  }

  private handleLine(line: string): void {
    const text = line.trim();
    if (!text) return;
    let payload: unknown;
    try {
      payload = JSON.parse(text);
    } catch (e) {
      this.emitError(
        new Error(`tui_worker 产出非 JSON 行: ${text.slice(0, 120)}`),
      );
      return;
    }
    if (!isAgentEvent(payload)) {
      // 未知结构：记日志但不中断
      this.stderrStream?.write(`[stdio] drop unknown event: ${text}\n`);
      return;
    }
    // 记录 ready，便于晚订阅者同步读取
    if (payload.type === "ready") {
      this._bootstrap = {
        agent_info: payload.agent_info,
        default_session_id: payload.default_session_id,
        worker_version: payload.worker_version,
      };
    }
    for (const cb of this.eventCbs) {
      try {
        cb(payload);
      } catch (e) {
        this.emitError(e instanceof Error ? e : new Error(String(e)));
      }
    }
  }

  private emitError(err: Error): void {
    for (const cb of this.errorCbs) {
      try {
        cb(err);
      } catch {
        /* swallow */
      }
    }
  }

  private emitClose(code: number | null): void {
    for (const cb of this.closeCbs) {
      try {
        cb(code);
      } catch {
        /* swallow */
      }
    }
  }

  private resolveStderrPath(): string {
    if (this.opts.stderrLogPath) return this.opts.stderrLogPath;
    return path.join(os.homedir(), ".jimiagent", "tui-worker.log");
  }
}
