import * as path from "node:path";
import * as fs from "node:fs";
import * as url from "node:url";
import { App } from "./App.js";
import { InkRenderer } from "./engine/InkRenderer.js";
import { StdioTransport } from "./transport/StdioTransport.js";

/**
 * TUI 入口。由 `jimi chat`（Python CLI）spawn 进来。
 *
 * 流程：
 *   1) 解析环境变量（JIMI_TUI_SESSION / JIMI_TUI_WORKER_CMD 等）
 *   2) spawn Python worker（StdioTransport.start）
 *   3) 等待 {type:"ready"} 后挂 React 树
 *   4) 监听 SIGINT/SIGTERM，graceful shutdown
 */
async function main(): Promise<number> {
  if (!process.stdout.isTTY) {
    process.stderr.write(
      "jimi-tui: 当前 stdout 不是 TTY；TUI 仅能在交互终端运行\n",
    );
    return 2;
  }

  const cwd = resolveProjectRoot();
  const workerCmd = process.env.JIMI_TUI_WORKER_CMD;
  const { command, args } = parseWorkerCmd(workerCmd);

  const transport = new StdioTransport({
    command,
    args,
    cwd,
    readyTimeoutMs: 15000,
  });

  try {
    await transport.start();
  } catch (err) {
    process.stderr.write(
      `jimi-tui: 启动 tui_worker 失败：${(err as Error).message}\n`,
    );
    process.stderr.write(`详细日志见 ~/.jimiagent/tui-worker.log\n`);
    try {
      await transport.close();
    } catch {
      /* ignore */
    }
    return 1;
  }

  const renderer = new InkRenderer();
  const handle = await renderer.mount(<App transport={transport} cwd={cwd} />);

  // 信号处理：Ctrl+C 等效 App.exit，触发 waitUntilExit
  const onSignal = () => {
    handle.unmount();
  };
  process.once("SIGINT", onSignal);
  process.once("SIGTERM", onSignal);
  process.once("SIGHUP", onSignal);

  try {
    await handle.waitUntilExit();
  } finally {
    process.off("SIGINT", onSignal);
    process.off("SIGTERM", onSignal);
    process.off("SIGHUP", onSignal);
    await transport.close();
  }
  return 0;
}

function resolveProjectRoot(): string {
  // dist/cli.js 位于 <root>/tui/dist/cli.js
  // __dirname 运行时在 dist/ 下
  const here = path.dirname(url.fileURLToPath(import.meta.url));
  // 向上两级回到项目根
  const candidate = path.resolve(here, "..", "..");
  // 能找到 server/core/tui_worker.py 视为根
  const probe = path.join(candidate, "server", "core");
  if (fs.existsSync(probe)) return candidate;
  return process.cwd();
}

function parseWorkerCmd(raw: string | undefined): {
  command: string;
  args: string[];
} {
  if (!raw) {
    return { command: "python", args: ["-m", "server.core.tui_worker"] };
  }
  const tokens = raw.trim().split(/\s+/);
  const cmd = tokens[0];
  if (!cmd) {
    return { command: "python", args: ["-m", "server.core.tui_worker"] };
  }
  return { command: cmd, args: tokens.slice(1) };
}

main().then(
  (code) => {
    process.exit(code);
  },
  (err) => {
    process.stderr.write(`jimi-tui 意外崩溃：${(err as Error).stack ?? err}\n`);
    process.exit(1);
  },
);
