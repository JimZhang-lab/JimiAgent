#!/usr/bin/env node
/**
 * Headless smoke test: spawn tui_worker 并走 NDJSON 协议，验证基本通信。
 *
 * 用法：
 *   node tui/scripts/smoke-transport.mjs
 *
 * 期望输出顺序：
 *   [ready] agent=openai/... session=...
 *   [pong]
 *   [sessions] N items
 *   OK
 *
 * 非零退出 = smoke 失败。
 */
import { spawn } from "node:child_process";
import * as readline from "node:readline";
import * as path from "node:path";
import * as url from "node:url";

const here = path.dirname(url.fileURLToPath(import.meta.url));
const projectRoot = path.resolve(here, "..", "..");

// 允许用 JIMI_TUI_WORKER_CMD 覆盖；默认走 conda env `jimiAgent312`
const rawCmd = process.env.JIMI_TUI_WORKER_CMD;
const { cmd, args } = rawCmd
  ? parseCmd(rawCmd)
  : {
      cmd: "conda",
      args: [
        "run",
        "-n",
        "jimiAgent312",
        "--no-capture-output",
        "python",
        "-m",
        "server.core.tui_worker",
      ],
    };

function parseCmd(raw) {
  const tokens = raw.trim().split(/\s+/);
  return { cmd: tokens[0], args: tokens.slice(1) };
}

console.log(`[smoke] spawn: ${cmd} ${args.join(" ")} (cwd=${projectRoot})`);

const child = spawn(cmd, args, {
  cwd: projectRoot,
  env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" },
  stdio: ["pipe", "pipe", "pipe"],
});

const rl = readline.createInterface({ input: child.stdout, crlfDelay: Infinity });

let gotReady = false;
let gotPong = false;
let gotSessions = false;
let gotMemories = false;
const timeoutMs = 30_000;
const timer = setTimeout(() => {
  console.error("[smoke] timeout");
  child.kill("SIGTERM");
  process.exit(1);
}, timeoutMs);

child.stderr.on("data", (buf) => {
  // 不主动打印日志，避免干扰；只在失败时 dump
  _stderrBuf += buf.toString();
});
let _stderrBuf = "";

function send(req) {
  child.stdin.write(JSON.stringify(req) + "\n");
}

rl.on("line", (line) => {
  let ev;
  try {
    ev = JSON.parse(line);
  } catch (e) {
    console.error("[smoke] 非 JSON 行:", line.slice(0, 200));
    return;
  }
  switch (ev.type) {
    case "ready":
      gotReady = true;
      console.log(
        `[ready] agent=${ev.agent_info?.model ?? "?"} session=${ev.default_session_id ?? "?"}`,
      );
      send({ kind: "ping", req_id: "r1" });
      break;
    case "pong":
      gotPong = true;
      console.log("[pong]");
      send({ kind: "list_sessions", req_id: "r2" });
      break;
    case "sessions":
      gotSessions = true;
      console.log(`[sessions] ${ev.items?.length ?? 0} items`);
      send({ kind: "list_memories", req_id: "r3", limit: 10 });
      break;
    case "memories":
      gotMemories = true;
      console.log(`[memories] ${ev.items?.length ?? 0} items`);
      send({ kind: "shutdown" });
      break;
    case "error":
      console.error("[smoke] agent error:", ev.content);
      break;
    default:
      // 忽略 log / trace / done 等
      break;
  }
});

child.on("exit", (code) => {
  clearTimeout(timer);
  if (gotReady && gotPong && gotSessions && gotMemories) {
    console.log("OK");
    process.exit(0);
  }
  console.error(
    `[smoke] 失败: exit=${code} ready=${gotReady} pong=${gotPong} sessions=${gotSessions} memories=${gotMemories}`,
  );
  if (_stderrBuf) {
    console.error("[smoke] worker stderr tail:");
    const tail = _stderrBuf.split("\n").slice(-10).join("\n");
    console.error(tail);
  }
  process.exit(code ?? 1);
});
