#!/usr/bin/env node
/**
 * `/new` 路径 smoke。
 *
 * 验证 `new_session -> session/history/sessions -> switch_session` 这一整条链路。
 */
import { spawn } from "node:child_process";
import * as readline from "node:readline";
import * as path from "node:path";
import * as url from "node:url";

const here = path.dirname(url.fileURLToPath(import.meta.url));
const projectRoot = path.resolve(here, "..", "..");

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

console.log(`[smoke] spawn: ${cmd} ${args.join(" ")}`);

const child = spawn(cmd, args, {
  cwd: projectRoot,
  env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" },
  stdio: ["pipe", "pipe", "pipe"],
});

const rl = readline.createInterface({ input: child.stdout, crlfDelay: Infinity });

let gotReady = false;
let defaultSessionId = null;
let newSessionId = null;
let gotNewHistory = false;
let gotNewSessions = false;
let gotSwitchHistory = false;
let t0 = 0;

const timeoutMs = 20_000;
const timer = setTimeout(() => {
  console.error("[smoke] TIMEOUT — worker 没在", timeoutMs, "ms 内完成流程");
  console.error("状态:", {
    gotReady,
    defaultSessionId,
    newSessionId,
    gotNewHistory,
    gotNewSessions,
    gotSwitchHistory,
  });
  child.kill("SIGTERM");
  process.exit(1);
}, timeoutMs);

let _stderrBuf = "";
child.stderr.on("data", (buf) => {
  _stderrBuf += buf.toString();
});

function send(req) {
  const line = JSON.stringify(req) + "\n";
  child.stdin.write(line);
  console.log(`[>>>] ${JSON.stringify(req)}`);
}

rl.on("line", (line) => {
  let ev;
  try {
    ev = JSON.parse(line);
  } catch (e) {
    console.error("[非 JSON]", line.slice(0, 200));
    return;
  }
  const tag = ev.type;
  console.log(`[<<<] ${tag}`, describeEvent(ev));

  switch (tag) {
    case "ready":
      gotReady = true;
      defaultSessionId = ev.default_session_id;
      // 先发 new_session，开始计时
      t0 = Date.now();
      send({ kind: "new_session", title: "smoke-test", req_id: "r-new" });
      break;
    case "session":
      if (!newSessionId && ev.session_id !== defaultSessionId) {
        newSessionId = ev.session_id;
      }
      break;
    case "history":
      if (newSessionId && ev.session_id === newSessionId && !gotNewHistory) {
        gotNewHistory = true;
        console.log(
          `[timing] new_session→history: ${Date.now() - t0} ms (items=${ev.items.length})`,
        );
      } else if (ev.session_id === defaultSessionId && gotNewSessions) {
        gotSwitchHistory = true;
        console.log(
          `[timing] switch_session→history: items=${ev.items.length}`,
        );
      }
      break;
    case "sessions":
      if (newSessionId && !gotNewSessions) {
        gotNewSessions = true;
        console.log(
          `[timing] new_session→sessions: ${Date.now() - t0} ms (count=${ev.items.length})`,
        );
        // 再切回默认会话
        send({
          kind: "switch_session",
          session_id: defaultSessionId,
          req_id: "r-sw",
        });
      }
      break;
    case "done":
      if (gotSwitchHistory) {
        send({ kind: "shutdown" });
      }
      break;
    case "error":
      console.error("[worker error]", ev.content);
      break;
  }
});

function describeEvent(ev) {
  const preview = { ...ev };
  if (Array.isArray(preview.items)) {
    preview.items = `[${preview.items.length} items]`;
  }
  return JSON.stringify(preview).slice(0, 300);
}

child.on("exit", (code) => {
  clearTimeout(timer);
  const allOK =
    gotReady &&
    newSessionId &&
    gotNewHistory &&
    gotNewSessions &&
    gotSwitchHistory;
  if (allOK) {
    console.log("OK");
    process.exit(0);
  }
  console.error(
    `FAIL exit=${code} ready=${gotReady} newId=${newSessionId} newHistory=${gotNewHistory} newSessions=${gotNewSessions} switchHistory=${gotSwitchHistory}`,
  );
  if (_stderrBuf) {
    console.error("--- worker stderr tail ---");
    console.error(_stderrBuf.split("\n").slice(-30).join("\n"));
  }
  process.exit(code ?? 1);
});
