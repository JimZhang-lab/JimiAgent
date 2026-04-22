#!/usr/bin/env node
/**
 * 验证“幽灵会话”修复。
 *
 * 重点看两点：删光会话后是否会自动补默认会话，以及对已删 session_id 的 chat 是否会被拒绝。
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

// 状态机：每一步完成后推进到下一个目标
let step = "wait_ready";
let defaultSessionId = null;
let newSessionId = null;
let ghostId = null; // 将被删的 id，用于 L3 测试
let ensuredSessionId = null; // L4: 删光后 worker ensure 出的新 default
let gotEnsuredHistory = false;
let ghostChatError = false; // L3: 对 ghost 发 chat 应收 error
let ghostChatDone = false;
let ghostChatGotStream = false; // 不应该发生
let deletedOk = { default: false, new: false };

const timeoutMs = 25_000;
const timer = setTimeout(() => {
  console.error("[smoke] TIMEOUT");
  console.error("状态:", {
    step,
    defaultSessionId,
    newSessionId,
    ghostId,
    ensuredSessionId,
    gotEnsuredHistory,
    ghostChatError,
    ghostChatDone,
    ghostChatGotStream,
    deletedOk,
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
  } catch {
    return;
  }
  const tag = ev.type;
  console.log(`[<<<] ${tag}`, describeEvent(ev));

  if (tag === "ready") {
    defaultSessionId = ev.default_session_id;
    step = "create_new";
    send({ kind: "new_session", title: "ghost-smoke", req_id: "r-new" });
    return;
  }

  if (tag === "session") {
    // 这里只关心 new_session 的新会话，和删光后 ensure 出来的默认会话。
    if (step === "create_new" && !newSessionId && ev.session_id !== defaultSessionId) {
      newSessionId = ev.session_id;
    } else if (step === "waiting_ensure" && ev.session_id !== defaultSessionId && ev.session_id !== newSessionId) {
      ensuredSessionId = ev.session_id;
    }
    return;
  }

  if (tag === "history") {
    if (ensuredSessionId && ev.session_id === ensuredSessionId) {
      gotEnsuredHistory = true;
    }
    return;
  }

  if (tag === "session_deleted") {
    if (ev.ok && ev.session_id === defaultSessionId) {
      deletedOk.default = true;
    } else if (ev.ok && ev.session_id === newSessionId) {
      deletedOk.new = true;
    }
    return;
  }

  if (tag === "sessions") {
    if (step === "create_new" && newSessionId) {
      // 进入删除阶段
      step = "delete_default";
      send({ kind: "delete_session", session_id: defaultSessionId, req_id: "r-d1" });
    } else if (step === "delete_default" && deletedOk.default) {
      step = "delete_new"; // 删最后一条 → 触发 L4 ensure
      ghostId = newSessionId;
      send({ kind: "delete_session", session_id: newSessionId, req_id: "r-d2" });
      step = "waiting_ensure";
    } else if (step === "waiting_ensure" && ensuredSessionId && gotEnsuredHistory) {
      // L4 完成后，用 ghost id 发 chat 测 L3
      step = "chat_ghost";
      send({ kind: "chat", session_id: ghostId, message: "hello ghost", req_id: "r-g" });
    }
    return;
  }

  if (tag === "error") {
    if (step === "chat_ghost") {
      ghostChatError = true;
    }
    return;
  }

  if (tag === "text" || tag === "tool") {
    if (step === "chat_ghost") {
      ghostChatGotStream = true;
    }
    return;
  }

  if (tag === "done") {
    if (step === "chat_ghost" && ev.req_id === "r-g") {
      ghostChatDone = true;
      send({ kind: "shutdown" });
    }
    return;
  }
});

function describeEvent(ev) {
  const p = { ...ev };
  if (Array.isArray(p.items)) p.items = `[${p.items.length} items]`;
  return JSON.stringify(p).slice(0, 220);
}

child.on("exit", (code) => {
  clearTimeout(timer);
  const l4_ok = deletedOk.default && deletedOk.new && !!ensuredSessionId && gotEnsuredHistory;
  const l3_ok = ghostChatError && ghostChatDone && !ghostChatGotStream;
  if (l4_ok && l3_ok) {
    console.log("OK  L4(ensure_default) + L3(ghost chat rejected)");
    process.exit(0);
  }
  console.error(
    `FAIL exit=${code}\n` +
      `  L4: deletedOk=${JSON.stringify(deletedOk)} ensured=${ensuredSessionId} history=${gotEnsuredHistory}\n` +
      `  L3: error=${ghostChatError} done=${ghostChatDone} stream=${ghostChatGotStream}`,
  );
  if (_stderrBuf) {
    console.error("--- worker stderr tail ---");
    console.error(_stderrBuf.split("\n").slice(-30).join("\n"));
  }
  process.exit(code ?? 1);
});
