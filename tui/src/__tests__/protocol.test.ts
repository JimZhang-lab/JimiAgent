import { describe, expect, it } from "vitest";
import {
  isAgentEvent,
  isStreamEvent,
  type AgentEvent,
} from "../protocol/events.js";

describe("protocol/events", () => {
  it("isAgentEvent 过滤非对象", () => {
    expect(isAgentEvent(null)).toBe(false);
    expect(isAgentEvent(undefined)).toBe(false);
    expect(isAgentEvent("hello")).toBe(false);
    expect(isAgentEvent(42)).toBe(false);
  });

  it("isAgentEvent 要求 type 是非空字符串", () => {
    expect(isAgentEvent({})).toBe(false);
    expect(isAgentEvent({ type: "" })).toBe(false);
    expect(isAgentEvent({ type: 123 })).toBe(false);
    expect(isAgentEvent({ type: "text", content: "x" })).toBe(true);
  });

  it("isStreamEvent 识别 agent 原生事件", () => {
    const cases: { ev: AgentEvent; expected: boolean }[] = [
      { ev: { type: "text", content: "x" }, expected: true },
      { ev: { type: "tool", name: "t" }, expected: true },
      { ev: { type: "session", session_id: "s" }, expected: true },
      { ev: { type: "error", content: "e" }, expected: true },
      {
        ev: { type: "trace", event: "a", name: "b", run_id: "r" },
        expected: true,
      },
      {
        ev: {
          type: "confirm_required",
          interrupt_id: "i",
          resumable: true,
          payload: {},
        },
        expected: true,
      },
    ];
    for (const { ev, expected } of cases) {
      expect(isStreamEvent(ev), ev.type).toBe(expected);
    }
  });

  it("isStreamEvent 排除 meta 事件", () => {
    const metas: AgentEvent[] = [
      {
        type: "ready",
        worker_version: "0.1",
        agent_info: {
          model: "m",
          embedding: "e",
          skills: [],
          gateway: { host: "h", port: 1 },
          memory: "memory",
        },
        default_session_id: "s",
      },
      { type: "done", session_id: "s" },
      { type: "sessions", items: [] },
      { type: "commands", items: [] },
      {
        type: "status",
        data: {
          session: null,
          agent: {
            model: "m",
            embedding: "e",
            skills: [],
            gateway: { host: "h", port: 1 },
            memory: "memory",
          },
          session_count: 0,
        },
      },
      { type: "session_deleted", session_id: "s", ok: true },
      { type: "pong" },
      { type: "log", level: "info", message: "m" },
    ];
    for (const ev of metas) {
      expect(isStreamEvent(ev), ev.type).toBe(false);
    }
  });
});
