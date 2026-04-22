import React, { useEffect, useCallback } from "react";
import { useApp, useInput, useStdout } from "ink";
import { MainLayout } from "./screens/MainLayout.js";
import type { AgentTransport } from "./transport/AgentTransport.js";
import { useStore, nextMessageId } from "./state/store.js";
import { wireTransport, sendRequest } from "./state/wiring.js";
import { useAgent } from "./hooks/useAgent.js";
import { useScopedBindings } from "./keybindings/useKeybinding.js";
import { useSelection } from "./hooks/useSelection.js";

export interface AppProps {
  transport: AgentTransport;
  /** 工作目录（由 cli 传入，供 Header / StatusBar 展示）。 */
  cwd?: string;
}

export function App({ transport, cwd }: AppProps): React.ReactElement {
  const { exit } = useApp();
  const { stdout } = useStdout();
  const setDims = useStore((s) => s.setDims);
  const setWorkspaceCwd = useStore((s) => s.setWorkspaceCwd);
  const setPaletteOpen = useStore((s) => s.setPaletteOpen);
  const setHistoryOpen = useStore((s) => s.setHistoryOpen);
  const setMemoryOpen = useStore((s) => s.setMemoryOpen);
  const setMemoryLoading = useStore((s) => s.setMemoryLoading);
  const setFocus = useStore((s) => s.setFocus);
  const paletteOpen = useStore((s) => s.paletteOpen);
  const historyOpen = useStore((s) => s.historyOpen);
  const memoryOpen = useStore((s) => s.memoryOpen);
  const streaming = useStore((s) => s.streaming);
  const activity = useStore((s) => s.activity);
  const connected = useStore((s) => s.connected);
  const focus = useStore((s) => s.focus);
  const selection = useStore((s) => s.selection);
  const clearSelection = useStore((s) => s.clearSelection);
  const overlayOwnsEscape = useStore((s) => s.overlayOwnsEscape);
  const { cancel } = useAgent();
  // 广义"忙碌"：streaming 或 thinking / tool 阶段都视作有活跃任务，
  // Ctrl+C 在此期间行为是"取消"，而非"提示再按一次退出"。
  const busy =
    streaming ||
    activity?.kind === "thinking" ||
    activity?.kind === "tool";

  // 挂 transport
  useEffect(() => {
    const unbind = wireTransport(transport);
    return () => {
      unbind();
    };
  }, [transport]);

  // 把 cwd 写入 store 供 Header / StatusBar 读取
  useEffect(() => {
    if (cwd) setWorkspaceCwd(cwd);
  }, [cwd, setWorkspaceCwd]);

  // 向注册表上报全局键位元数据（/keys 命令会列出）。
  // 只登记实际绑定的键位，避免 /help、/keys 输出虚假信息。
  useScopedBindings("global", [
    { key: "ctrl+c", description: "取消生成 / 关闭弹层 / 两次退出" },
    { key: "escape", description: "关闭弹层 / 退选区 / 焦点归位" },
    { key: "ctrl+p", description: "打开/关闭命令面板" },
    { key: "ctrl+s", description: "打开/关闭会话列表（/history）" },
    { key: "ctrl+m", description: "打开/关闭记忆管理" },
    { key: "tab", description: "切换焦点（输入区 ↔ 消息区）" },
  ]);
  useScopedBindings("selection", [
    { key: "v", description: "进入消息区选区（messages 焦点或 Vim normal）" },
    { key: "up / k", description: "扩大/移动选区向上" },
    { key: "down / j", description: "扩大/移动选区向下" },
    { key: "g / G", description: "跳到最早 / 最新消息" },
    { key: "u / U", description: "跳上一条 / 下一条 user 消息" },
    { key: "a / A", description: "跳上一条 / 下一条 assistant 消息" },
    { key: "y", description: "拷贝选区消息到系统剪贴板" },
    { key: "escape", description: "退出选区" },
  ]);

  // 选区键位：只在"非弹层 & 非 pendingConfirm"时激活
  const selectionActive = !paletteOpen && !historyOpen && !memoryOpen;
  useSelection({ isActive: selectionActive });

  // 初始一次会话列表（ready 后拉取，供 HistoryPanel 使用）
  useEffect(() => {
    if (!connected) return;
    sendRequest({ kind: "list_sessions" });
  }, [connected]);

  // 跟踪终端尺寸
  useEffect(() => {
    const onResize = () => {
      setDims(stdout.columns ?? 80, stdout.rows ?? 24);
    };
    onResize();
    stdout.on("resize", onResize);
    return () => {
      stdout.off("resize", onResize);
    };
  }, [stdout, setDims]);

  const closeOverlays = useCallback(() => {
    setPaletteOpen(false);
    setHistoryOpen(false);
    setMemoryOpen(false);
    setFocus("prompt");
  }, [setPaletteOpen, setHistoryOpen, setMemoryOpen, setFocus]);

  const openPalette = useCallback(() => {
    setHistoryOpen(false);
    setMemoryOpen(false);
    setPaletteOpen(true);
    setFocus("palette");
  }, [setPaletteOpen, setHistoryOpen, setMemoryOpen, setFocus]);

  const openHistory = useCallback(() => {
    setPaletteOpen(false);
    setMemoryOpen(false);
    setHistoryOpen(true);
    setFocus("history");
    sendRequest({ kind: "list_sessions" });
  }, [setPaletteOpen, setHistoryOpen, setMemoryOpen, setFocus]);

  const openMemory = useCallback(() => {
    setPaletteOpen(false);
    setHistoryOpen(false);
    setMemoryOpen(true);
    setFocus("memory");
    setMemoryLoading(true);
    sendRequest({ kind: "list_memories", limit: 100 });
  }, [setPaletteOpen, setHistoryOpen, setMemoryOpen, setFocus, setMemoryLoading]);

  // Ctrl+C 双按退出的计时戳。空闲状态下第一次 Ctrl+C 只给提示，
  // 2s 内再按才真退出，避免误触直接杀会话。
  const ctrlCArmedAt = React.useRef<number | null>(null);
  const appendMessage = useStore((s) => s.appendMessage);
  // 全局键盘：Ctrl+C / Ctrl+P / Ctrl+S / Ctrl+M / Esc
  useInput((input, key) => {
    // Ctrl+C
    if (key.ctrl && input === "c") {
      if (paletteOpen || historyOpen || memoryOpen) {
        closeOverlays();
        ctrlCArmedAt.current = null;
        return;
      }
      if (busy) {
        cancel();
        ctrlCArmedAt.current = null;
        return;
      }
      const now = Date.now();
      if (ctrlCArmedAt.current && now - ctrlCArmedAt.current < 2000) {
        exit();
        return;
      }
      ctrlCArmedAt.current = now;
      appendMessage({
        id: nextMessageId(),
        role: "system",
        content: "再按一次 Ctrl+C 退出（或用 /quit）。",
        createdAt: now,
      });
      return;
    }
    // Esc：弹层有子模式时让步；否则依次关弹层/退选区/焦点归位。
    if (key.escape) {
      if (overlayOwnsEscape) {
        // 弹层自己消费本次 Esc（如搜索模式、待删除确认态）
        return;
      }
      if (paletteOpen || historyOpen || memoryOpen) {
        closeOverlays();
        return;
      }
      if (selection) {
        clearSelection();
        return;
      }
      if (focus !== "prompt") {
        setFocus("prompt");
      }
      return;
    }
    // Enter：focus=messages 且没弹层/选区/确认时，把焦点切回 prompt；避免
    // 用户在消息区按回车"完全没反应"。
    if (
      key.return &&
      focus === "messages" &&
      !selection &&
      !paletteOpen &&
      !historyOpen &&
      !memoryOpen
    ) {
      setFocus("prompt");
      return;
    }
    // Tab：prompt ↔ messages。
    // 排除场景：弹层打开（面板内 Tab 有自己的语义）、slash 补全激活（Tab 补全命令名）
    const promptDraft = useStore.getState().promptDraft;
    const slashActive = promptDraft.startsWith("/") && !promptDraft.includes(" ");
    if (
      key.tab &&
      !paletteOpen &&
      !historyOpen &&
      !memoryOpen &&
      !slashActive
    ) {
      const next: "prompt" | "messages" =
        focus === "prompt" ? "messages" : "prompt";
      setFocus(next);
      return;
    }
    // Ctrl+P：命令面板
    if (key.ctrl && input === "p") {
      if (paletteOpen) closeOverlays();
      else openPalette();
      return;
    }
    // Ctrl+S：会话列表
    if (key.ctrl && input === "s") {
      if (historyOpen) closeOverlays();
      else openHistory();
      return;
    }
    // Ctrl+M：记忆面板
    if (key.ctrl && input === "m") {
      if (memoryOpen) closeOverlays();
      else openMemory();
      return;
    }
    // —————— 消息区滚动 ——————
    // 彻底移除 TUI 内部虚拟滚动，改走原生终端 scrollback：
    //   - 消息通过 Ink <Static> 打印到 stdout，终端自行记录 scrollback
    //   - 用户用终端原生能力（鼠标滚轮 / PgUp / Cmd+↑ / 三指上滑）回看
    //   - /history 命令打开会话列表，Enter 切换后服务端回放历史，全部进 scrollback
    // 因此这里不再绑定任何滚动相关快捷键。
  });

  return (
    <MainLayout
      onExit={exit}
      onOpenPalette={openPalette}
      onOpenHistory={openHistory}
      onOpenMemory={openMemory}
      onCloseOverlays={closeOverlays}
    />
  );
}
