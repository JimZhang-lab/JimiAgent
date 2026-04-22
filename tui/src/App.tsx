import React, { useEffect, useCallback } from "react";
import { useApp, useInput, useStdout } from "ink";
import { MainLayout } from "./screens/MainLayout.js";
import { HistoryStatic } from "./components/HistoryStatic.js";
import type { AgentTransport } from "./transport/AgentTransport.js";
import { useStore, nextMessageId } from "./state/store.js";
import { wireTransport, sendRequest } from "./state/wiring.js";
import { useAgent } from "./hooks/useAgent.js";
import { useScopedBindings } from "./keybindings/useKeybinding.js";
import { useSelection } from "./hooks/useSelection.js";

export interface AppProps {
  transport: AgentTransport;
  /** cli 传入的工作目录，供 Header / StatusBar 展示。 */
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
  // streaming / thinking / tool 都算忙碌；此时 Ctrl+C 优先视为取消。
  const busy =
    streaming ||
    activity?.kind === "thinking" ||
    activity?.kind === "tool";

  // 绑定 transport
  useEffect(() => {
    const unbind = wireTransport(transport);
    return () => {
      unbind();
    };
  }, [transport]);

  // 把 cwd 写入 store
  useEffect(() => {
    if (cwd) setWorkspaceCwd(cwd);
  }, [cwd, setWorkspaceCwd]);

  // 上报实际生效的键位，供 /keys 展示。
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

  // 选区键位只在非弹层时激活
  const selectionActive = !paletteOpen && !historyOpen && !memoryOpen;
  useSelection({ isActive: selectionActive });

  // ready 后拉一次 sessions，供 HistoryPanel 使用
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

  // 空闲时双按 Ctrl+C 才退出，避免误触。
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
    // Esc：优先让弹层子模式处理，否则依次关弹层/退选区/焦点归位。
    if (key.escape) {
      if (overlayOwnsEscape) {
        // 弹层自己消费本次 Esc
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
    // messages 焦点下按 Enter 时回到 prompt，避免“没反应”。
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
    // Tab：prompt ↔ messages。弹层和 slash 补全激活时让出 Tab。
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

  // 根层结构：HistoryStatic 和 MainLayout 是 Fragment 的直接子元素。
  //
  // Ink 5 的 <Static>（由 HistoryStatic 内部使用）要求挂在 Fragment 根下，
  // 不能嵌套在 Box 里；否则 Static 的绝对定位会与父 Box 的 flex 布局冲突，
  // 导致 `/new` 清空 messages 时 Static 重挂载错位甚至渲染卡死。
  //
  // HistoryStatic 永远挂载（items=[] 时也留着占位），MainLayout 负责所有
  // 动态 UI（Header / pending 流式区 / PromptInput / 面板弹层 / StatusBar）。
  return (
    <>
      <HistoryStatic />
      <MainLayout
        onExit={exit}
        onOpenPalette={openPalette}
        onOpenHistory={openHistory}
        onOpenMemory={openMemory}
        onCloseOverlays={closeOverlays}
      />
    </>
  );
}
