import React, { useEffect, useCallback } from "react";
import { useApp, useInput, useStdout } from "ink";
import { MainLayout } from "./screens/MainLayout.js";
import type { AgentTransport } from "./transport/AgentTransport.js";
import { useStore } from "./state/store.js";
import { wireTransport, sendRequest } from "./state/wiring.js";
import { useAgent } from "./hooks/useAgent.js";

export interface AppProps {
  transport: AgentTransport;
}

export function App({ transport }: AppProps): React.ReactElement {
  const { exit } = useApp();
  const { stdout } = useStdout();
  const setDims = useStore((s) => s.setDims);
  const setPaletteOpen = useStore((s) => s.setPaletteOpen);
  const setHistoryOpen = useStore((s) => s.setHistoryOpen);
  const setMemoryOpen = useStore((s) => s.setMemoryOpen);
  const setMemoryLoading = useStore((s) => s.setMemoryLoading);
  const setFocus = useStore((s) => s.setFocus);
  const paletteOpen = useStore((s) => s.paletteOpen);
  const historyOpen = useStore((s) => s.historyOpen);
  const memoryOpen = useStore((s) => s.memoryOpen);
  const streaming = useStore((s) => s.streaming);
  const connected = useStore((s) => s.connected);
  const scrollBy = useStore((s) => s.scrollBy);
  const scrollToBottom = useStore((s) => s.scrollToBottom);
  const scrollToTop = useStore((s) => s.scrollToTop);
  const { cancel } = useAgent();

  // 挂 transport
  useEffect(() => {
    const unbind = wireTransport(transport);
    return () => {
      unbind();
    };
  }, [transport]);

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

  // 全局键盘：Ctrl+C / Ctrl+P / Ctrl+S / Ctrl+M / Esc
  useInput((input, key) => {
    // Ctrl+C
    if (key.ctrl && input === "c") {
      if (paletteOpen || historyOpen || memoryOpen) {
        closeOverlays();
        return;
      }
      if (streaming) {
        cancel();
        return;
      }
      exit();
      return;
    }
    // Esc：关闭弹层
    if (key.escape) {
      if (paletteOpen || historyOpen || memoryOpen) closeOverlays();
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
    // 滚动（只在无弹层时生效；弹层内有自己的↑↓）
    if (paletteOpen || historyOpen || memoryOpen) return;
    if (key.pageUp) {
      scrollBy(5);
      return;
    }
    if (key.pageDown) {
      scrollBy(-5);
      return;
    }
    // Home / End：按键名（Ink 暴露为 ESC 码；这里用 ctrl+组合作退路）
    if (key.ctrl && input === "u") {
      scrollBy(10);
      return;
    }
    if (key.ctrl && input === "d") {
      scrollBy(-10);
      return;
    }
    if (key.ctrl && input === "g") {
      // Ctrl+G：跳到最新
      scrollToBottom();
      return;
    }
    if (key.ctrl && input === "t") {
      // Ctrl+T：跳到最早
      scrollToTop();
      return;
    }
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
