import { useCallback, useEffect, useState } from "react";
import clipboardy from "clipboardy";

/**
 * 简易剪贴板 hook，封装 clipboardy：
 *   - copy(text) → 同步吐出 Promise，更新 lastCopied
 *   - paste()    → 读系统剪贴板
 *   - lastError  → 最近一次失败
 *
 * 终端无 GUI 剪贴板时 clipboardy 会抛错；我们把错误 swallow 到 state，
 * 组件可据此提示"本终端不支持剪贴板"。
 */
export function useClipboard() {
  const [lastCopied, setLastCopied] = useState<string | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);
  const [available, setAvailable] = useState<boolean>(true);

  useEffect(() => {
    // 探测一次；失败就标记不可用
    (async () => {
      try {
        await clipboardy.read();
        setAvailable(true);
      } catch (e) {
        setAvailable(false);
      }
    })();
  }, []);

  const copy = useCallback(async (text: string) => {
    try {
      await clipboardy.write(text);
      setLastCopied(text);
      setLastError(null);
      return true;
    } catch (e) {
      setLastError((e as Error).message);
      return false;
    }
  }, []);

  const paste = useCallback(async (): Promise<string | null> => {
    try {
      return await clipboardy.read();
    } catch (e) {
      setLastError((e as Error).message);
      return null;
    }
  }, []);

  return { copy, paste, lastCopied, lastError, available };
}
