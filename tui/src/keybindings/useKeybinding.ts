import { useEffect, useRef } from "react";
import {
  registerKeybinding,
  unregisterKeybinding,
  type KeybindingScope,
} from "./registry.js";

export interface BindingMeta {
  key: string;
  description: string;
}

/**
 * 声明一组属于同一 scope 的键位元数据。
 *
 * 使用方式：组件挂载时调用一次，传入静态数组字面量；hook 会把它们注册到
 * 中心注册表（只做元数据收集，不影响真正的 useInput 路由），卸载时自动移除。
 *
 * 静态数组字面量 ≠ 稳定引用，因此这里内部用 ref 锁定首次传入的列表，
 * 之后的 re-render 不会重复注册。
 */
export function useScopedBindings(
  scope: KeybindingScope,
  bindings: BindingMeta[],
): void {
  const stableRef = useRef<BindingMeta[]>(bindings);
  useEffect(() => {
    const ids = stableRef.current.map((b) =>
      registerKeybinding(scope, b.key, b.description),
    );
    return () => {
      for (const id of ids) unregisterKeybinding(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scope]);
}
