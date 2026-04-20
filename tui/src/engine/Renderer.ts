import type { ReactElement } from "react";

/**
 * 渲染引擎抽象接口。默认实现是 InkRenderer（Ink 5）。
 *
 * 存在意义：
 *   1) 为未来可能替换的引擎（OpenTUI / 自研 VT 引擎）预留空间；
 *   2) 让测试环境可以注入 mock renderer（例如 ink-testing-library）。
 */
export interface Renderer {
  /** 挂载根节点，返回卸载函数。 */
  mount(node: ReactElement): Promise<RendererHandle>;
}

export interface RendererHandle {
  unmount(): void;
  /** 等待 unmount 完成（用于优雅退出）。 */
  waitUntilExit(): Promise<void>;
  /** 强制重绘；大多数实现会忽略（Ink 自动 diff）。 */
  rerender?(node: ReactElement): void;
}
