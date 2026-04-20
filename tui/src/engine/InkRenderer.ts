import type { ReactElement } from "react";
import { render } from "ink";
import type { Renderer, RendererHandle } from "./Renderer.js";

export class InkRenderer implements Renderer {
  async mount(node: ReactElement): Promise<RendererHandle> {
    const instance = render(node, {
      exitOnCtrlC: false,
      patchConsole: true,
    });
    return {
      unmount: () => instance.unmount(),
      waitUntilExit: () => instance.waitUntilExit(),
      rerender: (next) => instance.rerender(next),
    };
  }
}
