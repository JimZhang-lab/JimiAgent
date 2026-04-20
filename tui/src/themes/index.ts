import { darkTheme } from "./dark.js";
import { lightTheme } from "./light.js";

/**
 * 主题定义。所有颜色值遵守 Ink 的 color 约定：
 *   - 预设名（"cyan" / "green" / "red"）
 *   - 或 hex（"#a1b2c3"）
 * 预设名可在 16 色终端 fallback，hex 在 truecolor 终端更准。
 */
export interface Theme {
  name: string;
  background: "dark" | "light";
  colors: {
    primary: string;
    secondary: string;
    accent: string;
    text: string;
    textDim: string;
    border: string;
    borderActive: string;
    success: string;
    warning: string;
    error: string;
    info: string;
    user: string;
    assistant: string;
    tool: string;
    confirm: string;
  };
}

export type ThemeName = "auto" | "dark" | "light";

export function resolveTheme(name: ThemeName): Theme {
  if (name === "light") return lightTheme;
  if (name === "dark") return darkTheme;
  return detectAuto();
}

/**
 * 简易 auto detect：
 *   - JIMI_TUI_THEME 环境变量优先
 *   - COLORFGBG 存在时解析背景色（白 → light）
 *   - 否则默认 dark（大多数终端是暗色）
 */
export function detectAuto(): Theme {
  const forced = process.env.JIMI_TUI_THEME;
  if (forced === "dark") return darkTheme;
  if (forced === "light") return lightTheme;

  const fgbg = process.env.COLORFGBG;
  if (fgbg) {
    const parts = fgbg.split(";");
    const bg = parts[parts.length - 1];
    if (bg === "15" || bg === "7") return lightTheme;
  }

  return darkTheme;
}

export { darkTheme, lightTheme };
