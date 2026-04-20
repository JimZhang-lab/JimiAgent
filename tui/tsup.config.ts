import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/cli.tsx"],
  outDir: "dist",
  format: ["esm"],
  target: "node20",
  platform: "node",
  bundle: true,
  splitting: false,
  sourcemap: true,
  clean: true,
  shims: true,
  minify: false,
  dts: false,
  skipNodeModulesBundle: false,
  banner: {
    js: "#!/usr/bin/env node",
  },
  esbuildOptions(options) {
    options.chunkNames = "chunks/[name]-[hash]";
  },
  onSuccess: "chmod +x dist/cli.js",
});
