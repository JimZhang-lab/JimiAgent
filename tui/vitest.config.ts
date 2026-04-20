import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["src/**/__tests__/**/*.test.ts", "src/**/__tests__/**/*.test.tsx"],
    environment: "node",
    globals: false,
    testTimeout: 10_000,
    reporters: process.env.CI ? "default" : "verbose",
    coverage: {
      reporter: ["text", "html"],
    },
  },
  esbuild: {
    target: "node20",
  },
});
