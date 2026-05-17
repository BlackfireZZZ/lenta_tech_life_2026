import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// `@ -> src` alias and dev server on :5173. VITE_USE_POLLING=true is required
// for HMR under Windows+Docker (see docs/architecture.md §6).
export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  server: {
    port: 5173,
    host: true,
    watch: { usePolling: process.env.VITE_USE_POLLING === "true" },
  },
});
