import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

// `@ -> src` alias and dev server on :5173. VITE_USE_POLLING=true is required
// for HMR under Windows+Docker (see docs/architecture.md §6).
//
// `/api` is proxied to the backend gateway so the SPA talks to it
// same-origin: that keeps the <video> stream and the canvas crop free of
// CORS/tainting, and Range requests (seeking) just work. Override the target
// with VITE_API_PROXY for non-default backend hosts.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  server: {
    port: 5173,
    host: true,
    watch: { usePolling: process.env.VITE_USE_POLLING === "true" },
    proxy: {
      "/api": {
        target: process.env.VITE_API_PROXY ?? "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
