// Generates typed DTOs from the backend's OpenAPI so the frontend↔backend
// contract is free and typed. Flow: download backend /openapi.json next to
// this file → `npm run generate:types` → types appear in src/api/generated/.
// See docs/architecture.md §4.4.
import { defineConfig } from "@hey-api/openapi-ts";

export default defineConfig({
  input: "./openapi.json",
  output: "./src/api/generated",
  plugins: ["@hey-api/typescript"],
});
