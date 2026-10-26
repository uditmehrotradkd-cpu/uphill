import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies the UpHill backend (python -m uphill on :8800); production build is served by it.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8800",
      "/ws": { target: "ws://127.0.0.1:8800", ws: true },
    },
  },
});
