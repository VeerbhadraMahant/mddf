import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Served two ways:
//  - by the FastAPI app at "/" (default base "/")
//  - as a static GitHub Pages site (VITE_BASE=/mddf/, VITE_INFERENCE=client)
export default defineConfig({
  base: process.env.VITE_BASE || "/",
  plugins: [react(), tailwindcss()],
  build: { outDir: "dist", emptyOutDir: true },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:7860",
      "/metrics": "http://127.0.0.1:7860",
    },
  },
});
