import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { viteSingleFile } from "vite-plugin-singlefile";

// One emailable file: code, fonts and the joke clip are all inlined into brief.html.
export default defineConfig({
  base: "./",
  publicDir: false,
  plugins: [react(), tailwindcss(), viteSingleFile()],
  build: {
    outDir: "dist-brief",
    emptyOutDir: true,
    rollupOptions: { input: "brief.html" },
  },
});
