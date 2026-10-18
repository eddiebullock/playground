import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { viteSingleFile } from "vite-plugin-singlefile";

// Browsers refuse separate module scripts on file:// pages, so code, styles and fonts are inlined.
// Videos and images stay as files next to index.html and load by relative path.
export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss(), viteSingleFile()],
  build: {
    outDir: "dist-offline",
    emptyOutDir: true,
  },
});
