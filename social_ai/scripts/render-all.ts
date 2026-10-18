import { spawn, type ChildProcess } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { renderSceneA } from "./render-scene-a.ts";

const root = path.dirname(fileURLToPath(new URL("../package.json", import.meta.url)));
const port = 5199;
const baseUrl = `http://127.0.0.1:${port}`;

async function main() {
  const vite = startVite();
  try {
    await waitForServer(baseUrl);
    await renderSceneA(baseUrl);
    console.log("");
    console.log("Scene A assets are in output/.");
    console.log("Still to build: scenes B–D, the six-way modality set, settings, and the two diagrams.");
  } finally {
    vite.kill("SIGTERM");
  }
}

function startVite() {
  const child: ChildProcess = spawn(
    path.join(root, "node_modules", ".bin", "vite"),
    ["--host", "127.0.0.1", "--port", String(port), "--strictPort"],
    { cwd: root, stdio: ["ignore", "pipe", "pipe"] },
  );
  child.stdout?.on("data", (chunk) => process.stdout.write(chunk));
  child.stderr?.on("data", (chunk) => process.stderr.write(chunk));
  return child;
}

async function waitForServer(url: string) {
  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // Vite is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error(`Vite did not start at ${url}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
