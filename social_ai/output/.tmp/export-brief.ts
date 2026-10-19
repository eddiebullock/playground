import { chromium } from "playwright";
import { execSync } from "node:child_process";
import { mkdirSync, readdirSync, rmSync } from "node:fs";

const home = process.env.HOME!;
const url = "file://" + process.cwd() + "/dist-brief/brief.html";
const outDir = home + "/Documents/Social-glasses-brief";
const recDir = "output/.tmp/rec";
mkdirSync(outDir, { recursive: true });
rmSync(recDir, { recursive: true, force: true });

const browser = await chromium.launch();

// PDF with stills
{
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 }, deviceScaleFactor: 2 });
  await page.goto(url);
  await page.waitForFunction(() => [...document.querySelectorAll("video")].every((v) => v.readyState >= 3));
  const times = [3.5, 8.6, 8.6];
  await page.evaluate(async (times) => {
    const videos = [...document.querySelectorAll("video")];
    videos.forEach((v) => v.pause());
    await new Promise((r) => setTimeout(r, 300));
    for (const [i, v] of videos.entries()) {
      v.currentTime = times[i];
      await new Promise((r) => v.addEventListener("seeked", r, { once: true }));
    }
  }, times);
  await page.waitForTimeout(1500);
  const stages = await page.$$('[role="button"]');
  for (const [i, stage] of stages.entries()) {
    const png = await stage.screenshot({ type: "png" });
    await page.evaluate(
      ([i, data]) => {
        const el = document.querySelectorAll('[role="button"]')[i as number] as HTMLElement;
        const img = document.createElement("img");
        img.src = "data:image/png;base64," + data;
        img.style.cssText = "position:absolute;inset:0;width:100%;height:100%;z-index:50";
        el.appendChild(img);
      },
      [i, png.toString("base64")] as const,
    );
  }
  await page.evaluate(() => {
    document.querySelectorAll("button").forEach((b) => ((b.parentElement as HTMLElement).style.display = "none"));
    for (const p of document.querySelectorAll("p")) {
      if (p.textContent?.includes("Click a video to pause it.")) {
        p.textContent = "Filmed with Meta glasses. The same 19-second clip of two colleagues joking about work, shown three ways. Stills below; the moving version is in the attached video.";
      }
    }
  });
  await page.waitForTimeout(300);
  const height = await page.evaluate(() => document.documentElement.scrollHeight);
  await page.pdf({ path: outDir + "/Social-glasses-brief.pdf", width: "1200px", height: `${height + 2}px`, printBackground: true, pageRanges: "1" });
  await page.close();
}

// MP4 of the three demos playing together
{
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 }, recordVideo: { dir: recDir, size: { width: 1280, height: 800 } } });
  const page = await context.newPage();
  const born = Date.now();
  await page.goto(url);
  await page.addStyleTag({
    content: `main > header, main > div, main > section:nth-of-type(2), footer { display: none !important; }
      main { padding-top: 28px !important; } button { visibility: hidden; }`,
  });
  await page.evaluate(() => {
    for (const p of document.querySelectorAll("p")) {
      if (p.textContent?.includes("Click a video to pause it.")) {
        p.textContent = "Concept prototype, filmed with Meta glasses. The same clip of two colleagues joking about work, shown three ways.";
      }
    }
  });
  await page.waitForFunction(() => [...document.querySelectorAll("video")].every((v) => v.readyState >= 3));
  await page.waitForTimeout(800);
  await page.evaluate(() => {
    document.querySelectorAll("button").forEach((b) => b.textContent === "Restart" && b.click());
  });
  const start = (Date.now() - born) / 1000 + 0.25;
  await page.waitForTimeout(19500);
  await context.close();
  const webm = readdirSync(recDir).find((f) => f.endsWith(".webm"))!;
  execSync(
    `ffmpeg -y -loglevel error -ss ${start.toFixed(2)} -i ${recDir}/${webm} -t 18.6 -vf "crop=1120:780:80:0" -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "${outDir}/Social-glasses-three-demos.mp4"`,
  );
}

await browser.close();
console.log("done");
