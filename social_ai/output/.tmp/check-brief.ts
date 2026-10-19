import { chromium, webkit } from "playwright";

const url = "file://" + process.env.HOME + "/Documents/Gloss-brief.html";
for (const [name, engine] of [["chromium", chromium]] as const) {
  let browser;
  try {
    browser = await engine.launch();
  } catch (error) {
    console.log(name, "not available", String(error).slice(0, 120));
    continue;
  }
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.goto(url);
  await page.waitForTimeout(4000);
  const videos = await page.evaluate(() =>
    [...document.querySelectorAll("video")].map((v) => ({ ready: v.readyState, t: Number(v.currentTime.toFixed(2)), paused: v.paused })),
  );
  const height = await page.evaluate(() => document.documentElement.scrollHeight);
  console.log(name, JSON.stringify(videos), "pageHeight", height, "errors", JSON.stringify(errors));
  await page.screenshot({ path: `output/.tmp/brief-${name}.png`, fullPage: true });
  await browser.close();
}
