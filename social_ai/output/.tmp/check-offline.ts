import { chromium } from "playwright";

const url = "file://" + process.env.HOME + "/Documents/Gloss-demo/index.html";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const errors: string[] = [];
page.on("pageerror", (error) => errors.push(String(error)));
page.on("console", (msg) => {
  if (msg.type() === "error") errors.push(msg.text());
});
await page.goto(url);
await page.waitForTimeout(2500);
const state = await page.evaluate(() => {
  const v = document.querySelector("video");
  return { title: document.title, videoReady: v?.readyState, videoTime: v?.currentTime, src: v?.currentSrc };
});
console.log(JSON.stringify(state));
await page.screenshot({ path: "output/.tmp/offline-joke.png" });
await page.getByRole("button", { name: "Noise cancelling for the eyes" }).click();
await page.waitForTimeout(2500);
await page.screenshot({ path: "output/.tmp/offline-focus.png" });
await page.getByRole("button", { name: "Simple glasses: meeting" }).click();
await page.waitForTimeout(2500);
const clips = await page.evaluate(() => [...document.querySelectorAll("video")].map((v) => v.readyState));
console.log("meeting videos", JSON.stringify(clips));
await page.screenshot({ path: "output/.tmp/offline-meeting.png" });
console.log("errors", JSON.stringify(errors));
await browser.close();
