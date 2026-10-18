import { execFile } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { chromium, type Browser, type Page } from "playwright";
import { SCENE_A } from "../src/content/sceneA.ts";

const exec = promisify(execFile);

function run(cmd: string, args: string[]) {
  return exec(cmd, args, { maxBuffer: 20 * 1024 * 1024 });
}

async function launchBrowser() {
  try {
    return await chromium.launch();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (!message.includes("Executable doesn't exist")) throw error;
    console.log("Playwright Chromium is not installed. Installing it...");
    await run(path.join(root, "node_modules", ".bin", "playwright"), ["install", "chromium"]);
    return chromium.launch();
  }
}

async function waitForClips(page: Page) {
  await page.waitForFunction(() => {
    const clips = Array.from(document.querySelectorAll("[data-clip]"));
    if (clips.length < 2) return false;
    return clips.every((node) => {
      if (node instanceof HTMLVideoElement) return node.readyState >= 2;
      if (node instanceof HTMLImageElement) return node.complete && node.naturalWidth > 0;
      return false;
    });
  });
}
const root = path.dirname(fileURLToPath(new URL("../package.json", import.meta.url)));
const outDir = path.join(root, "output");
const tmpDir = path.join(outDir, ".tmp");

const BANNED = [
  "youtube",
  "netflix",
  "microsoft teams",
  "zoom",
  "eye contact",
  "smile more",
  "act normal",
  "more typical",
];

type Shot = {
  file: string;
  folder: "screenshots" | "reduced-motion";
  view: "desktop" | "glasses";
  t: number;
  cue: string;
  speaker: string;
  audio: boolean;
  quiet?: boolean;
  explainer?: boolean;
  reduced?: boolean;
};

const SHOTS: Shot[] = [
  { file: "scene-A-01-open.png", folder: "screenshots", view: "desktop", t: 2.5, cue: "none", speaker: "priya", audio: false },
  { file: "scene-A-02-sarcasm.png", folder: "screenshots", view: "desktop", t: 7.6, cue: "sarcasm", speaker: "sam", audio: true },
  { file: "scene-A-03-idiom.png", folder: "screenshots", view: "desktop", t: 12, cue: "idiom", speaker: "priya", audio: true },
  { file: "scene-A-04-pause.png", folder: "screenshots", view: "desktop", t: 19.4, cue: "pause", speaker: "none", audio: false },
  { file: "scene-A-05-topic.png", folder: "screenshots", view: "desktop", t: 28.2, cue: "topic", speaker: "jordan", audio: true },
  { file: "scene-A-06-glasses-sarcasm.png", folder: "screenshots", view: "glasses", t: 7.6, cue: "sarcasm", speaker: "sam", audio: true },
  { file: "scene-A-07-glasses-pause.png", folder: "screenshots", view: "glasses", t: 19.4, cue: "pause", speaker: "none", audio: false },
  { file: "scene-A-08-explainer.png", folder: "screenshots", view: "desktop", t: 7.6, cue: "sarcasm", speaker: "sam", audio: true, explainer: true },
  { file: "scene-A-09-quiet.png", folder: "screenshots", view: "desktop", t: 7.6, cue: "none", speaker: "sam", audio: false, quiet: true },
  { file: "scene-A-02-sarcasm.png", folder: "reduced-motion", view: "desktop", t: 7.6, cue: "sarcasm", speaker: "sam", audio: true, reduced: true },
  { file: "scene-A-04-pause.png", folder: "reduced-motion", view: "desktop", t: 19.4, cue: "pause", speaker: "none", audio: false, reduced: true },
];

type EarconSpec = {
  id: string;
  dur: number;
  notes: { f: number; t: number; d: number }[];
};

const EARCONS: EarconSpec[] = [
  { id: "headsup", dur: 0.74, notes: [{ f: 523.25, t: 0.02, d: 0.34 }, { f: 392, t: 0.3, d: 0.4 }] },
  { id: "idiom", dur: 0.62, notes: [{ f: 659.25, t: 0.02, d: 0.5 }] },
  { id: "topic", dur: 0.72, notes: [{ f: 440, t: 0.02, d: 0.3 }, { f: 554.37, t: 0.28, d: 0.4 }] },
  { id: "pause", dur: 0.8, notes: [{ f: 311.13, t: 0.02, d: 0.64 }] },
];

export async function renderSceneA(baseUrl: string) {
  await assertCopy();
  assertContrast();
  await mkdir(path.join(outDir, "screenshots"), { recursive: true });
  await mkdir(path.join(outDir, "reduced-motion"), { recursive: true });
  await mkdir(path.join(outDir, "video"), { recursive: true });
  await mkdir(tmpDir, { recursive: true });

  const browser = await launchBrowser();
  try {
    await captureShots(browser, baseUrl, false);
    await captureShots(browser, baseUrl, true);
    await renderEarcons(browser);
    await renderSpeech();
    const mix = await mixAudio();
    const leadSec = await recordVideo(browser, baseUrl);
    await muxVideo(leadSec, mix);
    await verifyVideo();
  } finally {
    await browser.close();
  }
}

async function assertCopy() {
  const files = [
    "src/content/sceneA.ts",
    "src/components/GlossPanel.tsx",
    "src/components/ExplainerCard.tsx",
    "src/scenes/MeetingDesktop.tsx",
    "src/scenes/MeetingGlasses.tsx",
  ];
  for (const file of files) {
    const text = (await readFile(path.join(root, file), "utf8")).toLowerCase();
    for (const phrase of BANNED) {
      if (text.includes(phrase)) throw new Error(`${file} contains banned phrase: ${phrase}`);
    }
  }
}

function relLuminance(hex: string) {
  const n = hex.replace("#", "");
  const channels = [0, 2, 4].map((i) => {
    const c = parseInt(n.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
}

function contrast(a: string, b: string) {
  const [l1, l2] = [relLuminance(a), relLuminance(b)].sort((x, y) => y - x);
  return (l1 + 0.05) / (l2 + 0.05);
}

function assertContrast() {
  const pairs: [string, string, string][] = [
    ["#14213D", "#F7F5F0", "navy on paper"],
    ["#4A5568", "#F7F5F0", "body on paper"],
    ["#F7F5F0", "#14213D", "paper on navy"],
    ["#F7F5F0", "#1A2332", "paper on harbor"],
    ["#F7F5F0", "#243044", "names on tiles"],
    ["#D6DEE8", "#243044", "roles on tiles"],
    ["#D6DEE8", "#151D29", "agenda meta"],
    ["#F7F5F0", "#151D29", "agenda labels"],
    ["#14213D", "#EFECE4", "cue on audio well"],
  ];
  for (const [fg, bg, label] of pairs) {
    const ratio = contrast(fg, bg);
    if (ratio < 4.5) throw new Error(`${label} contrast ${ratio.toFixed(2)} is below 4.5`);
    console.log(`contrast ${label}: ${ratio.toFixed(2)}`);
  }
}

async function captureShots(browser: Browser, baseUrl: string, reduced: boolean) {
  const shots = SHOTS.filter((shot) => Boolean(shot.reduced) === reduced);
  if (shots.length === 0) return;
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1,
    reducedMotion: reduced ? "reduce" : "no-preference",
  });
  const page = await context.newPage();
  for (const shot of shots) {
    const params = new URLSearchParams({
      scene: "A",
      view: shot.view,
      t: String(shot.t),
    });
    if (shot.quiet) params.set("quiet", "1");
    if (shot.explainer) params.set("explainer", "1");
    if (shot.reduced) params.set("reduced", "1");
    await page.goto(`${baseUrl}/?${params.toString()}`, { waitUntil: "networkidle" });
    await page.waitForSelector('[data-scene-ready="true"]');
    await waitForClips(page);
    const state = await page.locator("[data-scene-ready]").evaluate((el) => ({
      cue: el.getAttribute("data-cue"),
      speaker: el.getAttribute("data-speaker"),
      audio: el.getAttribute("data-audio"),
      quiet: el.getAttribute("data-quiet"),
      explainer: el.getAttribute("data-explainer"),
    }));
    assertEqual(`${shot.file} cue`, state.cue, shot.cue);
    assertEqual(`${shot.file} speaker`, state.speaker, shot.speaker);
    assertEqual(`${shot.file} audio`, state.audio, shot.audio ? "true" : "false");
    assertEqual(`${shot.file} quiet`, state.quiet, shot.quiet ? "true" : "false");
    assertEqual(`${shot.file} explainer`, state.explainer, shot.explainer ? "true" : "false");
    const tag = page.getByText("Concept prototype — simulated output");
    if ((await tag.count()) < 1) throw new Error(`${shot.file} is missing the honesty tag`);
    const dest = path.join(outDir, shot.folder, shot.file);
    await page.screenshot({ path: dest, type: "png" });
    const size = pngSize(await readFile(dest));
    if (size.w !== 1920 || size.h !== 1080) {
      throw new Error(`${shot.file} is ${size.w}x${size.h}, expected 1920x1080`);
    }
    console.log(`wrote ${shot.folder}/${shot.file}`);
  }
  await context.close();
}

function assertEqual(label: string, got: string | null, expected: string) {
  if (got !== expected) throw new Error(`${label}: expected ${expected}, got ${got}`);
}

function pngSize(buf: Buffer) {
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

async function renderEarcons(browser: Browser) {
  const page = await browser.newPage();
  // Built at runtime so the page receives plain JavaScript, not a bundled helper.
  const renderInPage = new Function(
    "specs",
    `return (async () => {
      const out = {};
      for (const spec of specs) {
        const rate = 44100;
        const ctx = new OfflineAudioContext(1, Math.ceil(rate * spec.dur), rate);
        for (const note of spec.notes) {
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.type = "sine";
          osc.frequency.value = note.f;
          gain.gain.setValueAtTime(0.0001, note.t);
          gain.gain.exponentialRampToValueAtTime(0.34, note.t + 0.02);
          gain.gain.setValueAtTime(0.34, note.t + 0.09);
          gain.gain.exponentialRampToValueAtTime(0.0001, note.t + note.d);
          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.start(note.t);
          osc.stop(note.t + note.d + 0.02);
        }
        const rendered = await ctx.startRendering();
        const samples = rendered.getChannelData(0);
        const dataSize = samples.length * 2;
        const ab = new ArrayBuffer(44 + dataSize);
        const view = new DataView(ab);
        const write = (offset, text) => {
          for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
        };
        write(0, "RIFF");
        view.setUint32(4, 36 + dataSize, true);
        write(8, "WAVE");
        write(12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, rendered.sampleRate, true);
        view.setUint32(28, rendered.sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        write(36, "data");
        view.setUint32(40, dataSize, true);
        let offset = 44;
        for (let i = 0; i < samples.length; i++) {
          const s = Math.max(-1, Math.min(1, samples[i]));
          view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
          offset += 2;
        }
        const bytes = new Uint8Array(ab);
        let binary = "";
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
        out[spec.id] = btoa(binary);
      }
      return out;
    })();`,
  ) as (specs: EarconSpec[]) => Promise<Record<string, string>>;
  const encoded = await page.evaluate(renderInPage, EARCONS);
  await page.close();
  for (const [id, b64] of Object.entries(encoded)) {
    await writeFile(path.join(tmpDir, `earcon-${id}.wav`), Buffer.from(b64, "base64"));
    console.log(`earcon ${id}`);
  }
}

async function renderSpeech() {
  const aiff = path.join(tmpDir, "speech-sarcasm.aiff");
  const wav = path.join(tmpDir, "speech-sarcasm.wav");
  const voices = ["Daniel", "Moira", "Samantha"];
  let used = "";
  let last = "";
  for (const voice of voices) {
    try {
      await exec("say", ["-v", voice, "-r", "148", "-o", aiff, "Possibly sarcastic."]);
      used = voice;
      break;
    } catch (error) {
      last = error instanceof Error ? error.message : String(error);
    }
  }
  if (!used) throw new Error(`speech synthesis failed: ${last}`);
  await run("ffmpeg", ["-y", "-i", aiff, "-ac", "1", "-ar", "44100", wav]);
  console.log(`speech voice ${used}`);
}

type Clip = { file: string; delayMs: number; volume: number };

async function mixAudio() {
  const clips: Clip[] = [];
  for (const cue of SCENE_A.cues) {
    if (!cue.audio) continue;
    clips.push({
      file: path.join(tmpDir, `earcon-${cue.audio.earcon}.wav`),
      delayMs: Math.round(cue.start * 1000),
      volume: 0.85,
    });
    if (cue.audio.speech) {
      clips.push({
        file: path.join(tmpDir, "speech-sarcasm.wav"),
        delayMs: Math.round((cue.start + 0.28) * 1000),
        volume: 0.9,
      });
    }
  }
  const mix = path.join(tmpDir, "scene-a-mix.wav");
  const args = ["-y", "-f", "lavfi", "-t", String(SCENE_A.duration), "-i", "anullsrc=r=44100:cl=mono"];
  for (const clip of clips) args.push("-i", clip.file);
  const filters = [
    "[0]aformat=sample_fmts=fltp:channel_layouts=mono[base]",
    ...clips.map((clip, index) => {
      return `[${index + 1}]aformat=sample_fmts=fltp:channel_layouts=mono,adelay=${clip.delayMs}:all=1,volume=${clip.volume}[c${index}]`;
    }),
  ];
  const mixInputs = ["[base]", ...clips.map((_, index) => `[c${index}]`)].join("");
  filters.push(
    `${mixInputs}amix=inputs=${clips.length + 1}:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.9[out]`,
  );
  args.push(
    "-filter_complex",
    filters.join(";"),
    "-map",
    "[out]",
    "-ac",
    "2",
    "-ar",
    "44100",
    mix,
  );
  await run("ffmpeg", args);
  console.log("mixed audio");
  return mix;
}

async function recordVideo(browser: Browser, baseUrl: string) {
  const videoDir = path.join(tmpDir, "video");
  await mkdir(videoDir, { recursive: true });
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1,
    recordVideo: { dir: videoDir, size: { width: 1920, height: 1080 } },
  });
  const page = await context.newPage();
  const started = Date.now();
  await page.goto(`${baseUrl}/?scene=A&view=desktop&play=1&hold=1&motion=1`, { waitUntil: "networkidle" });
  await page.waitForSelector('[data-scene-ready="true"]');
  await waitForClips(page);
  await page.waitForTimeout(180);
  const leadSec = (Date.now() - started) / 1000;
  await page.evaluate(() => window.__start?.());
  await page.waitForTimeout(SCENE_A.duration * 1000 + 500);
  const video = page.video();
  if (!video) throw new Error("Playwright did not record a video");
  await page.close();
  await context.close();
  const webm = await video.path();
  console.log(`recorded video, lead ${leadSec.toFixed(3)}s`);
  return { leadSec, webm };
}

async function muxVideo(recorded: { leadSec: number; webm: string }, mix: string) {
  const dest = path.join(outDir, "video", "scene-A-meeting.mp4");
  await run("ffmpeg", [
    "-y",
    "-i",
    recorded.webm,
    "-i",
    mix,
    "-filter_complex",
    `[0:v]trim=start=${recorded.leadSec.toFixed(3)}:duration=${SCENE_A.duration},setpts=PTS-STARTPTS[v];[1:a]atrim=duration=${SCENE_A.duration},asetpts=PTS-STARTPTS[a]`,
    "-map",
    "[v]",
    "-map",
    "[a]",
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-r",
    "30",
    "-crf",
    "18",
    "-c:a",
    "aac",
    "-b:a",
    "160k",
    "-movflags",
    "+faststart",
    dest,
  ]);
  console.log("wrote video/scene-A-meeting.mp4");
}

async function verifyVideo() {
  const dest = path.join(outDir, "video", "scene-A-meeting.mp4");
  const { stdout } = await run("ffprobe", [
    "-v",
    "error",
    "-select_streams",
    "v:0",
    "-show_entries",
    "stream=width,height,duration",
    "-of",
    "csv=p=0",
    dest,
  ]);
  const [w, h, duration] = String(stdout).trim().split(",");
  if (w !== "1920" || h !== "1080") throw new Error(`video is ${w}x${h}`);
  const seconds = Number(duration);
  if (seconds < SCENE_A.duration - 0.4 || seconds > SCENE_A.duration + 0.4) {
    throw new Error(`video duration ${seconds}s, expected ${SCENE_A.duration}s`);
  }
  const audio = await run("ffprobe", [
    "-v",
    "error",
    "-select_streams",
    "a:0",
    "-show_entries",
    "stream=codec_name",
    "-of",
    "csv=p=0",
    dest,
  ]);
  if (!String(audio.stdout).includes("aac")) throw new Error("video is missing an aac audio stream");

  const windows = [
    { name: "opening", ss: 1.0, dur: 1.0, sound: false },
    { name: "sarcasm", ss: 6.9, dur: 1.2, sound: true },
    { name: "idiom", ss: 10.84, dur: 0.35, sound: true },
    { name: "gap", ss: 16.3, dur: 0.8, sound: false },
    { name: "pause", ss: 19.3, dur: 0.8, sound: false },
    { name: "topic", ss: 27.04, dur: 0.35, sound: true },
  ];
  for (const window of windows) {
    const level = await meanVolume(dest, window.ss, window.dur);
    console.log(`audio ${window.name} @ ${window.ss}s mean ${level.toFixed(1)} dB`);
    if (window.sound && level < -42) {
      throw new Error(`${window.name} should be audible, mean ${level.toFixed(1)} dB`);
    }
    if (!window.sound && level > -38) {
      throw new Error(`${window.name} should be quiet, mean ${level.toFixed(1)} dB`);
    }
  }
}

async function meanVolume(file: string, ss: number, dur: number) {
  const result = await run("ffmpeg", [
    "-ss",
    String(ss),
    "-t",
    String(dur),
    "-i",
    file,
    "-af",
    "volumedetect",
    "-f",
    "null",
    "-",
  ]);
  const stderr = String(result.stderr ?? "");
  const match = stderr.match(/mean_volume:\s*(-?[\d.]+)\s*dB/);
  if (!match) throw new Error(`could not read volume at ${ss}s`);
  return Number(match[1]);
}
