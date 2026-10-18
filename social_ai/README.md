# Gloss — concept scenes

Visual material for a research meeting. Gloss is a fictional, on-device assistant that offers optional notes during a conversation. Harbor is a fictional work-meeting app. Nothing here is a recording of a real product, a real person, or a live model.

This pass is **Scene A**, a work video meeting, proved end to end: desktop overlay, display-glasses view, stills, and a video with generated audio. Later scenes, the settings walkthrough, and the two diagrams are not in this build yet.

## Regenerate

Requirements: Node, ffmpeg, Playwright’s Chromium, and macOS `say` (voice: Daniel, then Moira, then Samantha).

```bash
cd social_ai
npm install
npx playwright install chromium
npm run render:all
```

`npm run render:all` starts Vite, redraws every Scene A asset into `output/`, and checks contrast, copy, frame size, and that the soundtrack is present at the cue times. If Playwright’s Chromium is missing, the script installs it.

`npm run dev` plays Scene A at `http://127.0.0.1:5173/`. The bar along the top names what is happening and jumps to a moment. Meeting, YouTube, and Netflix switch the example. The practice zone is someone watching a video while Gloss offers notes: exaggeration, a paid mention, an unkind smile, a held look, and a short recap. The pictures in the meeting and the practice zone are short loops from Pexels, which allows this use. The words on screen are still the simulated script. A screenshot saved as `public/practice/custom-youtube.jpg` or `public/practice/custom-netflix.jpg` replaces that picture on reload. From inside this folder, the video file is `output/video/scene-A-meeting.mp4`.

Dialogue is on screen as subtitles. The soundtrack is the assistant only: soft earcons, rendered with the Web Audio API, and one spoken cue (“Possibly sarcastic.”). The meeting itself is not dubbed.

## Scene A assets

| File | Caption |
| --- | --- |
| `output/screenshots/scene-A-01-open.png` | Priya gives the bad news. Gloss is quiet: no suggestion yet. |
| `output/screenshots/scene-A-02-sarcasm.png` | Sam’s line, with a tentative sarcasm note, a labelled colour edge, a tone icon, and an audio cue. |
| `output/screenshots/scene-A-03-idiom.png` | “Take this offline”, explained as a suggestion, with a calm label. |
| `output/screenshots/scene-A-04-pause.png` | A silence. Gloss suggests Priya may be waiting, and shows a wristband pulse. |
| `output/screenshots/scene-A-05-topic.png` | The agenda moves to the budget, with a topic note. |
| `output/screenshots/scene-A-06-glasses-sarcasm.png` | The same sarcasm moment through display glasses. |
| `output/screenshots/scene-A-07-glasses-pause.png` | The pause, as a small heads-up panel in the lower right of the lens. |
| `output/screenshots/scene-A-08-explainer.png` | The “What is this?” card Alex can show to other people. |
| `output/screenshots/scene-A-09-quiet.png` | Quiet mode. Sam’s line is still on screen; the suggestion is held back. |
| `output/reduced-motion/scene-A-02-sarcasm.png` | The sarcasm frame with motion reduced to a still. |
| `output/reduced-motion/scene-A-04-pause.png` | The pause frame with the haptic pattern shown as a still. |
| `output/video/scene-A-meeting.mp4` | The meeting, about 35 seconds. Cues arrive in time with the subtitles. Quiet mode comes on at the end. |

Every frame carries the corner tag “Concept prototype — simulated output”.

## What the picture is doing

Feedback is a suggestion. It uses “possibly”, “may”, and “means”. It does not tell Alex how to sit, look, or seem. Colour is always paired with a word: Calm, Heads-up, or Uncertain. Quiet mode and the explainer card are part of the scene, because the user decides, and other people can be told what they are looking at.

Palette: navy `#14213D`, paper `#F7F5F0`, orange `#E07A2F`, blue `#3A7CA5`, body text `#4A5568`, uncertain `#8A94A6`. Headings are Libre Baskerville. Interface text is Public Sans. Both are bundled, so a render does not need a font CDN.
