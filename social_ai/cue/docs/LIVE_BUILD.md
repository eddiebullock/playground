# Cue: how to build the product

This is the build spec. Implement against it. The copy-paste Reflect app
in `app/` stays as research tooling for Papers 1-2. **The product is
`live/`**: a local HUD that reads a Zoom, Teams, Meet, or video window
and gives realtime cues on three separate tabs.

Nothing in this file is sent off the machine. Perception is on-device
(MediaPipe). No paid API. No cloud emotion service.

---

## 1. What it is

The user shares the call or the video (browser `getDisplayMedia`). Cue
mirrors that picture, finds faces in it, listens to two audio channels
when it can (the call vs the user's mic), and shows a calm overlay.

Three tabs, nothing else in the first version:

| Tab | Job | Input | Output (hedged) |
|---|---|---|---|
| **Eye contact** | Where to look so the other person *feels* looked-at on a video call | Call video + optional user camera + a draggable "camera is here" marker | A target on the physical webcam; "look here, not at their tile"; if the user camera is on, whether their gaze is toward the camera or toward the screen |
| **Turn taking** | When to speak, and how long you have been speaking | Call audio = them, mic = you | THEY ARE SPEAKING / PAUSE, YOU COULD START / YOU ARE SPEAKING (timer) / BOTH / you have been talking for N seconds |
| **Emotion** | A face-shape guess, never a feeling | Largest face in the call video, blendshapes | "This face looks like it might be smiling" plus the cues (mouth, brows). Always: faces are not feelings |

The user can put Cue on a second monitor or in split view next to
Zoom. Always-on-top (Electron) is version 2, not version 1.

---

## 2. Why this, not paste-a-message

Paste-a-message is a study instrument. It is not something people open
during a call. Superpower Glass and Brain Power already proved that live
cues are what people imagine; they failed on children, parent outcomes,
and "you are angry" labels. Cue is the adult, private, local
version: the user chooses the window, cues are hedged, eye contact is
*where the webcam is* (a video-call fact), not a demand to stare at
eyes in the street.

Known limits, which the UI must say out loud:

- Barrett et al. 2019: facial configurations do not reliably equal
  emotion. The emotion tab is a face-shape read, not mind-reading.
- EU AI Act Art. 5(1)(f): emotion inference from faces at work and in
  education is banned. Show a workplace warning. Personal / research
  use only in v1.
- Community: do not score the user. Never "you are failing at eye
  contact". The eye-contact tab is a camera-target, optional.
- Bystanders: capturing a call records other people. Lab use with
  consenting confederates until PREC says otherwise. The app stays on
  127.0.0.1.

---

## 3. Architecture

```
live/
  src/
    main.tsx, App.tsx, index.css
    session/useLive.ts     capture + rAF loop
    session/capture.ts     getDisplayMedia, getUserMedia
    session/vision.ts      MediaPipe Face Landmarker (call + user)
    session/audio.ts       RMS / speech flags, two channels
    analysis/gaze.ts       camera-ward vs screen-ward (pure)
    analysis/turntaking.ts state machine (pure)
    analysis/emotion.ts    blendshape -> hedged label (pure)
    tabs/EyeTab.tsx
    tabs/TurnTab.tsx
    tabs/EmotionTab.tsx
    hud/Stage.tsx          video + overlay canvas
```

Perception stack (from `docs/research/02_tech.md`):

- **Faces, pose, blendshapes:** `@mediapipe/tasks-vision` Face
  Landmarker, LIVE/VIDEO mode, GPU, 15 fps is enough.
- **Gaze (user):** iris + `eyeLook*` blendshapes from the user camera.
  Do not claim degrees of visual angle in v1; three bins: camera-ward,
  screen-ward, away.
- **Turn-taking:** two-channel energy + hysteresis. VAP (Skantze) is
  version 2 if an ONNX sidecar is worth it. Silero VAD is an optional
  upgrade; RMS is enough to ship the tab.
- **No VLM in the frame loop.** A 2B VLM is 1-3 s/frame. MediaPipe is
  the realtime path. A local VLM can be an on-demand "what just
  happened?" button later.

Call audio on macOS: **Chrome tab share with "share audio"** is the
path that works. Sharing a Zoom *window* often has video and no audio.
The turn-taking tab must say so rather than invent "them" from the
microphone.

---

## 4. Tab behaviour in detail

### Eye contact
1. User shares the call window. The other person's tile is a face box.
2. User drags a marker to the physical webcam (usually top-centre of
   the laptop).
3. Overlay: a stable ring at that marker, labelled "they see you
   looking if you look here". A faint line from the face-tile to the
   marker.
4. Optional "use my camera": Face Landmarker on the user. If
   `eyeLookDown` dominates, they are looking at the screen; if look
   vectors are near centre, camera-ward.
5. Copy never says "make eye contact". It says where the camera is.

### Turn taking
1. `themSpeech` from display-audio RMS above threshold with 120 ms
   hangover. `youSpeech` from mic the same way.
2. States: them, you, both, silence. Floor-open after ~800 ms silence
   that followed them. Long-turn cue after 30 s of you (user-settable).
3. Big words, a timer, two level meters. No transcript in v1.

### Emotion
1. Largest face in the call frame.
2. Blendshape recipe (see `analysis/emotion.ts`): smile, frown, raised
   brows, still. Confidence is "low / some / higher" from the margin
   between the top two recipes, not from a 0-1 "happy" score.
3. Headline form: "This face looks like it might be X." Subline: the
   cues. Footer: "A face is not a feeling."

---

## 5. Implementation order

1. Shell: Vite + React, three tabs, share-screen / share-mic / share-
   camera, mirrored video. **This is already the product skeleton.**
2. Face boxes on the call video (MediaPipe).
3. Camera marker + eye-contact copy.
4. Two-channel meters + turn state machine.
5. Blendshape emotion, hedged.
6. User-camera gaze bins.
7. Tests for the three pure analysis modules.
8. Electron always-on-top and system audio only after 1-7 work in
   Chrome.

Do not wait for a language model. Do not paste Reflect into this UI.

---

## 6. Run

```
cd social_ai/cue/live
npm install
npm run dev
```

Open http://127.0.0.1:5174/ in Chrome (tab audio). Allow screen and
mic. Pick a Meet tab or a recorded video window to start.

---

## 7. Research that still sits beside this

- Papers 1-2 and the old `app/` engine stay. They answer different
  questions (pragmatics, disclosure).
- Safety for publication is E11 (`docs/research/05_frontier_safety.md`),
  a red-team of *frontier* models, not of this HUD.
- Study 5, if it happens, would be feasibility of *this* live tool, not
  of paste-a-message. That protocol needs a rewrite and PREC. Do not
  recruit until Eddie says so.
