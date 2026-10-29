# Live mode, deployment and the wearable path

Status: design. Built after Reflect, gated by E6 (latency). Focus-group
findings folded in on 2026-10-25 (`docs/research/04_focus_groups.md`, T11).
Technical sources: `docs/research/02_tech.md`.

**What the focus groups changed.** Most participants were uneasy about
glasses: personal discomfort, the media stigma, privacy for themselves and
others, varifocal lenses, and workplace bans on recording. Live text or a
voice in the ear was seen as distracting. The preferred forms were a phone
app, an earpiece, and a ring or wristband that vibrates as a "pause and
think" cue, with input from a small microphone. Live mode therefore defaults
to the lightest possible cue (a haptic tap or a one-word chip), with details
on demand after the moment, and an earpiece option. The camera path is not
built for Study 5.

## 1. One pipeline, three hosts

```
capture  ->  signals  ->  events  ->  interpretation  ->  presentation
(tab/mic,     (VAD, ASR,   (turn end,   (same engine as     (side panel,
 glasses       prosody,     question,    Reflect: readings   glasses HUD,
 camera,       optional     long pause,  + calibrated        haptic)
 Emteq         face)        idiom/irony  scorer, abstain)
 sensors)                   candidate)
```

The interpretation step is the Reflect engine (`app/src/engine`). Live mode
adds capture, signals and events. The glasses replace capture and
presentation. So every improvement measured in E1-E4 carries into every host.

## 2. Live call companion (Chrome extension, MV3)

- **Capture.** `chrome.tabCapture` gives the call tab's audio (the other
  people). The mic in an offscreen document gives the user. Keeping the
  channels separate replaces speaker diarization in 1:1 calls and gives
  turn-taking its two streams.
- **Signals.** Silero VAD (vad-web) on both channels. Moonshine streaming ASR
  in the browser via transformers.js, with Whisper base as fallback. Pitch and
  energy per utterance from the Web Audio API. No face analysis by default.
- **Events.** Turn end: remote speech stops, the pause exceeds about 600 ms,
  and pitch falls. Question detection from the transcript. Long silence after
  a question ("the floor is yours"). An idiom or irony candidate from a cheap
  lexical filter, then confirmed by the engine.
- **Interpretation.** Called only at events, never continuously. A short
  reading: "Possibly sarcastic", "Idiom: means ...", "They may be waiting for
  your answer". It abstains below the calibrated threshold, and abstaining is
  shown as nothing at all.
- **Presentation.** A side panel. One line at a time. Colour always paired
  with a word (Calm / Heads-up / Uncertain, as in the concept scenes). Quiet
  mode. A recap on demand after the call ("3 moments you might want to look
  at"), which feeds straight into Reflect.
- **Consent.** In the study, only in calls where everyone has been told: a
  standard line the user can paste into the chat, and the explainer card from
  the concept scenes. No audio is stored. Transcripts are discarded at the
  end of the call unless the user saves a moment to Reflect.

E6 measures: end-of-turn to overlay latency (target under 2.5 s, or it arrives
too late to help), CPU/GPU load while the call runs, and false-alarm rate per
minute in role-play calls.

## 3. Zero-install deployment for Study 5

Today Cue calls a local Ollama server. Participants would have to install
Ollama and a 5 GB model. Options, in order of preference:

1. **In-browser model (WebLLM or transformers.js, WebGPU).** The participant
   opens a URL in Chrome and the model downloads once into the browser cache.
   No installer, no unsigned-app warnings. The engine needs only a second
   backend that implements `json` and `optionDist`; WebLLM's
   OpenAI-compatible API supports `logprobs` and `top_logprobs`. Gate: an 8B
   model's 4-5 GB download and VRAM. Test whether an 8B q4 model runs on a
   16 GB M-series Mac and on a typical Windows laptop.
2. **Ollama plus a one-line installer script.** Fine for the lab and for
   technically confident participants.
3. **A university-hosted model.** Strongest model, no participant hardware
   requirement. Needs an IT agreement, and data leaves the device (within the
   University), which changes the DPIA.

## 4. Wearable (Study 6 with Emteq)

- Same events and engine. The phone hosts the model, and the glasses show one
  word or give a haptic tap.
- **Inward first.** Emteq's sensors measure the wearer's own facial muscle
  activity and arousal. Supporting the user's awareness of their own state
  (load, overwhelm, a cue to take a break) fits the community's priorities
  (emotional regulation over social-skills training; Williams & Gilbert
  2020), alexithymia research, and the law. Reading other people's faces at
  work is the use the EU AI Act bans.
- **Outward, carefully.** Eye-contact and attention estimation from a
  head-worn camera works at expert level (Chong et al. 2020), but gaze
  coaching is the feature autistic critics object to most. The focus
  groups did not ask for it, and one participant explicitly did not want to
  be expected to learn eye contact (S2). Not planned.
- Research continuity: Studies 2-3 (open-weight VLM probing and calibration)
  become the multimodal scorer for this host.
