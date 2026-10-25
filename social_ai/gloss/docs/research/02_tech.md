# 02 - Technical landscape: privacy-first, zero-cost social-interaction assistant

Scouting date: 2026-10-24. Dev machine: Apple M3, 16 GB unified RAM, ~20 GB free disk, Ollama (no models), Node 24, Python 3.13, ffmpeg.
Hard constraint: no money spent. Delivery: browser app/extension or Electron/Tauri desktop app on participants' own laptops; later Emteq smart glasses.

Verification legend: **[V]** = verified on a primary source (vendor docs, release page, paper) during this scan. **[S]** = secondary source only (blog/aggregator). **[U]** = unverified / my estimate; benchmark before relying on it.

---

## 0. Decision-relevant headlines

1. **Everything needed for a first prototype can run locally and free.** WebGPU is now in all three major browsers, transformers.js v4 covers ASR, VLMs and Gemma 4 audio-image-text in the browser, MediaPipe gives face blendshapes at webcam frame rate, and Ollama/llama.cpp can run audio-capable small omni models (Gemma 4 E2B/E4B) on the M3.
2. **Free cloud tiers are effectively ruled out for UK participants.** Google's Gemini API terms say apps made available to users in the UK/EEA/Switzerland may use **only Paid Services**, and unpaid-tier data may be read by human reviewers and used to improve Google products; the terms also say not to send personal information to Unpaid Services [V]. That is incompatible with both the no-money and the ethics constraints. HF free credit is $0.10/month [V], which is negligible. Groq is the only plausible free fallback, and its no-retention claim is [S].
3. **1:1 video calls give you speaker separation for free.** Tab audio = the other person(s); microphone = the user. That two-channel signal is exactly the stereo input Voice Activity Projection (VAP) expects, and it removes the need for in-browser diarization (which is still immature) for the 1:1 case.
4. **Chrome's built-in Gemini Nano is attractive but fragile for a study.** Prompt API is stable in extensions since Chrome 138 and on the web from Chrome 148; image and audio input are documented [V]. But it needs at least 22 GB of free disk on the profile volume and either a GPU with more than 4 GB VRAM or 16 GB RAM [V]. Your M3 currently has ~20 GB free, so it would not qualify without cleanup, and many participants' 8 GB laptops will not qualify either. Chrome only; no Safari/Firefox.
5. **Social-signal models are research-grade, not product-grade.** Turn-taking (VAP) is about 76% balanced accuracy on shift/hold [V]; MUStARD multimodal sarcasm detection tops out around 80 F1 under 5-fold CV [S]; speech emotion recognition is typically 4-class acted/elicited data. Present outputs to participants as hedged suggestions, never verdicts.

---

## 1. In-browser ML

### 1.1 WebGPU support (as of late 2025 to 2026)
Per web.dev (25 Nov 2025) [V]: https://web.dev/blog/webgpu-supported-major-browsers
- **Chrome/Edge**: since v113 on Windows, macOS and ChromeOS; Android since v121 (Android 12+, Qualcomm/ARM GPUs). Linux still rolling out.
- **Safari 26**: macOS Tahoe 26, iOS/iPadOS/visionOS 26. Participants on older macOS will not have it.
- **Firefox**: Windows from 141; Apple-silicon Macs on macOS Tahoe from 145. No Linux, Android or Intel Mac yet.
- Implication: target **Chrome/Edge first**, with WASM fallback (10-15x slower for transformer models [S], e.g. MiniLM embedding 386 ms WASM vs 24 ms WebGPU on a Mac [S], https://huggingface.co/spaces/Xenova/webgpu-embedding-benchmark/discussions/1).

### 1.2 transformers.js (Hugging Face) - current v4.3.1 (7 Oct 2026) [V]
Releases: https://github.com/huggingface/transformers.js/releases. Licence Apache-2.0.
- v4.0.0 (30 Mar 2026): new C++ WebGPU runtime built with the ONNX Runtime team, tested across about 200 architectures; also runs in Node/Bun/Deno. Added **Qwen2.5-VL, Qwen3-VL, Qwen3.5**, Gemma 3 VLM, **LFM2-VL**, **Voxtral Realtime** (streaming ASR), Granite Speech, Cohere ASR [V].
- v4.0.1/4.1.0 (Apr 2026): **Gemma 4**, SmolVLM re-enabled [V].
- v4.3.0 (16 Sep 2026): structured output (JSON-schema/regex constrained generation, experimental), Safari 26+ WebGPU support, fixes to Whisper, Moonshine, and the KV cache for **Gemma 3n and Gemma 4 image-audio-text** models [V]. So audio-in multimodal LLMs run in the browser through this library.
- Earlier: Parakeet CTC (3.7.6), Moonshine, Whisper, **pyannote-segmentation-3.0** (frame-level speaker segmentation) [V].
- A claim of GPT-OSS-20B at about 60 tok/s on an M4 Max via WebGPU [S]; M3 base will be far slower.

### 1.3 WebLLM / MLC
`@mlc-ai/web-llm` latest seen 0.2.8x (0.2.83 reported 2026-04-24 [S]); Apache-2.0; OpenAI-compatible API; WebGPU only; prebuilt Llama 3.x, Qwen2/2.5/3, Gemma 2, Phi including a vision variant [S]. https://github.com/mlc-ai/web-llm. Good for text LLMs with fast decoding; transformers.js v4 now has wider multimodal coverage, so WebLLM is mainly a speed option for the text "interpreter" LLM.

### 1.4 ONNX Runtime Web
The backend under transformers.js; use directly for custom exports (VAP, emotion2vec, L2CS gaze). WebGPU and WASM EPs. Example: L2CS-Net gaze at about 15-20 FPS in browser via ORT-Web WebGPU [S] (https://mmla.gse.harvard.edu/tools/l2cs-net).

### 1.5 MediaPipe Tasks (Web) - `@mediapipe/tasks-vision`, `@mediapipe/tasks-audio`
Apache-2.0. Docs: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/web_js
- **Face Landmarker**: 478 3D landmarks, **52 ARKit-style blendshape scores**, facial transformation matrix (head pose), LIVE_STREAM mode with GPU delegate [V]. Real-time on laptops (commonly reported 30+ FPS on Apple silicon [U]). The blendshapes (browInnerUp, mouthSmile, eyeSquint, jawOpen etc.) map naturally onto Emteq-style facial EMG/expression channels, which is useful continuity for the glasses phase.
- Also: Gesture Recognizer, Pose Landmarker, Hand Landmarker, **Audio Classifier** (YAMNet, 521 AudioSet classes, e.g. laughter, speech, music) [V from docs; latency U]. Exact latest npm version was not confirmed in this scan (check npm).

### 1.6 Chrome built-in AI (Gemini Nano)
https://developer.chrome.com/docs/ai/prompt-api and https://developer.chrome.com/docs/ai/built-in-apis [V]
- **Prompt API**: stable in Chrome Extensions since **Chrome 138**; on the web listed at **Chrome 148**. Sampling parameters (temperature/topK) are extensions-only; on the web there is an origin trial for `samplingMode`.
- **Multimodal**: `expectedInputs` of text, **image** (canvas, VideoFrame, ImageBitmap, etc.) and **audio** (AudioBuffer, Blob, ArrayBuffer), output text only; MDN says "first released in Chrome 148" [V] (https://developer.mozilla.org/en-US/docs/Web/API/Prompt_API/Multimodal). **Audio input requires a GPU**.
- **Requirements**: Windows 10/11, macOS 13+, Linux, Chromebook Plus; **not Android/iOS**; at least **22 GB free disk**; GPU with more than 4 GB VRAM, or CPU with 16 GB RAM and 4+ cores; English, Japanese, Spanish, German, French [V].
- Summarizer, Translator, Language Detector stable since 138; Writer/Rewriter/Proofreader in trials [V].
- Pros: zero download size for your app, on-device, good for brief text interpretation. Cons: hardware gating excludes many participants, model version controlled by Google (reproducibility problem for a study), Chrome only, latency not published (my estimate: a few hundred ms to first token, 10-30 tok/s on M-series [U]).

### 1.7 Browser ASR
- **Moonshine v2 streaming** (Useful Sensors / Moonshine AI; paper arXiv 2602.12241, Feb 2026) [V for paper]: sliding-window attention encoder for bounded time-to-first-token. Vendor benchmarks (native, MacBook Pro) [S]: Tiny-Streaming 34M params, 12.0% WER, **34 ms** end-of-speech-to-final-transcript; Small-Streaming 123M, 7.84% WER, **73 ms**; Medium-Streaming 245M, 6.65% WER, **107 ms**; versus Whisper Small 1,940 ms and Whisper Large-v3 11,286 ms on the same metric (https://www.mintlify.com/moonshine-ai/moonshine/models/benchmarks). Browser package MoonshineJS (`@usefulsensors/moonshine-js`) has a `MicrophoneTranscriber` with built-in VAD; **no browser latency numbers published** [U]. Licence: English models MIT as far as I know; non-English models under a separate community licence [U - check model cards].
- **Whisper** via transformers.js WebGPU (tiny/base/small; "whisper-webgpu" realtime demo). Not natively streaming; chunked decoding adds 1-3 s lag [U]. MIT.
- **Voxtral Realtime** (Mistral, in transformers.js v4) and **Parakeet CTC** (NVIDIA, CC-BY-4.0) are newer options [V for availability; latency U].
- Recommendation: Moonshine Small/Medium streaming for English live captions; Whisper small/turbo offline for post-hoc accuracy.

### 1.8 Speaker diarization in browser
- transformers.js supports **pyannote-segmentation-3.0** (frame-level, up to 3 local speakers per window) [V]. Full diarization also needs speaker embeddings plus clustering; no complete browser pipeline found [V absence]. A Rust reimplementation of pyannote community-1 (`speakrs`, ORT/CoreML) exists for desktop apps [S].
- For video calls, **channel separation (tab vs mic) replaces diarization** for the user-vs-others split. Multi-party calls: Meet caption DOM carries speaker names (brittle, see 4.3).

### 1.9 Voice activity detection
`@ricky0123/vad-web` (ISC licence) runs **Silero VAD** through ONNX Runtime Web; Silero v5 is opt-in (`model: "v5"`), 512-sample frames (32 ms at 16 kHz) [V] (https://docs.vad.ricky0123.com/user-guide/silero-v5/). Silero model is MIT, about 2 MB, sub-millisecond per frame on CPU [U]. The old `@ricky0123/vad` package is deprecated.

---

## 2. Small open VLMs / omni models on a 16 GB Mac

Memory rule of thumb on 16 GB unified memory: macOS plus browser take 5-7 GB, so keep model weights at or below about 6-7 GB, and keep at most two models on the ~20 GB free disk.

| Model | Params | Modalities in | Runtimes (verified) | Size (Q4-ish) | Licence |
|---|---|---|---|---|---|
| **Gemma 4 E2B** | 2.3B effective / 5.1B with embeddings | text, image, video, **audio** | Ollama `gemma4:e2b` (4.6-7.5 GB, 128K ctx); llama.cpp mtmd lists E2B/E4B as **audio+vision** [V]; transformers.js v4 [V] | 4.4 GB GGUF [S] | Apache-2.0 [V] |
| **Gemma 4 E4B** | 4.5B effective / about 8B | text, image, video, **audio** | Ollama `gemma4:e4b` (6.6-9.5 GB); llama.cpp; transformers.js | 6.3 GB GGUF [S] | Apache-2.0 |
| Gemma 4 12B "unified" | 12B | incl. audio [S] | Ollama `gemma4:12b` 7.7-8.0 GB | tight on 16 GB | Apache-2.0 |
| **Qwen3-VL 2B / 4B** | 2B / 4B | text, image, video | Ollama `qwen3-vl:2b` 1.9 GB, `:4b` 3.3 GB, 256K ctx [V]; transformers.js v4 [V]; MLX | 1.9 / 3.3 GB | Apache-2.0 [U] |
| Qwen2.5-VL 3B | 3B | image, video | llama.cpp, Ollama, transformers.js | about 3 GB | Qwen research licence for 3B [U] |
| **SmolVLM2** 256M/500M/2.2B | tiny | image, video | llama.cpp [V], transformers.js [V] | 0.2-2 GB | Apache-2.0 |
| Moondream 2 (2025-04-14) | 1.9B | image | llama.cpp [V] | about 1.7 GB | Apache-2.0 |
| Moondream 3 / 3.1 | 9B MoE, 2B active | image | own runtime | about 5+ GB Q4 [U] | Moondream licence, commercial restrictions [S] |
| Qwen2.5-Omni 3B / 7B | 3B / 7B | text, image, **audio**, video | llama.cpp mtmd [V] | 3-5 GB | 3B: research-only [S]; 7B Apache-2.0 |
| Qwen3-Omni 30B-A3B | 30B MoE | omni | llama.cpp [V] | too large for 16 GB | Apache-2.0 |
| MiniCPM-o 4.5 | 9B (Qwen3-8B + Whisper-medium + SigLIP2 + CosyVoice2) | omni, **full-duplex** | GGUF Q4_K_M; `llama.cpp-omni` fork with WebRTC demo on Mac [S] | about 5-6 GB [U] | MiniCPM model licence (free registration for commercial) [U] |
| Phi-4-multimodal | 5.6B | text, image, audio | HF transformers; llama.cpp audio not confirmed [U] | about 4 GB | MIT |
| Voxtral Mini 3B | 3B | audio (+text) | llama.cpp mtmd [V], transformers.js (Realtime) | about 2.5 GB | Apache-2.0 |
| Ultravox 0.5 1B/8B | 1B/8B | audio | llama.cpp [V] | 1-5 GB | MIT |

Sources: https://ollama.com/library/gemma4, https://ollama.com/library/qwen3-vl, https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md, https://simonwillison.net/2026/Apr/2/gemma-4/, https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf. Note that llama.cpp calls audio support "highly experimental, may have reduced quality" [V]. Ollama's gemma4 tag table lists only Text/Image even though the model info says E2B/E4B accept audio, so audio through Ollama specifically is [U]; use llama-server/llama-mtmd-cli or MLX-VLM (Gemma 3n audio confirmed in MLX-VLM [S]) for audio tests.

**Latency per frame on M3 16 GB [U - extrapolated, benchmark these first]:**
- Qwen3-VL-4B on an M4 (10-core GPU) via MLX: about 33 tok/s decode, TTFT about 2.6 s at 1k context [S] (https://omlx.ai/benchmarks/xwbhcddm). Expect M3 (10-core GPU, lower bandwidth) about 20-30 tok/s, **1.5-3 s per image** for a short (about 30-token) description.
- Qwen3-VL-2B / Gemma 4 E2B: about 0.7-1.5 s per frame with short output [U].
- SmolVLM-500M via llama-server: the ngxson realtime-webcam demo runs locally on a MacBook M3 with a configurable 100 ms - several s interval; no published FPS [S] (https://github.com/ngxson/smolvlm-realtime-webcam). Practically about 2-4 captions/s [U].
- Audio into Gemma 4 E2B/E4B: audio is tokenised at about 6.25 tokens/s of audio [S for Gemma 3n], so a 10 s clip is about 63 tokens; prompt processing is cheap, so expect **1-3 s** to an answer for a 10-20 s clip [U].
- Conclusion: VLMs are **not** frame-rate perception engines. Use MediaPipe for continuous face signals (30 FPS) and call a VLM/omni model at most every few seconds or on events (turn end, long silence, user request).

**Prosody-capable (audio-in) local models**: Gemma 4 E2B/E4B (best licence and size fit), Qwen2.5-Omni 7B (Apache, about 5 GB Q4), MiniCPM-o 4.5, Voxtral Mini, Phi-4-multimodal. No published speech-emotion or prosody benchmark was found for Gemma 3n/4 [V absence]; that is an evaluation gap you would need to fill yourself (e.g. on IEMOCAP/MSP-Podcast subsets, or the EmoBox protocol).

---

## 3. Social-signal models

### 3.1 Speech emotion recognition / prosody
- **emotion2vec+** (Alibaba DAMO / SJTU; repo MIT [V]; weights licence not stated in repo [U]): seed, base (about 90M params, 4,788 h pseudo-labelled), large (about 300M, 42,526 h). 9 classes: angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown. Runs through FunASR (`pip install funasr`); no official ONNX [V] (https://github.com/ddlBoJack/emotion2vec). The original emotion2vec paper reports SOTA linear-probe results on IEMOCAP among SSL models; IEMOCAP 4-class weighted accuracy around 72-77% for SSL+linear setups is the usual range [U - check https://arxiv.org/abs/2312.15185]. In-browser use would need an ONNX export of a data2vec-style encoder (feasible, unverified).
- **wav2vec2/WavLM SER** fine-tunes on Hugging Face (e.g. `superb/wav2vec2-base-superb-er`, 4-class IEMOCAP, about 63-65% accuracy [U]); these are already ONNX-exportable and run in transformers.js as audio-classification pipelines.
- Dimensional (arousal/valence/dominance) models such as the audEERING wav2vec2 MSP-Podcast model are more informative for "tone" feedback than categorical labels, but that model is CC-BY-NC-SA (fine for research, not for a product) [U].
- Simple, explainable prosody features (pitch contour, speaking rate, loudness, pause length) computed in the browser with Web Audio plus a YIN/CREPE-tiny pitch tracker are arguably more useful for autistic users who want to know *why* something sounded a certain way. Recommend combining them.

### 3.2 Sarcasm detection
- **MUStARD** (690 TV clips, text+audio+video). Best supervised multimodal attention models reach about **80 F1** (5-fold CV) per a 2025 systematic review [S] (https://arxiv.org/pdf/2509.04605). Zero-shot text-only LLM method (BiCAL preprint) reports **0.787 macro-F1** [S]. A May 2026 preprint claims +3.14 macro-F1 over the best multimodal baseline [S]. On MUStARD++: text 69.1, audio 67.5, text+audio 70.8 F1 [S] (https://arxiv.org/pdf/2506.00955). LoRA-fine-tuned audio MLLMs: 71.6% F1 English spoken sarcasm (different dataset) [S].
- GPT-4-class models struggle with audiovisual incongruity [S]. Small local models will be worse. Treat sarcasm detection as "possible non-literal meaning, here is why" with cue explanations (contradiction between words and tone, exaggeration, context), and evaluate with autistic participants rather than relying on MUStARD numbers (sitcom data, laugh-track bias).

### 3.3 Turn-taking: Voice Activity Projection (VAP)
- Official repo (PyTorch, stereo input, one channel per speaker): https://github.com/ErikEkstedt/VoiceActivityProjection. Realtime implementation: https://github.com/inokoj/VAP-Realtime (Inoue, Jiang, Ekstedt, Kawahara, Skantze, arXiv 2401.04868).
- Realtime results [V] (Japanese travel-agency dialogues, Intel Xeon CPU, 50 Hz frames): shift/hold **balanced accuracy 76.2%** with 1 s context, **14.6 ms per frame, real-time factor 0.73**; 0.5 s context 75.4% at RTF 0.66; 20 s context is slower (RTF 13.7) and no better. English (Switchboard) and Mandarin models reported as "similar". Multilingual model (English/Mandarin/Japanese) performs comparably to monolingual ones; monolingual models do not transfer across languages [V abstract].
- Fit: tab-audio plus mic gives exactly the two channels. An M3 should run the 1 s context model comfortably in a Python sidecar; an ONNX/browser port is plausible (CPC encoder plus small transformers) but unverified effort. Use cases: "they may be about to speak, pause" or "this is a good moment to respond" cues; backchannel timing.

### 3.4 Backchannel prediction
VAP-derived backchannel prediction and dedicated models exist (e.g. Skantze group, Kyoto Kawahara lab; MM-VAP multimodal variants) [U - no current benchmark verified in this scan]. Practical approach: VAP short-horizon probability of the other speaker resuming, plus pause length, plus prosodic cues (falling pitch). Expect modest F1 (0.3-0.6 on frame-level tasks in the literature [U]).

### 3.5 Gaze / eye contact
- **Chong et al. 2020, Nature Communications**: eye-contact detection from a camera worn by the adult examiner (point-of-view, glasses-like) trained on 4.34M frames from 103 children (57 autistic); **precision 0.936, recall 0.943**, on par with 10 trained human coders (0.918 / 0.946) [V] (https://pmc.ncbi.nlm.nih.gov/articles/PMC7736573/). Directly relevant to the Emteq glasses phase, but it detects the *partner* looking at the wearer, and was trained on children.
- Webcam gaze: L2CS-Net (ResNet-50, Gaze360-trained) about 15-20 FPS in browser via ORT-Web WebGPU [S]; ETH-XGaze leaderboard best about 3.1 degrees angular error (ensemble) [V]. MediaPipe iris landmarks plus head pose give a cheap "looking at camera/screen" proxy. Webcam methods remain below commercial eye trackers [V, Frontiers 2024].
- **Ethics note**: eye-contact feedback is contentious in the autistic community (pressure to mask). Make it opt-in, framed as information about the other person ("they are looking at the shared screen") rather than coaching the user to make eye contact.

---

## 4. Capture paths for a browser tool

### 4.1 Chrome extension (Manifest V3) - recommended for video calls
- From **Chrome 116**, the service worker calls `chrome.tabCapture.getMediaStreamId({targetTabId})` after a user gesture (toolbar click), passes the ID to an **offscreen document** (`chrome.offscreen.createDocument`, reason `USER_MEDIA`), which calls `getUserMedia({audio/video: {mandatory: {chromeMediaSource: 'tab', chromeMediaSourceId}}})` [V] (https://developer.chrome.com/docs/extensions/mv3/screen_capture). You must route captured audio back to an `AudioContext` destination, otherwise the user stops hearing the call [V].
- Gives the call tab's audio (other participants) and video (their tiles) at full quality with no screen-picker dialog; works for Meet, Zoom web client and Teams web. The offscreen document can also open the microphone (permission prompt) for the user's own channel. Heavy inference should run in the offscreen document or a worker with WebGPU.
- Reference implementation: https://github.com/recallai/chrome-recording-transcription-extension.

### 4.2 Web app with getDisplayMedia
- `getDisplayMedia({video: true, audio: true})`: Chrome lets the user pick a tab with "share tab audio". **System audio on macOS** appeared in Chrome around v141, built on Core Audio taps (macOS 14.2+) [S] (https://blog.addpipe.com/getdisplaymedia-allows-capturing-the-screen-with-system-sounds-on-chrome-on-macos/). Safari and Firefox do not capture tab/system audio on macOS [U]. This is how you would capture the **native Zoom/Teams desktop apps**, which a tab-capture extension cannot reach.
- The picker must be re-confirmed each session; fine for a study.

### 4.3 Live captions
- Chrome's OS-level Live Caption is **not exposed to extensions or pages** [U - no API found]. Meet's own captions can be read from the DOM by a content script (existing open-source extensions do this) but it is brittle to UI changes and lossy (paraphrased, truncated, inconsistent punctuation), though it carries speaker names [S] (https://recall.ai/blog/how-to-get-transcripts-from-google-meet-developer-edition). Useful as a cheap multi-party speaker-label source, not as the primary ASR.

### 4.4 Desktop app (Electron/Tauri) and macOS constraints
- **Electron** `desktopCapturer` with `audio: 'loopback'`; since Electron 39 Chromium uses Apple's **Core Audio Tap** API by default; macOS 14.2+ needs `NSAudioCaptureUsageDescription` in Info.plist or the audio fails silently [V] (https://www.electronjs.org/docs/latest/api/desktop-capturer).
- **Tauri**: no official system-audio plugin; use the `screencapturekit` Rust crate (v8.0.1, includes a Tauri 2 example) or Core Audio taps; open-source Meetily (Tauri+Rust) is a working reference [S]. ScreenCaptureKit audio-only still requires the Screen Recording permission; Apple recommends Core Audio taps for audio-only [S, Apple forums].
- Unsigned apps: without a paid Apple Developer ID ($99/yr, which violates the no-money rule) participants hit Gatekeeper warnings and must right-click to open. **This is a real argument for a Chrome extension**, which can be loaded unpacked or published to the Chrome Web Store for a one-time $5 fee (still money; unpacked loading costs nothing).

### 4.5 In-person mode
`getUserMedia` on a phone browser (Chrome Android has WebGPU; iOS Safari 26 has WebGPU) or the laptop. No channel separation, so you need VAD plus diarization (pyannote-segmentation in browser) or a "tap when the other person speaks" UI. Bystander consent is the main issue.

### 4.6 Privacy / consent UX patterns
- Visible always-on recording indicator plus one-click pause; Chrome's own tab-capture indicator is shown automatically.
- Consent script for the other party (Meet/Zoom bots announce themselves; an assistive tool should at least offer an "I am using a captioning/interpretation aid" message the user can paste).
- Process in memory, store nothing by default; explicit opt-in "save this conversation for reflection", with local-only encrypted storage (IndexedDB) and a delete-all button.
- Show which models run where (badge: "on this device"); no network calls after model download (can be enforced with extension CSP and verified by participants in DevTools).
- For the study: log only derived features/consented events, not raw audio/video.

---

## 5. Free-tier cloud fallback (and why it is mostly a no)

| Service | Free allowance | Data terms | Verdict |
|---|---|---|---|
| **Gemini API (AI Studio)** | Per-project RPM/TPM/RPD, shown only in AI Studio; no published table [V]; Flash/Flash-Lite in the low thousands RPD and Pro models effectively removed from free tier since April 2026 [S] | Unpaid: Google "uses the content you submit... to provide, improve, and develop" products; **human reviewers may read** input/output; "**Do not submit sensitive, confidential, or personal information to the Unpaid Services**"; must be 18+; "**You may use only Paid Services when making API Clients available to users in the EEA, Switzerland or UK**" [V] (https://ai.google.dev/gemini-api/terms) | **Not usable** for a UK participant-facing tool; at most for the developer's own testing on synthetic data |
| **Groq** | e.g. llama-3.1-8b-instant 30 RPM / 6k TPM / 14.4k RPD; llama-3.3-70b 30 RPM / 12k TPM / 1k RPD [S] | Reported: no retention of inference I/O by default, Zero Data Retention on request, no training [S - verify https://groq.com/privacy-policy] | Possible text-only fallback with ethics-committee sign-off; limited multimodal; US processing (GDPR transfer issue) |
| **HF Inference Providers** | **$0.10/month** for free accounts, stops when exhausted [V] (https://huggingface.co/docs/inference-providers/pricing) | Depends on the routed provider | Negligible |

Ethics implication: free tiers are paid for with data or carry no contractual guarantees. With autistic participants' conversations (special category data, third-party bystanders, likely UK GDPR plus university ethics), **local-only should be the design rule**, with cloud only for researcher-side analysis on consented, de-identified text if ever.

---

## 6. Recommended architectures

### 6(a) Live video-call companion

**Option A1 (recommended): Chrome MV3 extension, fully in-browser**
- Capture: tabCapture (remote audio+video) plus mic in the offscreen document; channels kept separate.
- Pipeline (all WebGPU/WASM in the offscreen doc/workers):
  - Silero VAD (vad-web) on both channels, 32 ms frames, so you get speaking/pausing state at under 50 ms.
  - Streaming ASR: Moonshine Small/Medium streaming (or Voxtral Realtime / Whisper base fallback). Expected final-transcript latency **0.2-1 s after end of utterance in browser** [U; native is 73-107 ms].
  - MediaPipe Face Landmarker on downscaled tab video (the other person's tile, 5-15 FPS is enough) for blendshapes, smile/frown/brow, head nods; optional on the user's own webcam for self-reflection only if the user wants it.
  - Prosody: Web Audio pitch/energy/rate features per utterance (instant); optional wav2vec2 SER ONNX per utterance (about 100-300 ms [U]).
  - Interpretation LLM, called at turn ends or on user tap: Gemini Nano Prompt API when available (zero download), else transformers.js Gemma 4 E2B (q4, about 2-3 GB download, audio-capable) or WebLLM Qwen3-1.7B/4B text-only. Expected **1-3 s** to a short hint [U].
- Turn-taking: start with heuristic (VAD pause length plus pitch fall); VAP ONNX port as a research add-on.
- Pros: one install, no signing cost, works on Windows and Mac, no native permissions beyond mic. Cons: Chrome/Edge only; WebGPU memory limits; first-load model download (budget 1-3 GB); native Zoom/Teams apps not covered (participants must use the web client).

**Option A2: Extension front end plus local sidecar (Ollama/llama-server plus Python) on the participant's or lab laptop**
- Same capture; frames/audio sent to `localhost` (llama-server with Gemma 4 E4B for audio+image; Python with VAP-Realtime, emotion2vec+ via FunASR, pyannote).
- Best quality and the fastest research iteration on the M3 today. Latency: VAP 15 ms/frame; emotion2vec+ base per utterance about 100-200 ms [U]; Gemma 4 E4B per-turn audio+frame reasoning 2-4 s [U].
- Cons: install burden for participants (Ollama plus models 5-10 GB), unsigned-binary friction, 16 GB minimum RAM. Good for lab sessions and Wizard-of-Oz studies, poor for take-home deployment.

**Option A3: Electron/Tauri desktop app with system-audio capture**
- Needed only if participants must use native Zoom/Teams. Core Audio taps (macOS 14.2+) / WASAPI loopback; bundles llama.cpp. Highest engineering cost; unsigned app warnings without a paid developer certificate. Defer.

### 6(b) Post-hoc conversation / message interpreter

**Option B1 (recommended): static web app (PWA), in-browser LLM**
- Paste a message/thread or upload a short recording. Text: transformers.js/WebLLM with Qwen3-4B or Gemma 4 E4B (q4, about 3-5 GB, cached once), structured JSON output (transformers.js v4.3 constrained decoding) for "literal meaning / possible implied meaning / tone cues / suggested replies / confidence". Latency **3-15 s** per message on an M3 [U]; acceptable for reflection.
- Gemini Nano Prompt API as a zero-download fast path on eligible Chrome installs.
- Recording mode: Whisper (small/turbo) offline transcription plus pyannote segmentation plus per-utterance prosody plus SER, then the LLM explanation. A 5-minute clip in about 1-3 minutes [U].

**Option B2: Local Ollama app** (gemma4:e4b or qwen3-vl:4b for screenshots of chats) with a simple web UI. Best for your own experiments and for evaluating prompts against participant-co-designed examples before porting to B1.

**Option B3: cloud LLM** - rejected for participant data (Section 5).

### What to run on the M3 today (fits ~20 GB disk, 16 GB RAM)
1. `ollama pull gemma4:e2b` (about 4.6-7.5 GB depending on quant) for text+image; test audio via llama.cpp `llama-mtmd-cli`/`llama-server` with the same GGUF family (Ollama audio support for Gemma 4 is unconfirmed).
2. `qwen3-vl:2b` (1.9 GB) as a fast vision baseline. Skip E4B/12B until disk is freed.
3. Browser: transformers.js v4.3.1 demos (Moonshine/Whisper WebGPU, Gemma 4 E2B), MediaPipe Face Landmarker demo, vad-web. These cache in the browser profile and count against disk.
4. Python: VAP-Realtime (CPU) on recorded two-channel test calls; emotion2vec+ base via FunASR.
5. Do **not** enable Gemini Nano until at least 22 GB is free, and expect it to take that space.
6. First benchmarks to run (all [U] above): Moonshine-JS latency in Chrome; Gemma 4 E2B TTFT with a 10 s audio clip plus a frame; MediaPipe FPS while the LLM is generating (GPU contention is the likely bottleneck).

### Glasses path (Emteq)
Keep the pipeline modular: capture -> features (VAD, prosody, blendshapes/expressions, gaze) -> event bus -> interpretation LLM. Emteq's on-face sensors would replace webcam-derived expression features, and a phone would host the models (Chrome Android WebGPU or native MediaPipe/LiteRT; Gemma 4 E2B is designed for on-device). Chong et al. 2020 shows point-of-view eye-contact detection at expert level is feasible from a head-worn camera.

---

## 7. Open questions / items to verify
- Exact latest versions of `@mediapipe/tasks-vision`, `@ricky0123/vad-web`, `@mlc-ai/web-llm`, MLX-VLM (0.6.x reported [S]).
- Gemma 4 audio through Ollama (model card says yes, tag table says text/image).
- emotion2vec+ weight licence; Moonshine non-English licence; Qwen3-VL licence text; MiniCPM-o 4.5 licence and Q4 file size.
- Browser-measured latency for every ASR option on the M3.
- Groq privacy policy wording (no training, retention).
- Whether Chrome's web Prompt API multimodal is fully shipped or still behind flags in the participant's Chrome channel (MDN says first released in 148).
