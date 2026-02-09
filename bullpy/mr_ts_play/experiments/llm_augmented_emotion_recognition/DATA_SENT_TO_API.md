# What the model actually receives (no path/filename leakage)

The model **does not** see video paths, audio paths, or filenames. It only sees **content**: pixels, audio bytes, and a text prompt that lists the candidate labels.

## What is sent to the API

| Input | What is sent | What is NOT sent |
|-------|----------------|------------------|
| **Video** | Decoded **frames** → converted to JPEG → **base64 image data** (pixel content only) | Video path, filename (e.g. `0402101Y5Vneedled.mov`), stimulus_path |
| **Audio** | **Audio file read from disk** → **base64 audio data** (raw bytes; MIME type from extension only, e.g. `audio/mpeg`) | Audio path, filename (e.g. `fix__EV12H.mp3`), folder name |
| **Text prompt** | Fixed instruction text + **list of candidate labels** (e.g. `"needled, bewitched, disinclined, concerned"`) | Any path, any filename, stimulus_path, video_path, audio_path |

So the model cannot “read the file title”: it never receives `.../0402101Y5Vneedled.mov` or `.../Joking/fix__EV12H.mp3`. It receives:

1. **Pixels** from the video (frames as images).
2. **Audio waveform** (raw audio bytes) when audio is used.
3. **Prompt text** that only contains the task description and the set of possible labels (forced choice).

Implementation details (for verification):

- **Gemini:** `llm_wrapper.py` builds `parts = [{"text": prompt}, *image_parts]` then, if `audio_path` exists, appends `{"inline_data": {"mime_type": ..., "data": audio_base64}}`. No path or filename is added to the payload.
- **OpenAI:** Same idea: `content = [{"type": "text", "text": prompt}, *image_urls]` and, when sending audio, `{"type": "input_audio", "input_audio": {"data": audio_base64, "format": "mp3"}}`. Only extension is used for `format`; path/filename are not sent.
- **Anthropic:** Text + image blocks only (no audio); no paths in the message.

`video_path` and `audio_path` are used only for **caching** (cache key) and **logging**; they are not passed to the provider APIs.

## Optional sanity check (to confirm the model uses content)

To confirm the model is using audio/video **content** and not some hidden cue:

1. **Audio:** Run a small subset (e.g. 20 trials) with **wrong** audio (e.g. audio from a different emotion) and compare accuracy to the same trials with correct audio. If accuracy drops when audio is wrong, the model is using the audio signal.
2. **Video:** Similarly, you could swap frames from another trial and see if accuracy drops (more involved).

This is optional; the code inspection above already shows that paths/filenames are not sent.

## Summary

- **Video:** Model sees only **frame pixels** (base64 images). No path or filename.
- **Audio:** Model sees only **audio bytes** (base64). No path or filename.
- **Prompt:** Only **instruction + candidate labels**. No paths or filenames.

So the model is **listening to the audio and watching the video**, not reading file titles.
