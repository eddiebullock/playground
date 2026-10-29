export type Clip = {
  id: string;
  startedAt: string;
  durationMs: number;
  url: string;
};

export type SpeechRecorder = {
  stop: () => Promise<Blob>;
};

export function createSpeechRecorder(stream: MediaStream): SpeechRecorder {
  const tracks = stream.getAudioTracks().filter((t) => t.readyState === "live");
  if (tracks.length === 0) {
    throw new Error("No live microphone track to record.");
  }
  const audio = new MediaStream(tracks);
  const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].find((t) =>
    MediaRecorder.isTypeSupported(t),
  );
  const rec = mime ? new MediaRecorder(audio, { mimeType: mime }) : new MediaRecorder(audio);
  const chunks: Blob[] = [];
  rec.addEventListener("dataavailable", (e) => {
    if (e.data.size) chunks.push(e.data);
  });
  rec.start(250);
  return {
    stop() {
      return new Promise((resolve) => {
        rec.addEventListener(
          "stop",
          () => resolve(new Blob(chunks, { type: rec.mimeType || "audio/webm" })),
          { once: true },
        );
        if (rec.state === "inactive") {
          resolve(new Blob(chunks, { type: rec.mimeType || "audio/webm" }));
          return;
        }
        rec.stop();
      });
    },
  };
}

export function revokeClips(clips: Clip[]): void {
  for (const c of clips) URL.revokeObjectURL(c.url);
}
