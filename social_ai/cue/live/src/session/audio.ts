export type Meter = { rms: number; speaking: boolean; hasAudio: boolean };

const silent: Meter = { rms: 0, speaking: false, hasAudio: false };

function meterFromAnalyser(analyser: AnalyserNode, stop: () => void, threshold: number): {
  tick: () => Meter;
  stop: () => void;
} {
  const data = new Float32Array(analyser.fftSize);
  let hang = 0;
  return {
    tick() {
      analyser.getFloatTimeDomainData(data);
      let sum = 0;
      for (const x of data) sum += x * x;
      const rms = Math.sqrt(sum / data.length);
      if (rms >= threshold) hang = 5;
      else if (hang > 0) hang -= 1;
      return { rms, speaking: hang > 0, hasAudio: true };
    },
    stop,
  };
}

export function attachMeter(stream: MediaStream | null, threshold = 0.045): {
  tick: () => Meter;
  stop: () => void;
} {
  if (!stream || stream.getAudioTracks().length === 0) {
    return { tick: () => silent, stop: () => undefined };
  }
  const ctx = new AudioContext();
  void ctx.resume();
  const src = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  src.connect(analyser);
  return meterFromAnalyser(analyser, () => void ctx.close(), threshold);
}

/** Fallback when captureStream() has no audio. Hearing still goes through this graph. */
export function attachElementMeter(el: HTMLMediaElement | null, threshold = 0.045): {
  tick: () => Meter;
  stop: () => void;
} {
  if (!el) return { tick: () => silent, stop: () => undefined };
  const ctx = new AudioContext();
  void ctx.resume();
  try {
    const src = ctx.createMediaElementSource(el);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    src.connect(analyser);
    src.connect(ctx.destination);
    return meterFromAnalyser(analyser, () => void ctx.close(), threshold);
  } catch {
    void ctx.close();
    return { tick: () => silent, stop: () => undefined };
  }
}
