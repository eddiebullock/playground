import type { TurnCue } from "../analysis/turntaking.ts";
import type { Meter } from "../session/audio.ts";
import type { LiveSource } from "../session/useLive.ts";

type Props = {
  source: LiveSource;
  turn: TurnCue;
  them: Meter;
  you: Meter;
  callAudio: boolean;
  hasMic: boolean;
};

function bar(rms: number): number {
  return Math.min(100, Math.round(rms * 400));
}

export function TurnTab({ source, turn, them, you, callAudio, hasMic }: Props) {
  const watching = source === "file";
  return (
    <section className="panel" aria-label="Turn taking">
      <h2>When to speak</h2>
      {!callAudio && watching && (
        <p className="warn">This file has no usable audio, so turns cannot be heard.</p>
      )}
      {!callAudio && !watching && (
        <p className="warn">
          No audio from the call. Open Zoom or Teams in Chrome, share that tab,
          and tick share audio. A desktop window is usually silent.
        </p>
      )}
      {!hasMic && watching && (
        <p className="detail">Share a microphone only if you want to practise talking over this clip.</p>
      )}
      {!hasMic && !watching && (
        <p className="detail">Share your microphone so your side is included.</p>
      )}
      <p className="status">{turn.headline}</p>
      <p className="detail">{turn.detail}</p>
      <div className="meters" aria-hidden="true">
        <label>
          {watching ? "Video" : "Them"}
          <span className="meter">
            <span style={{ width: `${bar(them.rms)}%` }} />
          </span>
        </label>
        <label>
          You
          <span className="meter">
            <span style={{ width: `${bar(you.rms)}%` }} />
          </span>
        </label>
      </div>
      <p className="fine">A long-turn note appears after about 30 seconds of you talking.</p>
    </section>
  );
}
