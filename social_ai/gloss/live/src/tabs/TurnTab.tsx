import type { TurnCue } from "../analysis/turntaking.ts";
import type { Meter } from "../session/audio.ts";

type Props = {
  turn: TurnCue;
  them: Meter;
  you: Meter;
  callAudio: boolean;
  hasMic: boolean;
};

function bar(rms: number): number {
  return Math.min(100, Math.round(rms * 400));
}

export function TurnTab({ turn, them, you, callAudio, hasMic }: Props) {
  return (
    <section className="panel" aria-label="Turn taking">
      <h2>When to speak</h2>
      {!callAudio && (
        <p className="warn">
          No audio from the call. In Chrome, share the Meet/Zoom <em>tab</em> and
          tick share audio. A desktop window usually has no sound.
        </p>
      )}
      {!hasMic && <p className="detail">Share your microphone so Gloss can hear your side.</p>}
      <p className="status">{turn.headline}</p>
      <p className="detail">{turn.detail}</p>
      <div className="meters" aria-hidden="true">
        <label>
          Them
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
