import { LONG_YOU_MS, type TurnCue } from "../analysis/turntaking.ts";
import type { Meter } from "../session/audio.ts";
import type { Clip } from "../session/record.ts";
import type { LiveSource } from "../session/useLive.ts";

type Props = {
  source: LiveSource;
  turn: TurnCue;
  them: Meter;
  you: Meter;
  callAudio: boolean;
  hasMic: boolean;
  recording: boolean;
  clips: Clip[];
  onClearClips: () => void;
};

function bar(rms: number): number {
  return Math.min(100, Math.round(rms * 400));
}

function clock(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return m > 0 ? `${m}:${String(r).padStart(2, "0")}` : `${s}s`;
}

export function TurnTab({
  source,
  turn,
  them,
  you,
  callAudio,
  hasMic,
  recording,
  clips,
  onClearClips,
}: Props) {
  const watching = source === "file";
  const fill = Math.min(100, Math.round((turn.youMs / LONG_YOU_MS) * 100));
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
      {!hasMic && (
        <p className="detail">Share your microphone to record your turns and time them.</p>
      )}
      {recording && <p className="rec-pill">Recording</p>}
      <p className={`status${turn.longYou ? " too-long" : ""}`}>{turn.headline}</p>
      <p className="detail">{turn.detail}</p>
      {(turn.talk === "you" || turn.talk === "both") && (
        <div className="long-meter" aria-label="Your turn length">
          <span className={`long-fill${turn.longYou ? " long-fill-hot" : ""}`} style={{ width: `${fill}%` }} />
        </div>
      )}
      {turn.longYou && (
        <p className="too-long-flag">Too long — {clock(turn.youMs)}. Let them in.</p>
      )}
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
      <h3 className="clips-head">Your turns</h3>
      {clips.length === 0 ? (
        <p className="fine">When you start speaking, that turn is saved here on this computer.</p>
      ) : (
        <>
          <ul className="clips">
            {clips.map((c, i) => (
              <li key={c.id}>
                <span>Turn {i + 1} · {clock(c.durationMs)}</span>
                <audio src={c.url} controls preload="metadata" />
              </li>
            ))}
          </ul>
          <button type="button" onClick={onClearClips}>
            Clear recordings
          </button>
        </>
      )}
    </section>
  );
}
