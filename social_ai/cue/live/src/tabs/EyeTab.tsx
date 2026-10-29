import type { GazeRead } from "../analysis/gaze.ts";
import type { LiveSource } from "../session/useLive.ts";

type Props = {
  source: LiveSource;
  gaze: GazeRead;
  hasCam: boolean;
  hasCall: boolean;
  hasFace: boolean;
  blur: number;
  onBlur: (n: number) => void;
};

export function EyeTab({ source, gaze, hasCam, hasCall, hasFace, blur, onBlur }: Props) {
  const watching = source === "file";
  return (
    <section className="panel" aria-label="Eye contact">
      <h2>Where to look</h2>
      <p className="lede">
        The face stays sharp. Everything else is blurred so the look target is
        obvious.
      </p>
      <p className="status">
        {hasFace ? "Look at the clear face." : hasCall ? "No face in this frame yet." : "Open a video or share a call."}
      </p>
      <label className="blur-set">
        Background blur
        <input
          type="range"
          min={0}
          max={24}
          value={blur}
          onChange={(e) => onBlur(Number(e.target.value))}
          aria-valuemin={0}
          aria-valuemax={24}
          aria-valuenow={blur}
        />
        <span>{blur}</span>
      </label>
      {watching ? (
        <p className="detail">This is a recording. Look at the unblurred face in the picture.</p>
      ) : (
        <p className="detail">
          On a live call they still see you looking if you look at the webcam
          ring, not only at their tile.
        </p>
      )}
      {hasCam ? (
        <>
          <p className="status">{gaze.headline}</p>
          <p className="detail">{gaze.detail}</p>
        </>
      ) : (
        <p className="detail">
          Share your camera if you want a guess about whether you are looking
          at the lens. You can leave it off.
        </p>
      )}
      <p className="fine">This is not a score. Looking away is allowed.</p>
    </section>
  );
}
