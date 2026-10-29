import type { GazeRead } from "../analysis/gaze.ts";

type Props = {
  gaze: GazeRead;
  hasCam: boolean;
  hasCall: boolean;
  hasFace: boolean;
};

export function EyeTab({ gaze, hasCam, hasCall, hasFace }: Props) {
  return (
    <section className="panel" aria-label="Eye contact">
      <h2>Where to look</h2>
      <p className="lede">
        On a video call they see you looking at them when you look at the
        webcam, not when you look at their face on the screen.
      </p>
      <p className="status">{hasCall && hasFace ? "Look at the ring, not the tile." : "Share the call, then drag the ring onto your webcam."}</p>
      <p className="detail">
        Drag anywhere on the picture to put the camera marker where the
        physical lens is (usually the top centre of a laptop).
      </p>
      {hasCam ? (
        <>
          <p className="status">{gaze.headline}</p>
          <p className="detail">{gaze.detail}</p>
        </>
      ) : (
        <p className="detail">Share your camera if you want a guess about whether you are looking at the lens. You can leave it off.</p>
      )}
      <p className="fine">This is not a score. Looking away is allowed.</p>
    </section>
  );
}
