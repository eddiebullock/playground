import type { GazeRead } from "../analysis/gaze.ts";
import type { LiveSource } from "../session/useLive.ts";

type Props = {
  source: LiveSource;
  gaze: GazeRead;
  hasCam: boolean;
  hasCall: boolean;
  hasFace: boolean;
};

export function EyeTab({ source, gaze, hasCam, hasCall, hasFace }: Props) {
  const watching = source === "file";
  return (
    <section className="panel" aria-label="Eye contact">
      <h2>Where to look</h2>
      {watching ? (
        <>
          <p className="lede">
            This is a recording. The boxes follow faces in the video as it
            plays. The webcam ring is for live calls, not for watching a file.
          </p>
          <p className="status">
            {hasFace ? "A face is in the picture." : hasCall ? "No face in this frame yet." : "Open a video with a visible face."}
          </p>
          <p className="detail">
            If you are practising being on a call, share your camera and look
            at the lens, not at the recording.
          </p>
        </>
      ) : (
        <>
          <p className="lede">
            On a video call they see you looking at them when you look at the
            webcam, not when you look at their face on the screen.
          </p>
          <p className="status">
            {hasCall && hasFace
              ? "Look at the ring, not the tile."
              : "Share Zoom, Teams, or another window, then drag the ring onto your webcam."}
          </p>
          <p className="detail">
            Drag anywhere on the picture to put the camera marker where the
            physical lens is (usually the top centre of a laptop).
          </p>
        </>
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
