import { othersAreTalking, stageFocus, type SceneFrame } from "../lib/sceneState";
import type { Scene } from "../types";
import { LookHere, RapportFrame, TalkingBar, YourTurn } from "../components/CallCues";

const LENS = { x: 320, y: 96, w: 1280, h: 860 };

type Props = {
  scene: Scene;
  frame: SceneFrame;
  quiet: boolean;
  reduced: boolean;
  eyeContact: boolean;
};

/** Mixed while the sarcasm cue is up. The rest of the meeting reads as easy. */
function meetingRapport(frame: SceneFrame, quiet: boolean) {
  if (quiet) return "unsure" as const;
  if (frame.t >= 5.2 && frame.t < 10.8) return "mixed" as const;
  return "good" as const;
}

export function SimpleGlassesMeeting({ scene, frame, quiet, reduced, eyeContact }: Props) {
  const { main } = stageFocus(scene, frame);
  const others = othersAreTalking(scene, frame);
  const yourTurn = !others && frame.t >= 17.6 && frame.t < 25.2;
  const poster = `clips/${main.avatar}.jpg`;

  return (
    <div
      style={{ position: "absolute", inset: 0, background: "#0e1218" }}
      data-talking={others ? "true" : "false"}
      data-turn={yourTurn ? "true" : "false"}
    >
      <div
        style={{
          position: "absolute",
          left: LENS.x,
          top: LENS.y,
          width: LENS.w,
          height: LENS.h,
          borderRadius: 56,
          overflow: "hidden",
          background: "#121820",
        }}
      >
        {reduced ? (
          <img src={poster} alt="" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }} />
        ) : (
          <video
            key={main.avatar}
            src={`clips/${main.avatar}.mp4`}
            poster={poster}
            autoPlay
            muted
            loop
            playsInline
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }}
          />
        )}
        <TalkingBar on={others} reduced={reduced} rounded={false} />
        {eyeContact && main.id !== "alex" && !quiet ? <LookHere person={main} /> : null}
        <YourTurn on={yourTurn} scale={1.4} top={34} />
        <RapportFrame level={meetingRapport(frame, quiet)} reduced={reduced} radius={56} scale={1.4} />
      </div>
    </div>
  );
}
