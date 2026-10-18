import { othersAreTalking, stageFocus, type SceneFrame } from "../lib/sceneState";
import type { Scene } from "../types";
import { LookHere, TalkingBar } from "../components/CallCues";
import { GlossPanel } from "../components/GlossPanel";
import { PauseIcon } from "../components/Icons";
import { PersonTile } from "../components/PersonTile";
import { SubtitleBar } from "../components/SubtitleBar";

const GRID = { x: 72, y: 78, w: 1280, h: 860, gap: 18 };

type Props = {
  scene: Scene;
  frame: SceneFrame;
  quiet: boolean;
  reduced: boolean;
  frozen: boolean;
  onExplain: () => void;
  onToggleQuiet: () => void;
  eyeContact: boolean;
};

export function MeetingGlasses({ scene, frame, quiet, reduced, frozen, onExplain, onToggleQuiet, eyeContact }: Props) {
  const localT = frame.cue ? frame.t - frame.cue.start : 0;
  const { main, pip } = stageFocus(scene, frame);
  const others = othersAreTalking(scene, frame);

  return (
    <div style={{ position: "absolute", inset: 0, background: "#121820" }}>
      <div style={{ position: "absolute", left: GRID.x, top: GRID.y, width: GRID.w, height: GRID.h }}>
        <PersonTile
          person={main}
          speaking={frame.speaker?.id === main.id}
          cue={frame.cue?.aboutId === main.id ? frame.cue : null}
          width={GRID.w}
          height={GRID.h}
          reduced={reduced}
          chipOffset={others ? 72 : 14}
        />
        {pip ? (
          <div style={{ position: "absolute", left: 18, bottom: 18, zIndex: 2 }}>
            <PersonTile
              person={pip}
              speaking={frame.speaker?.id === pip.id}
              cue={null}
              width={240}
              height={160}
              reduced={reduced}
            />
          </div>
        ) : null}
      </div>

      <div
        aria-hidden="true"
        style={{
          position: "absolute",
          inset: 0,
          pointerEvents: "none",
          background:
            "radial-gradient(ellipse at 46% 48%, rgba(18,24,32,0) 38%, rgba(8,10,14,0.28) 62%, rgba(6,8,12,0.82) 100%)",
          boxShadow: "inset 0 0 90px rgba(6,8,12,0.45)",
        }}
      />
      <div
        aria-hidden="true"
        style={{
          position: "absolute",
          inset: 14,
          borderRadius: 48,
          border: "2px solid rgba(247,245,240,0.28)",
          pointerEvents: "none",
        }}
      />

      <div
        style={{
          position: "absolute",
          top: 28,
          left: "50%",
          transform: "translateX(-50%)",
          zIndex: 5,
          background: "rgba(20,33,61,0.88)",
          color: "#F7F5F0",
          borderRadius: 999,
          padding: "7px 12px",
          fontSize: 14,
          fontWeight: 600,
        }}
      >
        Display glasses
      </div>

      <div style={{ position: "absolute", left: GRID.x, top: GRID.y, width: GRID.w, height: GRID.h, zIndex: 5, pointerEvents: "none" }}>
        <TalkingBar on={others} reduced={reduced} />
        {eyeContact && main.id !== "alex" ? <LookHere person={main} /> : null}
      </div>

      {quiet ? (
        <div
          style={{
            position: "absolute",
            top: 28,
            left: 36,
            zIndex: 5,
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            background: "#F7F5F0",
            color: "#14213D",
            borderRadius: 999,
            padding: "7px 12px",
            fontSize: 14,
            fontWeight: 600,
          }}
        >
          <PauseIcon size={16} color="#14213D" />
          Quiet mode — suggestions are paused
        </div>
      ) : null}

      <div style={{ position: "absolute", left: 72, top: 968, width: 1100, zIndex: 5 }}>
        <SubtitleBar line={frame.line} speaker={frame.speaker} width={1100} />
      </div>

      <div style={{ position: "absolute", right: 40, bottom: 48, width: 430, height: 390, zIndex: 6 }}>
        <GlossPanel
          cue={frame.cue}
          audioOn={frame.audioOn}
          quiet={quiet}
          reduced={reduced}
          frozen={frozen}
          localT={localT}
          onExplain={onExplain}
          onToggleQuiet={onToggleQuiet}
          compact
        />
      </div>
    </div>
  );
}
