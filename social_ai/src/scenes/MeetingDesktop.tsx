import type { ReactNode } from "react";
import { othersAreTalking, stageFocus, type SceneFrame } from "../lib/sceneState";
import type { Scene } from "../types";
import { LookHere, TalkingBar } from "../components/CallCues";
import { CameraIcon, HarborMark, LeaveIcon, MicIcon, PauseIcon, ShareIcon } from "../components/Icons";
import { GlossPanel } from "../components/GlossPanel";
import { PersonTile } from "../components/PersonTile";
import { SubtitleBar } from "../components/SubtitleBar";

const GRID = { x: 264, y: 96, w: 1156, h: 804, gap: 16 };

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

export function MeetingDesktop({ scene, frame, quiet, reduced, frozen, onExplain, onToggleQuiet, eyeContact }: Props) {
  const localT = frame.cue ? frame.t - frame.cue.start : 0;
  const { main, pip } = stageFocus(scene, frame);
  const others = othersAreTalking(scene, frame);

  return (
    <div style={{ position: "absolute", inset: 0, background: "#1A2332" }}>
      <header
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: 72,
          display: "flex",
          alignItems: "center",
          gap: 18,
          padding: "0 24px",
          borderBottom: "1px solid #314158",
          background: "#1A2332",
        }}
      >
        <HarborMark />
        <div style={{ fontSize: 20, fontWeight: 600, letterSpacing: "0.01em" }}>Harbor</div>
        <div style={{ width: 1, height: 22, background: "#314158" }} />
        <div style={{ fontSize: 16, color: "#F7F5F0" }}>{scene.meetingTitle}</div>
        {quiet ? (
          <div
            style={{
              marginLeft: 18,
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              background: "#F7F5F0",
              color: "#14213D",
              borderRadius: 999,
              padding: "6px 12px",
              fontSize: 14,
              fontWeight: 600,
            }}
          >
            <PauseIcon size={16} color="#14213D" />
            Quiet mode — suggestions are paused
          </div>
        ) : null}
        <div style={{ marginLeft: "auto", marginRight: 340, fontSize: 15, color: "#D6DEE8" }}>
          {scene.clock} · Speaker view
        </div>
      </header>

      <aside
        style={{
          position: "absolute",
          top: 72,
          left: 0,
          width: 240,
          bottom: 0,
          background: "#151D29",
          borderRight: "1px solid #314158",
          padding: "22px 18px",
        }}
      >
        <div style={{ fontSize: 13, fontWeight: 600, letterSpacing: "0.08em", color: "#D6DEE8" }}>AGENDA</div>
        <ol style={{ listStyle: "none", margin: "16px 0 0", padding: 0, display: "flex", flexDirection: "column", gap: 8 }}>
          {scene.agenda.map((item, index) => {
            const active = item.id === frame.agenda.id;
            return (
              <li
                key={item.id}
                style={{
                  borderRadius: 12,
                  padding: "10px 12px",
                  background: active ? "rgba(247,245,240,0.06)" : "transparent",
                  borderLeft: active ? "3px solid #8AA899" : "3px solid transparent",
                }}
              >
                <div style={{ fontSize: 12, color: "#D6DEE8" }}>{index + 1}</div>
                <div style={{ fontSize: 15, fontWeight: 600, color: "#F7F5F0", marginTop: 2 }}>{item.label}</div>
                {active ? <div style={{ fontSize: 13, marginTop: 4, color: "#F7F5F0" }}>Now</div> : null}
              </li>
            );
          })}
        </ol>
      </aside>

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
        <TalkingBar on={others} reduced={reduced} />
        {eyeContact && main.id !== "alex" ? <LookHere person={main} /> : null}
        {pip ? (
          <div style={{ position: "absolute", right: 18, bottom: 18, zIndex: 2 }}>
            <PersonTile
              person={pip}
              speaking={frame.speaker?.id === pip.id}
              cue={frame.cue?.aboutId === pip.id ? frame.cue : null}
              width={280}
              height={188}
              reduced={reduced}
            />
          </div>
        ) : null}
      </div>

      <div style={{ position: "absolute", left: 264, top: 916, width: 1156 }}>
        <SubtitleBar line={frame.line} speaker={frame.speaker} width={1156} />
      </div>

      <div
        style={{
          position: "absolute",
          left: 264,
          top: 996,
          width: 1156,
          display: "flex",
          justifyContent: "center",
          gap: 10,
        }}
      >
        <Control icon={<MicIcon color="#F7F5F0" />} label="Mic" />
        <Control icon={<CameraIcon color="#F7F5F0" />} label="Camera" />
        <Control icon={<ShareIcon color="#F7F5F0" />} label="Share" />
        <Control icon={<LeaveIcon color="#F7F5F0" />} label="Leave" tone="leave" />
      </div>

      <aside
        style={{
          position: "absolute",
          left: 1452,
          top: 96,
          width: 440,
          height: 280,
          background: "#1E2938",
          borderRadius: 18,
          border: "1px solid #314158",
          padding: "18px 18px 12px",
        }}
      >
        <div style={{ fontSize: 13, fontWeight: 600, letterSpacing: "0.08em", color: "#D6DEE8" }}>IN THE ROOM</div>
        <ul style={{ listStyle: "none", margin: "12px 0 0", padding: 0 }}>
          {scene.people.map((person) => {
            const speaking = frame.speaker?.id === person.id;
            return (
              <li
                key={person.id}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "8px 0",
                  borderTop: "1px solid rgba(49,65,88,0.7)",
                }}
              >
                <span style={{ fontSize: 16, color: "#F7F5F0" }}>
                  {person.name}
                  <span style={{ color: "#D6DEE8" }}> · {person.role}</span>
                </span>
                {speaking ? <span style={{ fontSize: 13, fontWeight: 600 }}>Speaking</span> : null}
              </li>
            );
          })}
        </ul>
      </aside>

      <div style={{ position: "absolute", left: 1452, top: 400, width: 440, height: 640 }}>
        <GlossPanel
          cue={frame.cue}
          audioOn={frame.audioOn}
          quiet={quiet}
          reduced={reduced}
          frozen={frozen}
          localT={localT}
          onExplain={onExplain}
          onToggleQuiet={onToggleQuiet}
        />
      </div>
    </div>
  );
}

function Control({ icon, label, tone }: { icon: ReactNode; label: string; tone?: "leave" }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4, width: 72 }}>
      <div
        style={{
          width: 44,
          height: 36,
          borderRadius: 12,
          display: "grid",
          placeItems: "center",
          background: tone === "leave" ? "#8F4545" : "#243044",
        }}
      >
        {icon}
      </div>
      <div style={{ fontSize: 12, color: "#F7F5F0" }}>{label}</div>
    </div>
  );
}
