import { useState } from "react";
import type { MeetKind } from "../session/meet.ts";
import type { LiveSource } from "../session/useLive.ts";

type Props = {
  source: LiveSource;
  fileName: string;
  hasCall: boolean;
  hasMic: boolean;
  hasCam: boolean;
  error: string;
  onFile: (file: File) => void;
  onDemo: () => void;
  onShare: () => void;
  onMeeting: (kind: MeetKind, raw: string) => void;
  onShareMeeting: () => void;
  onMic: () => void;
  onCam: () => void;
  onStop: () => void;
};

export function Sources({
  source,
  fileName,
  hasCall,
  hasMic,
  hasCam,
  error,
  onFile,
  onDemo,
  onShare,
  onMeeting,
  onShareMeeting,
  onMic,
  onCam,
  onStop,
}: Props) {
  const [zoom, setZoom] = useState("");
  const [teams, setTeams] = useState("");

  return (
    <div className="sources">
      <section className="source-card">
        <h3>Watch a video</h3>
        <p>The three tabs follow the file as it plays, on this computer only.</p>
        <div className="controls">
          <label className="file-btn">
            Open a video file
            <input
              type="file"
              accept="video/*,.mp4,.webm,.mov,.m4v"
              onChange={(e) => {
                const file = e.target.files?.[0];
                e.target.value = "";
                if (file) onFile(file);
              }}
            />
          </label>
          <button type="button" onClick={() => void onDemo()}>
            Try a short sample
          </button>
        </div>
        {source === "file" && fileName && <p className="muted">Playing {fileName}</p>}
      </section>

      <section className="source-card">
        <h3>Zoom</h3>
        <p>
          Paste an invite or meeting ID. Cue cannot sit inside the Zoom app; it
          watches the Zoom tab you share.
        </p>
        <input
          type="text"
          value={zoom}
          onChange={(e) => setZoom(e.target.value)}
          placeholder="zoom.us/j/… or meeting ID"
          aria-label="Zoom invite"
        />
        <div className="controls">
          <button type="button" onClick={() => onMeeting("zoom", zoom)}>
            Open Zoom
          </button>
          <button type="button" onClick={() => void onShareMeeting()}>
            Share the Zoom tab
          </button>
        </div>
      </section>

      <section className="source-card">
        <h3>Teams</h3>
        <p>
          Paste the meeting link from the calendar or chat, then share that
          Chrome tab with audio.
        </p>
        <input
          type="text"
          value={teams}
          onChange={(e) => setTeams(e.target.value)}
          placeholder="teams.microsoft.com/l/meetup-join/…"
          aria-label="Teams invite"
        />
        <div className="controls">
          <button type="button" onClick={() => onMeeting("teams", teams)}>
            Open Teams
          </button>
          <button type="button" onClick={() => void onShareMeeting()}>
            Share the Teams tab
          </button>
        </div>
      </section>

      <section className="source-card">
        <h3>Any window</h3>
        <p>Meet, a movie, or the desktop Zoom/Teams window. Window share is often silent.</p>
        <div className="controls">
          <button type="button" onClick={() => void onShare()}>
            {hasCall && source === "display" ? "Share a different window" : "Share a window"}
          </button>
          <button type="button" onClick={() => void onMic()}>
            {hasMic ? "Microphone on" : "Share microphone"}
          </button>
          <button type="button" onClick={() => void onCam()}>
            {hasCam ? "Camera on" : "Share my camera"}
          </button>
          <button type="button" onClick={onStop}>
            Stop
          </button>
        </div>
      </section>

      {error && <p className="warn">{error}</p>}
    </div>
  );
}
