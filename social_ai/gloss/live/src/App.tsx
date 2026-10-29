import { useEffect, useState } from "react";
import { Stage, type Marker } from "./hud/Stage.tsx";
import { useLive, type TabId } from "./session/useLive.ts";
import { EmotionTab } from "./tabs/EmotionTab.tsx";
import { EyeTab } from "./tabs/EyeTab.tsx";
import { TurnTab } from "./tabs/TurnTab.tsx";

const MARK_KEY = "gloss-live-camera-marker";

function loadMarker(): Marker {
  try {
    const raw = localStorage.getItem(MARK_KEY);
    if (raw) return JSON.parse(raw) as Marker;
  } catch {
    /* ignore */
  }
  return { x: 0.5, y: 0.08 };
}

export default function App() {
  const live = useLive();
  const [tab, setTab] = useState<TabId>("eye");
  const [marker, setMarker] = useState<Marker>(loadMarker);
  const [workplace, setWorkplace] = useState(false);

  useEffect(() => {
    localStorage.setItem(MARK_KEY, JSON.stringify(marker));
  }, [marker]);

  const hasCall = Boolean(live.callStream);
  const hasFace = live.snap.callFaces.length > 0;

  return (
    <div className="shell">
      <header className="top">
        <div>
          <h1>Gloss Live</h1>
          <p className="tagline">Realtime cues on a call or a video, on this computer only.</p>
        </div>
        <label className="work">
          <input
            type="checkbox"
            checked={workplace}
            onChange={(e) => setWorkplace(e.target.checked)}
          />
          Work or education setting
        </label>
      </header>

      <nav className="mode" aria-label="Tabs">
        <button type="button" aria-selected={tab === "eye"} onClick={() => setTab("eye")}>
          Eye contact
        </button>
        <button type="button" aria-selected={tab === "turn"} onClick={() => setTab("turn")}>
          Turn taking
        </button>
        <button type="button" aria-selected={tab === "emotion"} onClick={() => setTab("emotion")}>
          Emotion
        </button>
      </nav>

      <div className="controls">
        <button type="button" onClick={() => void live.startCall()}>
          {hasCall ? "Share a different window" : "Share call or video"}
        </button>
        <button type="button" onClick={() => void live.startMic()}>
          {live.micStream ? "Microphone on" : "Share microphone"}
        </button>
        <button type="button" onClick={() => void live.startCam()}>
          {live.camStream ? "Camera on" : "Share my camera"}
        </button>
        <button type="button" onClick={live.stopAll}>
          Stop
        </button>
      </div>
      {live.error && <p className="warn">{live.error}</p>}
      {live.snap.visionError && (
        <p className="warn">Face model: {live.snap.visionError}. Check the network for the first load.</p>
      )}

      <div className="layout">
        <Stage
          callStream={live.callStream}
          camStream={live.camStream}
          callVideo={live.callVideo}
          camVideo={live.camVideo}
          faces={live.snap.callFaces}
          marker={marker}
          onMarker={setMarker}
          showMarker={tab === "eye"}
          showFaces={tab !== "turn"}
        />
        {tab === "eye" && (
          <EyeTab
            gaze={live.snap.gaze}
            hasCam={Boolean(live.camStream)}
            hasCall={hasCall}
            hasFace={hasFace}
          />
        )}
        {tab === "turn" && (
          <TurnTab
            turn={live.snap.turn}
            them={live.snap.them}
            you={live.snap.you}
            callAudio={live.snap.callAudio}
            hasMic={Boolean(live.micStream)}
          />
        )}
        {tab === "emotion" && (
          <EmotionTab emotion={live.snap.emotion} hasFace={hasFace} workplace={workplace} />
        )}
      </div>

      <footer>
        <p>
          Nothing is uploaded. Faces and audio are processed in this browser.
          Not a medical or support service. A face is not a feeling.
        </p>
        {!live.snap.visionReady && !live.snap.visionError && (
          <p className="muted">Loading the on-device face model…</p>
        )}
      </footer>
    </div>
  );
}
