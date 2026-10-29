import { useEffect, useState } from "react";
import { Sources } from "./hud/Sources.tsx";
import { Stage, type Marker } from "./hud/Stage.tsx";
import { useLive, type TabId } from "./session/useLive.ts";
import { EmotionTab } from "./tabs/EmotionTab.tsx";
import { EyeTab } from "./tabs/EyeTab.tsx";
import { TurnTab } from "./tabs/TurnTab.tsx";

const MARK_KEY = "cue-camera-marker";
const OLD_MARK_KEY = "gloss-live-camera-marker";

function loadMarker(): Marker {
  try {
    const raw = localStorage.getItem(MARK_KEY) ?? localStorage.getItem(OLD_MARK_KEY);
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

  const hasPicture = Boolean(live.callStream || live.fileUrl);
  const hasFace = live.snap.callFaces.length > 0;

  return (
    <div className="shell">
      <header className="top">
        <div>
          <h1>Cue</h1>
          <p className="tagline">Realtime cues on a video, Zoom, or Teams, on this computer only.</p>
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

      <Sources
        source={live.source}
        fileName={live.fileName}
        hasCall={Boolean(live.callStream)}
        hasMic={Boolean(live.micStream)}
        hasCam={Boolean(live.camStream)}
        error={live.error}
        onFile={live.startFile}
        onDemo={() => void live.startDemo()}
        onShare={() => void live.startCall()}
        onMeeting={(kind, raw) => live.startMeeting(kind, raw)}
        onShareMeeting={() => void live.shareMeeting()}
        onMic={() => void live.startMic()}
        onCam={() => void live.startCam()}
        onStop={live.stopAll}
      />
      {live.snap.visionError && (
        <p className="warn">Face model: {live.snap.visionError}. Check the network for the first load.</p>
      )}

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

      <div className="layout">
        <Stage
          source={live.source}
          callStream={live.callStream}
          camStream={live.camStream}
          fileUrl={live.fileUrl}
          fileName={live.fileName}
          callVideo={live.callVideo}
          camVideo={live.camVideo}
          faces={live.snap.callFaces}
          marker={marker}
          onMarker={setMarker}
          onFilePlaying={live.tapFileAudio}
          showMarker={tab === "eye"}
          showFaces={tab !== "turn"}
        />
        {tab === "eye" && (
          <EyeTab
            source={live.source}
            gaze={live.snap.gaze}
            hasCam={Boolean(live.camStream)}
            hasCall={hasPicture}
            hasFace={hasFace}
          />
        )}
        {tab === "turn" && (
          <TurnTab
            source={live.source}
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
