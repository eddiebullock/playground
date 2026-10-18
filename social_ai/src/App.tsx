import { MotionConfig, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import type { ViewMode } from "./types";
import { HonestyTag } from "./components/HonestyTag";
import { ExplainerCard } from "./components/ExplainerCard";
import { SCENE_A } from "./content/sceneA";
import { INTRO_CLIP, JOKE_CLIP, fieldCueAt, fieldStatus } from "./content/field";
import { NETFLIX_WATCH, YOUTUBE_WATCH } from "./content/practice";
import { readParams } from "./lib/params";
import { frameAt, othersAreTalking } from "./lib/sceneState";
import { FieldWatch } from "./scenes/FieldWatch";
import { SettingsPanel } from "./scenes/SettingsPanel";
import { MeetingDesktop } from "./scenes/MeetingDesktop";
import { MeetingGlasses } from "./scenes/MeetingGlasses";
import { PracticeWatch } from "./scenes/PracticeWatch";
import { SimpleGlassesMeeting } from "./scenes/SimpleGlassesMeeting";

type Demo =
  | "joke"
  | "intro"
  | "meeting"
  | "youtube"
  | "netflix"
  | "glasses-joke"
  | "glasses-intro"
  | "glasses-meeting"
  | "focus-joke";

const isFieldDemo = (demo: Demo) =>
  demo === "joke" || demo === "intro" || demo === "glasses-joke" || demo === "glasses-intro" || demo === "focus-joke";
const isMeetingDemo = (demo: Demo) => demo === "meeting" || demo === "glasses-meeting";
const isSimpleDemo = (demo: Demo) => demo.startsWith("glasses-") || demo === "focus-joke";
const clipFor = (demo: Demo) => (demo === "intro" || demo === "glasses-intro" ? INTRO_CLIP : JOKE_CLIP);

const params = readParams(window.location.search);

export function App() {
  const osReduced = useReducedMotion();
  const reduced = params.reduced || (!params.motion && Boolean(osReduced));
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [freezeT, setFreezeT] = useState(params.t);
  const [elapsed, setElapsed] = useState(0);
  const [explainOn, setExplainOn] = useState(params.explainer);
  const [quietDev, setQuietDev] = useState<boolean | null>(params.quiet ? true : null);

  useEffect(() => {
    let live = true;
    const load = async () => {
      await Promise.all([
        document.fonts.load('400 32px "Libre Baskerville"'),
        document.fonts.load('700 32px "Libre Baskerville"'),
        document.fonts.load('400 18px "Public Sans"'),
        document.fonts.load('600 16px "Public Sans"'),
        document.fonts.ready,
      ]);
      if (live) setReady(true);
    };
    void load();
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    window.__start = () => setPlaying(true);
    window.__seek = (time: number) => {
      setPlaying(false);
      setFreezeT(time);
    };
    return () => {
      delete window.__start;
      delete window.__seek;
    };
  }, []);

  useEffect(() => {
    if (!ready || !params.play || params.hold) return;
    setPlaying(true);
  }, [ready]);

  useEffect(() => {
    if (!playing) return;
    const started = performance.now();
    let raf = 0;
    const loop = (now: number) => {
      const next = Math.min(SCENE_A.duration, (now - started) / 1000);
      setElapsed(next);
      if (next < SCENE_A.duration) raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  if (!params.scene) {
    return <Preview reduced={Boolean(osReduced)} />;
  }

  const t = playing ? elapsed : freezeT;
  const quiet = quietDev ?? t >= SCENE_A.quietAt;
  const frame = frameAt(SCENE_A, t, quiet);
  const frozen = !playing;

  return (
    <MotionConfig reducedMotion={reduced ? "always" : "never"}>
      <div
        className={reduced ? "reduced-motion" : "respect-os-motion"}
        style={{ position: "relative", width: 1920, height: 1080 }}
        data-scene-ready={ready ? "true" : "false"}
        data-t={t.toFixed(2)}
        data-cue={frame.cue?.id ?? "none"}
        data-speaker={frame.speaker?.id ?? "none"}
        data-audio={frame.audioOn ? "true" : "false"}
        data-quiet={quiet ? "true" : "false"}
        data-explainer={explainOn ? "true" : "false"}
      >
        {params.view === "glasses" ? (
          <MeetingGlasses
            scene={SCENE_A}
            frame={frame}
            quiet={quiet}
            reduced={reduced}
            frozen={frozen}
            onExplain={() => setExplainOn(true)}
            onToggleQuiet={() => setQuietDev(!quiet)}
            eyeContact
          />
        ) : (
          <MeetingDesktop
            scene={SCENE_A}
            frame={frame}
            quiet={quiet}
            reduced={reduced}
            frozen={frozen}
            onExplain={() => setExplainOn(true)}
            onToggleQuiet={() => setQuietDev(!quiet)}
            eyeContact
          />
        )}
        {explainOn ? <ExplainerCard onClose={() => setExplainOn(false)} /> : null}
        <HonestyTag />
      </div>
    </MotionConfig>
  );
}

function Preview({ reduced }: { reduced: boolean }) {
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(true);
  const [run, setRun] = useState(0);
  const [t, setT] = useState(0);
  const [view, setView] = useState<ViewMode>("desktop");
  const [demo, setDemo] = useState<Demo>("joke");
  const [explainOn, setExplainOn] = useState(false);
  const [quietOn, setQuietOn] = useState(false);
  const [sound, setSound] = useState(false);
  const [eyeContact, setEyeContact] = useState(true);
  const [settingsOn, setSettingsOn] = useState(false);
  const customYoutube = useOptionalImage("practice/custom-youtube.jpg");
  const customNetflix = useOptionalImage("practice/custom-netflix.jpg");
  const duration = clipDuration(demo);
  const field = isFieldDemo(demo);
  const simple = isSimpleDemo(demo);
  const meeting = isMeetingDemo(demo);
  const fieldClip = clipFor(demo);
  const barRef = useRef<HTMLDivElement>(null);
  const [barHeight, setBarHeight] = useState(96);
  const scale = useStageScale(barHeight);

  useEffect(() => {
    let live = true;
    document.fonts.ready.then(() => {
      if (live) setReady(true);
    });
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    const bar = barRef.current;
    if (!bar) return;
    const measure = () => setBarHeight(bar.offsetHeight);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(bar);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!playing || field) return;
    const started = performance.now() - t * 1000;
    let raf = 0;
    const loop = (now: number) => {
      const next = (now - started) / 1000;
      if (next >= duration) {
        setT(duration);
        setPlaying(false);
        return;
      }
      setT(next);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
    // t is read once per play/seek, then the clock owns it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, run, duration, field]);

  const quiet = meeting ? quietOn || t >= SCENE_A.quietAt : quietOn;
  const frame = frameAt(SCENE_A, t, quiet);
  const watchScene = demo === "netflix" ? NETFLIX_WATCH : YOUTUBE_WATCH;

  function choose(next: Demo) {
    setDemo(next);
    setExplainOn(false);
    setQuietOn(false);
    setView("desktop");
    if (!isFieldDemo(next) && !isMeetingDemo(next)) setSettingsOn(false);
    seek(0, true);
  }

  function openSettings() {
    if (settingsOn) {
      setSettingsOn(false);
      return;
    }
    setSettingsOn(true);
    setExplainOn(false);
    if (meeting || simple) return;
    if (!field) {
      setDemo("joke");
      setView("desktop");
      seek(1.2, false);
      return;
    }
    if (!fieldCueAt(fieldClip, t, false)?.eyeContact) seek(demo === "intro" ? 1.5 : 1.2, false);
  }

  function seek(time: number, play: boolean) {
    setT(time);
    setPlaying(play);
    setRun((value) => value + 1);
  }

  useEffect(() => {
    window.__fieldSeek = (time: number) => seek(time, false);
    return () => {
      delete window.__fieldSeek;
    };
  });

  function showMoment(time: number, next: { explain?: boolean; quiet?: boolean; view?: ViewMode }) {
    setExplainOn(Boolean(next.explain));
    setQuietOn(Boolean(next.quiet));
    if (next.view) setView(next.view);
    seek(time, false);
  }

  return (
    <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", background: "#1A2332" }}>
      <div
        ref={barRef}
        style={{
          flex: "0 0 auto",
          background: "#F7F5F0",
          color: "#14213D",
          padding: "12px 16px 14px",
          display: "flex",
          flexDirection: "column",
          gap: 8,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "baseline" }}>
          <div style={{ fontFamily: "var(--serif)", fontSize: 22, lineHeight: 1.2 }}>{demoTitle(demo)}</div>
          <div style={{ fontSize: 15, color: "#4A5568", lineHeight: 1.35 }}>
            {statusLine(demo, t, explainOn, quiet, eyeContact, settingsOn)}
          </div>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          <BarButton pressed={demo === "joke"} onClick={() => choose("joke")}>
            Work joke
          </BarButton>
          <BarButton pressed={demo === "intro"} onClick={() => choose("intro")}>
            Introductions
          </BarButton>
          <BarButton pressed={demo === "meeting"} onClick={() => choose("meeting")}>
            Meeting
          </BarButton>
          <BarButton pressed={demo === "youtube"} onClick={() => choose("youtube")}>
            YouTube
          </BarButton>
          <BarButton pressed={demo === "netflix"} onClick={() => choose("netflix")}>
            Netflix
          </BarButton>
          <BarButton pressed={demo === "glasses-joke"} onClick={() => choose("glasses-joke")}>
            Simple glasses: joke
          </BarButton>
          <BarButton pressed={demo === "glasses-intro"} onClick={() => choose("glasses-intro")}>
            Simple glasses: intro
          </BarButton>
          <BarButton pressed={demo === "glasses-meeting"} onClick={() => choose("glasses-meeting")}>
            Simple glasses: meeting
          </BarButton>
          <BarButton pressed={demo === "focus-joke"} onClick={() => choose("focus-joke")}>
            Noise cancelling for the eyes
          </BarButton>
          <BarButton
            pressed={playing}
            onClick={() => {
              if (t >= duration - 0.05) seek(0, true);
              else {
                setPlaying((value) => !value);
                setRun((value) => value + 1);
              }
            }}
          >
            {playing ? "Pause" : "Play"}
          </BarButton>
          <BarButton pressed={false} onClick={() => seek(0, true)}>
            Restart
          </BarButton>
          {field
            ? fieldClip.jumps.map((jump) => (
                <BarButton key={jump.label} pressed={false} onClick={() => showMoment(jump.t, {})}>
                  {jump.label}
                </BarButton>
              ))
            : null}
          {field ? (
            <BarButton pressed={sound} onClick={() => setSound((value) => !value)}>
              {sound ? "Sound on" : "Sound"}
            </BarButton>
          ) : null}
          <BarButton pressed={settingsOn} onClick={openSettings}>
            Settings
          </BarButton>
          {meeting ? (
            <>
              <BarButton pressed={false} onClick={() => showMoment(7.6, {})}>
                Sarcasm
              </BarButton>
              <BarButton pressed={false} onClick={() => showMoment(12, {})}>
                Idiom
              </BarButton>
              <BarButton pressed={false} onClick={() => showMoment(19.4, {})}>
                {simple ? "Your turn" : "Waiting"}
              </BarButton>
              <BarButton pressed={false} onClick={() => showMoment(28.2, {})}>
                Topic
              </BarButton>
            </>
          ) : null}
          {demo === "meeting" ? (
            <>
              <BarButton pressed={view === "desktop"} onClick={() => setView("desktop")}>
                Laptop
              </BarButton>
              <BarButton pressed={view === "glasses"} onClick={() => setView("glasses")}>
                Glasses
              </BarButton>
            </>
          ) : null}
          {demo === "youtube" ? (
            <>
              <BarButton pressed={false} onClick={() => showMoment(4, {})}>
                Comedy
              </BarButton>
              <BarButton pressed={false} onClick={() => showMoment(12, {})}>
                Advert
              </BarButton>
              <BarButton pressed={false} onClick={() => showMoment(19, {})}>
                Back to the video
              </BarButton>
            </>
          ) : null}
          {demo === "netflix" ? (
            <>
              <BarButton pressed={false} onClick={() => showMoment(5, {})}>
                Unkind
              </BarButton>
              <BarButton pressed={false} onClick={() => showMoment(12, {})}>
                Tension
              </BarButton>
              <BarButton pressed={false} onClick={() => showMoment(20, {})}>
                Recap
              </BarButton>
            </>
          ) : null}
          <BarButton
            pressed={explainOn}
            onClick={() => showMoment(sampleTime(demo), { explain: true })}
          >
            What is this?
          </BarButton>
          <BarButton
            pressed={quietOn}
            onClick={() => showMoment(sampleTime(demo), { quiet: true })}
          >
            Quiet
          </BarButton>
        </div>
        {demo === "focus-joke" ? (
          <div style={{ fontSize: 13, color: "#4A5568", lineHeight: 1.4 }}>
            Noise cancelling for the eyes. Everything is softened except the face of the person Gloss is following. When the camera moves to the other person, her face comes into focus. The top bar and the rapport edge stay.
          </div>
        ) : simple ? (
          <div style={{ fontSize: 13, color: "#4A5568", lineHeight: 1.4 }}>
            Three signals only. The top bar shows someone else is talking. Look here sits on the speaker. The colour around the edge shows rapport: green for good, orange for mixed, grey when Gloss is not sure.
          </div>
        ) : field ? (
          <div style={{ fontSize: 13, color: "#4A5568", lineHeight: 1.4 }}>
            The colour bar means someone else is talking, and Look here follows them. When the bar goes, there is a place for you to speak. Eye contact reminders can be turned off in Settings.
          </div>
        ) : null}
        {demo === "youtube" || demo === "netflix" ? (
          <div style={{ fontSize: 13, color: "#4A5568", lineHeight: 1.4 }}>
            The picture is stock footage of real people. Save a screenshot as public/practice/custom-youtube.jpg or
            public/practice/custom-netflix.jpg and reload to use yours.
          </div>
        ) : null}
      </div>
        <div style={{ flex: 1, display: "grid", placeItems: "center", minHeight: 0, position: "relative" }}>
        {settingsOn && meeting ? (
          <div style={{ position: "absolute", top: 16, left: 16, zIndex: 40, maxHeight: "calc(100% - 32px)", overflow: "auto" }}>
            <SettingsPanel eyeContact={eyeContact} onToggleEyeContact={() => setEyeContact((value) => !value)} />
          </div>
        ) : null}
        {field ? (
          <div
            className={reduced ? "reduced-motion" : "respect-os-motion"}
            style={{ position: "relative", width: "100%", height: "100%", minHeight: 0 }}
          >
            <div style={{ position: "relative", width: "100%", height: "100%" }}>
              <FieldWatch
                clip={fieldClip}
                t={t}
                playing={playing && !explainOn}
                seekId={run}
                quiet={quiet}
                sound={sound}
                eyeContact={eyeContact}
                reduced={reduced}
                simple={simple}
                focus={demo === "focus-joke"}
                onTime={(time) => setT(time)}
                onEnded={() => {
                  setT(fieldClip.duration);
                  setPlaying(false);
                }}
              />
              {settingsOn ? (
                <div style={{ position: "absolute", top: 16, left: 16, zIndex: 40, maxHeight: "calc(100% - 32px)", overflow: "auto" }}>
                  <SettingsPanel eyeContact={eyeContact} onToggleEyeContact={() => setEyeContact((value) => !value)} />
                </div>
              ) : null}
              {explainOn ? <ExplainerCard onClose={() => setExplainOn(false)} /> : null}
              <HonestyTag />
            </div>
          </div>
        ) : (
        <div style={{ width: 1920 * scale, height: 1080 * scale }}>
          <MotionConfig reducedMotion={reduced ? "always" : "never"}>
            <div
              className={reduced ? "reduced-motion" : undefined}
              style={{
                width: 1920,
                height: 1080,
                transform: `scale(${scale})`,
                transformOrigin: "top left",
                position: "relative",
              }}
              data-scene-ready={ready ? "true" : "false"}
            >
              {demo === "meeting" && view === "glasses" ? (
                <MeetingGlasses
                  scene={SCENE_A}
                  frame={frame}
                  quiet={quiet}
                  reduced={reduced}
                  frozen={!playing}
                  onExplain={() => setExplainOn(true)}
                  onToggleQuiet={() => setQuietOn((value) => !value)}
                  eyeContact={eyeContact}
                />
              ) : null}
              {demo === "glasses-meeting" ? (
                <SimpleGlassesMeeting scene={SCENE_A} frame={frame} quiet={quiet} reduced={reduced} eyeContact={eyeContact} />
              ) : null}
              {demo === "meeting" && view === "desktop" ? (
                <MeetingDesktop
                  scene={SCENE_A}
                  frame={frame}
                  quiet={quiet}
                  reduced={reduced}
                  frozen={!playing}
                  onExplain={() => setExplainOn(true)}
                  onToggleQuiet={() => setQuietOn((value) => !value)}
                  eyeContact={eyeContact}
                />
              ) : null}
              {demo === "youtube" || demo === "netflix" ? (
                <PracticeWatch
                  variant={demo}
                  scene={watchScene}
                  t={t}
                  quiet={quiet}
                  reduced={reduced}
                  frozen={!playing}
                  customStill={demo === "youtube" ? customYoutube : customNetflix}
                  onExplain={() => setExplainOn(true)}
                  onToggleQuiet={() => setQuietOn((value) => !value)}
                />
              ) : null}
              {explainOn ? <ExplainerCard onClose={() => setExplainOn(false)} /> : null}
              <HonestyTag />
            </div>
          </MotionConfig>
        </div>
        )}
      </div>
    </div>
  );
}

function statusLine(demo: Demo, t: number, explain: boolean, quiet: boolean, eyeContact: boolean, settingsOn: boolean) {
  if (explain) return "The card someone can show to another person. Nothing is recorded.";
  if (settingsOn && (isFieldDemo(demo) || isMeetingDemo(demo))) {
    return eyeContact
      ? "Eye contact reminders are on. Look here sits on their face."
      : "Eye contact reminders are off. Look here is left out.";
  }
  if (demo === "focus-joke") {
    if (quiet) return "Quiet mode. Nothing is blurred. The clip keeps going.";
    return fieldStatus(JOKE_CLIP, t, quiet, false).replace(
      "The colour bar stays across the top.",
      "Her face stays in focus. The rest is softened.",
    );
  }
  if (isFieldDemo(demo)) return fieldStatus(clipFor(demo), t, quiet, eyeContact);
  if (demo === "youtube") {
    if (quiet) return "Quiet mode. The video keeps playing. The suggestion is paused.";
    if (t >= 16.8) return "The advert mention is over. Gloss rests.";
    if (t >= 9.2) return "She says a company paid for this part. Gloss marks it as an advert.";
    if (t >= 2.2) return "The praise may be a joke. Gloss calls it exaggeration, as a suggestion.";
    return "Mara starts the vlog. Gloss has nothing to add yet.";
  }
  if (demo === "netflix") {
    if (quiet) return "Quiet mode. The episode keeps playing. The suggestion is paused.";
    if (t >= 18.2) return "After the scene, Gloss offers two moments to revisit.";
    if (t >= 9.4) return "Nobody speaks. Gloss suggests there may be tension.";
    if (t >= 2.4) return "Helen smiles. The words may be unkind. Gloss does not tell anyone what to do.";
    return "Sunday lunch. Gloss waits before it suggests anything.";
  }
  if (othersAreTalking(SCENE_A, frameAt(SCENE_A, t, quiet))) {
    return "Someone else is talking. The colour bar stays across the top.";
  }
  if (t >= 17.6 && t < 25.2) return "The bar has gone. It is your turn to talk.";
  if (quiet) return "Quiet mode. Sam’s line is still there. The suggestion is paused.";
  if (t >= 31) return "Alex turned cues off. The meeting continues.";
  if (t >= 27) return "Jordan moves to the budget. Gloss notes the topic change.";
  if (t >= 21.4 && t < 25.2) return "Alex answers. Gloss does not comment on Alex.";
  if (t >= 18) return "Nobody is speaking. Gloss suggests Priya may be waiting.";
  if (t >= 10.8) return "Priya says “take this offline”. Gloss explains the idiom.";
  if (t >= 6.2) return "Sam’s praise may be sarcastic. The note is a suggestion.";
  return "Priya gives the news. Gloss has nothing to add yet.";
}

function demoTitle(demo: Demo) {
  if (demo === "joke") return "In the room · work joke";
  if (demo === "intro") return "In the room · introductions";
  if (demo === "glasses-joke") return "Simple glasses · work joke";
  if (demo === "glasses-intro") return "Simple glasses · introductions";
  if (demo === "glasses-meeting") return "Simple glasses · work video meeting";
  if (demo === "focus-joke") return "Noise cancelling for the eyes · work joke";
  if (demo === "youtube") return "Practice zone · YouTube";
  if (demo === "netflix") return "Practice zone · Netflix";
  return "Scene A · work video meeting";
}

function clipDuration(demo: Demo) {
  if (isFieldDemo(demo)) return clipFor(demo).duration;
  if (demo === "youtube") return YOUTUBE_WATCH.duration;
  if (demo === "netflix") return NETFLIX_WATCH.duration;
  return SCENE_A.duration;
}

function sampleTime(demo: Demo) {
  if (isFieldDemo(demo)) return 5.6;
  if (demo === "youtube") return 4;
  if (demo === "netflix") return 5;
  return 7.6;
}

function useOptionalImage(src: string) {
  const [ok, setOk] = useState(false);
  useEffect(() => {
    let live = true;
    const img = new Image();
    img.onload = () => {
      if (live) setOk(true);
    };
    img.onerror = () => {
      if (live) setOk(false);
    };
    img.src = src;
    return () => {
      live = false;
    };
  }, [src]);
  return ok ? src : null;
}

function BarButton({
  children,
  onClick,
  pressed,
}: {
  children: string;
  onClick: () => void;
  pressed: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={pressed}
      style={{
        border: "1px solid rgba(20, 33, 61, 0.22)",
        background: pressed ? "#14213D" : "transparent",
        color: pressed ? "#F7F5F0" : "#14213D",
        borderRadius: 999,
        padding: "7px 12px",
        fontSize: 14,
        fontWeight: 600,
      }}
    >
      {children}
    </button>
  );
}

function useStageScale(barHeight: number) {
  const [scale, setScale] = useState(1);
  useEffect(() => {
    const fit = () => {
      const sx = window.innerWidth / 1920;
      const sy = Math.max(window.innerHeight - barHeight, 200) / 1080;
      setScale(Math.min(sx, sy));
    };
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, [barHeight]);
  return scale;
}
