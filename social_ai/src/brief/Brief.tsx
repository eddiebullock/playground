import { useEffect, useMemo, useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import jokeData from "../../public/field/joke.mp4?inline";
import { JOKE_CLIP } from "../content/field";
import { FieldWatch } from "../scenes/FieldWatch";

const NAVY = "#14213D";
const BODY = "#4A5568";
const PAPER = "#F7F5F0";
const RULE = "rgba(20, 33, 61, 0.14)";
const PREPRINT =
  "https://scholar.google.com/citations?view_op=view_citation&hl=en&user=toL9Lq0AAAAJ&citation_for_view=toL9Lq0AAAAJ:IjCSPb-OGe4C";

/** The clip ships inside this file as base64. A blob URL lets every browser seek and loop it. */
function useClipUrl() {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    const [, base64] = jokeData.split(",");
    const bytes = Uint8Array.from(atob(base64), (char) => char.charCodeAt(0));
    const next = URL.createObjectURL(new Blob([bytes], { type: "video/mp4" }));
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, []);
  return url;
}

function useReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(query.matches);
    const change = () => setReduced(query.matches);
    query.addEventListener("change", change);
    return () => query.removeEventListener("change", change);
  }, []);
  return reduced;
}

export function Brief() {
  const src = useClipUrl();
  const reduced = useReducedMotion();

  return (
    <main
      style={{
        maxWidth: 1120,
        margin: "0 auto",
        padding: "36px 32px 28px",
        color: NAVY,
        fontFamily: "var(--sans)",
        fontSize: 15,
        lineHeight: 1.5,
      }}
    >
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", gap: 24, flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase", color: BODY }}>
            Autism Research Centre · University of Cambridge
          </div>
          <h1 style={{ margin: "6px 0 0", fontFamily: "var(--serif)", fontWeight: 400, fontSize: 34, lineHeight: 1.15 }}>
            Smart glasses that explain social situations, in real time
          </h1>
        </div>
        <div style={{ fontSize: 14, color: BODY }}>
          Eddie Bullock, PhD student (year 2)
          <br />
          Supervisors: Simon Baron-Cohen, Carrie Allison, Simon Braschi, Charles Nduka
        </div>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1.1fr)", gap: 28, marginTop: 24 }}>
        <section>
          <Heading>The idea</Heading>
          <p style={para}>
            Much of social life runs on unspoken conventions. Wearable glasses combined with vision-language models
            can make that context explicit in the moment, so social interaction becomes more accessible for autistic
            people, on their own terms.
          </p>
          <p style={para}>
            The user stays in control: they choose what to see and how, every cue is a suggestion, the system says
            when it is unsure, and it is private by design, with nothing recorded.
          </p>
        </section>

        <section>
          <Heading>Why now: models already read mental states well</Heading>
          <p style={{ ...para, marginBottom: 10 }}>
            Our preprint,{" "}
            <a href={PREPRINT} style={link}>
              How accurate are Large Language Models on Multimodal Theory-of-Mind Tasks? A comparison with Humans
            </a>
            , tested four commercial models on validated video batteries. On EU-Emotion (20 mental states), the top
            models beat the human benchmark.
          </p>
          <Bars />
          <p style={{ ...para, fontSize: 13, color: BODY, marginTop: 10 }}>
            Audio only, Gemini 3 Flash scored 58.5% against a human benchmark of 45.2%. On Mindreading (359 states)
            its accuracy rose from 65.4% with video alone to 85.3% with audio and video. Models remain weaker on
            complex and low-intensity states.
          </p>
        </section>
      </div>

      <section style={{ marginTop: 22 }}>
        <Heading>Three ways to show the same moment</Heading>
        <p style={{ ...para, marginBottom: 14 }}>
          Filmed with Meta glasses. The same 19-second clip of two colleagues joking about work, shown three ways.
          Click a video to pause it.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 20 }}>
          <DemoCard
            title="1. Full notes"
            text="Short written cues sit beside the speaker: social context and rapport, a Look here eye-contact marker, and a bar that disappears when it is your turn."
            src={src}
            reduced={reduced}
          />
          <DemoCard
            title="2. Simple"
            text="No words. Three signals only: the turn-taking bar, Look here on the speaker, and a coloured edge for rapport."
            src={src}
            reduced={reduced}
            simple
          />
          <DemoCard
            title="3. Noise cancelling for the eyes"
            text="Everything is softened except the face of the person speaking. Focus moves when the conversation does."
            src={src}
            reduced={reduced}
            simple
            focus
          />
        </div>
      </section>

      <section style={{ marginTop: 22, padding: "16px 20px", background: "#fff", border: `1px solid ${RULE}`, borderRadius: 14 }}>
        <Heading>What we are testing now</Heading>
        <p style={{ ...para, marginBottom: 12 }}>
          We are running user testing with autistic adults to find which level and modality of feedback is actually
          helpful, for each kind of social information.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 18 }}>
          <Column label="Information">
            {["Social context", "Rapport", "Eye contact", "Turn-taking", "Emotion recognition"]}
          </Column>
          <Column label="Level">{["Detailed notes", "Minimal signals", "Perceptual (focus and blur)", "Off"]}</Column>
          <Column label="Modality">{["Text on the lens", "Ambient colour and focus", "Audio tones or words", "Haptic pulses"]}</Column>
        </div>
      </section>

      <section style={{ marginTop: 18 }}>
        <Heading>Ongoing projects</Heading>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 20 }}>
          {PROJECTS.map((project) => (
            <div key={project.title}>
              <div style={{ fontWeight: 700, fontSize: 15 }}>{project.title}</div>
              <div style={{ fontSize: 13.5, color: BODY, lineHeight: 1.45 }}>{project.text}</div>
            </div>
          ))}
        </div>
      </section>

      <footer
        style={{
          marginTop: 18,
          paddingTop: 12,
          borderTop: `1px solid ${RULE}`,
          display: "flex",
          justifyContent: "space-between",
          gap: 16,
          flexWrap: "wrap",
          fontSize: 13,
          color: BODY,
        }}
      >
        <span>Concept prototype. Cues are scripted for this clip and are not live model output.</span>
        <span>
          Eddie Bullock ·{" "}
          <a href="mailto:eb2007@cam.ac.uk" style={link}>
            eb2007@cam.ac.uk
          </a>
        </span>
      </footer>
    </main>
  );
}

const para: CSSProperties = { margin: "0 0 8px", color: NAVY };
const link: CSSProperties = { color: "#2C6488", fontWeight: 600 };

function Heading({ children }: { children: ReactNode }) {
  return <h2 style={{ margin: "0 0 8px", fontFamily: "var(--serif)", fontWeight: 400, fontSize: 21, lineHeight: 1.2 }}>{children}</h2>;
}

const PROJECTS = [
  {
    title: "Community focus groups",
    text: "Focus groups with autistic adults to understand what the community wants from this technology, so it aligns with their values.",
  },
  {
    title: "Mechanistic interpretability",
    text: "Looking inside vision-language models to understand how they represent mental states, and why they succeed or fail.",
  },
  {
    title: "Feasibility and acceptability",
    text: "Testing whether assistive AI for social interaction is practical, comfortable and welcome in everyday life.",
  },
];

const RESULTS: { label: string; value: number; human?: boolean; note?: string }[] = [
  { label: "Human benchmark", value: 63.0, human: true },
  { label: "Claude Opus 4.5", value: 72.0, note: "marginal" },
  { label: "GPT-5", value: 73.7 },
  { label: "Gemini 3 Flash", value: 74.6 },
  { label: "Gemini 3 Flash, audio and video", value: 81.4 },
];

function Bars() {
  return (
    <figure style={{ margin: 0 }} aria-label="EU-Emotion accuracy, models compared with the human benchmark">
      <div style={{ display: "grid", gap: 5 }}>
        {RESULTS.map((row) => (
          <div key={row.label} style={{ display: "grid", gridTemplateColumns: "200px 1fr 52px", alignItems: "center", gap: 10, fontSize: 13 }}>
            <span style={{ color: row.human ? BODY : NAVY, fontWeight: row.human ? 600 : 500 }}>{row.label}</span>
            <span style={{ height: 12, background: "rgba(20,33,61,0.07)", borderRadius: 999, overflow: "hidden" }}>
              <span
                style={{
                  display: "block",
                  height: "100%",
                  width: `${row.value}%`,
                  borderRadius: 999,
                  background: row.human ? "#8A94A6" : "#3A7CA5",
                }}
              />
            </span>
            <span style={{ fontVariantNumeric: "tabular-nums", fontWeight: 600 }}>{row.value.toFixed(1)}%</span>
          </div>
        ))}
      </div>
      <figcaption style={{ fontSize: 12, color: BODY, marginTop: 6 }}>
        EU-Emotion accuracy, video only unless stated. GPT-5 and Gemini 3 Flash were significantly above the human
        benchmark (p &lt; .05); Claude Opus 4.5 was marginal (p = .051).
      </figcaption>
    </figure>
  );
}

function Column({ label, children }: { label: string; children: string[] }) {
  return (
    <div>
      <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", color: BODY, marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {children.map((item) => (
          <span
            key={item}
            style={{ border: `1px solid ${RULE}`, background: PAPER, borderRadius: 999, padding: "4px 10px", fontSize: 13, fontWeight: 500 }}
          >
            {item}
          </span>
        ))}
      </div>
    </div>
  );
}

const STAGE = { w: 390, h: 520 };

function DemoCard({
  title,
  text,
  src,
  reduced,
  simple = false,
  focus = false,
}: {
  title: string;
  text: string;
  src: string | null;
  reduced: boolean;
  simple?: boolean;
  focus?: boolean;
}) {
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [seekId, setSeekId] = useState(0);
  const [sound, setSound] = useState(false);
  const clip = useMemo(() => (src ? { ...JOKE_CLIP, src } : null), [src]);
  const scale = 0.82;

  return (
    <figure style={{ margin: 0 }}>
      <div
        role="button"
        tabIndex={0}
        aria-label={`${title}. ${playing ? "Pause" : "Play"} video`}
        onClick={() => {
          setPlaying((value) => !value);
          setSeekId((value) => value + 1);
        }}
        onKeyDown={(event) => {
          if (event.key === " " || event.key === "Enter") {
            event.preventDefault();
            setPlaying((value) => !value);
            setSeekId((value) => value + 1);
          }
        }}
        className={reduced ? "reduced-motion" : "respect-os-motion"}
        style={{
          position: "relative",
          width: STAGE.w * scale,
          height: STAGE.h * scale,
          margin: "0 auto",
          borderRadius: simple ? 30 : 14,
          overflow: "hidden",
          background: "#0e1218",
          cursor: "pointer",
        }}
      >
        <div style={{ position: "absolute", left: 0, top: 0, width: STAGE.w, height: STAGE.h, transform: `scale(${scale})`, transformOrigin: "top left" }}>
          {clip ? (
            <FieldWatch
              clip={clip}
              t={t}
              playing={playing}
              seekId={seekId}
              quiet={false}
              sound={sound}
              eyeContact
              reduced={reduced}
              simple={simple}
              focus={focus}
              wordmark={false}
              onTime={setT}
              onEnded={() => {
                setT(0);
                setSeekId((value) => value + 1);
                setPlaying(true);
              }}
            />
          ) : null}
        </div>
        {!playing ? (
          <div
            aria-hidden="true"
            style={{
              position: "absolute",
              inset: 0,
              display: "grid",
              placeItems: "center",
              background: "rgba(14,18,24,0.35)",
              color: PAPER,
              fontWeight: 700,
              fontSize: 16,
              zIndex: 20,
            }}
          >
            Paused. Click to play
          </div>
        ) : null}
      </div>
      <div style={{ display: "flex", justifyContent: "center", gap: 8, marginTop: 8 }}>
        <SmallButton
          onClick={() => {
            setT(0);
            setSeekId((value) => value + 1);
            setPlaying(true);
          }}
        >
          Restart
        </SmallButton>
        <SmallButton pressed={sound} onClick={() => setSound((value) => !value)}>
          {sound ? "Sound on" : "Sound off"}
        </SmallButton>
      </div>
      <figcaption style={{ marginTop: 8 }}>
        <div style={{ fontWeight: 700, fontSize: 15 }}>{title}</div>
        <div style={{ fontSize: 13.5, color: BODY, lineHeight: 1.45 }}>{text}</div>
      </figcaption>
    </figure>
  );
}

function SmallButton({ children, onClick, pressed = false }: { children: string; onClick: () => void; pressed?: boolean }) {
  return (
    <button
      type="button"
      aria-pressed={pressed}
      onClick={onClick}
      style={{
        border: `1px solid ${RULE}`,
        background: pressed ? NAVY : "#fff",
        color: pressed ? PAPER : NAVY,
        borderRadius: 999,
        padding: "5px 12px",
        fontSize: 13,
        fontWeight: 600,
        fontFamily: "var(--sans)",
        cursor: "pointer",
      }}
    >
      {children}
    </button>
  );
}
