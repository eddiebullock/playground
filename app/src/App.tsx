import { useEffect, useRef, useState } from "react";
import { DEFAULT_BASE, listModels } from "./engine/llm.ts";
import { reflect, type Mode, type ReflectInput, type ReflectResult } from "./engine/reflect.ts";
import { addEntry, clearEntries, exportEntries, loadEntries, type Verdict } from "./log.ts";

const RELATIONS = ["", "friend", "colleague", "manager", "partner", "family member", "someone I just met", "teacher or tutor"];
const CHANNELS = ["", "text message", "email", "work chat (Slack, Teams)", "in person", "video call", "phone call", "social media"];

const CLARITY_TEXT = {
  clear: { word: "Fairly clear", detail: "One reading stands out." },
  some: { word: "Somewhat open", detail: "One reading leads, but others are possible." },
  high: { word: "Hard to tell", detail: "The words alone do not settle it. Asking is reasonable." },
} as const;

function likelihood(p: number): string {
  if (p >= 0.6) return "Likely";
  if (p >= 0.25) return "Possible";
  return "Less likely";
}

function readStored(key: string, fallback: string): string {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

function store(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage unavailable; settings just will not persist */
  }
}

export default function App() {
  const [mode, setMode] = useState<Mode>("them");
  const [person, setPerson] = useState("");
  const [relation, setRelation] = useState("");
  const [channel, setChannel] = useState("");
  const [context, setContext] = useState("");
  const [text, setText] = useState("");
  const [intended, setIntended] = useState("");
  const [busy, setBusy] = useState<null | "writing" | "weighing">(null);
  const [result, setResult] = useState<ReflectResult | null>(null);
  const [lastInput, setLastInput] = useState<ReflectInput | null>(null);
  const [error, setError] = useState("");
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const [note, setNote] = useState("");
  const [base, setBase] = useState(() => readStored("gloss.base", DEFAULT_BASE));
  const [model, setModel] = useState(() => readStored("gloss.model", "qwen3:4b"));
  const [models, setModels] = useState<string[]>([]);
  const [showSettings, setShowSettings] = useState(false);
  const [saved, setSaved] = useState(() => loadEntries().length);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    listModels(base)
      .then((m) => {
        setModels(m);
        setError("");
      })
      .catch(() => setError("Gloss cannot reach the local model. Is Ollama running?"));
  }, [base]);

  useEffect(() => store("gloss.model", model), [model]);
  useEffect(() => store("gloss.base", base), [base]);

  async function run() {
    if (!text.trim() || busy) return;
    const input: ReflectInput = { mode, text, context, person, relation, channel, intended: mode === "me" ? intended : undefined };
    abort.current = new AbortController();
    setError("");
    setResult(null);
    setVerdict(null);
    setNote("");
    try {
      const r = await reflect(base, model, input, abort.current.signal, setBusy);
      setResult(r);
      setLastInput(input);
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(String((e as Error).message ?? e));
    } finally {
      setBusy(null);
    }
  }

  function give(v: Verdict) {
    if (!result || !lastInput) return;
    setVerdict(v);
    setSaved(addEntry({ at: new Date().toISOString(), input: lastInput, result, verdict: v, note }));
  }

  const isThem = mode === "them";
  const who = person.trim() || (isThem ? "they" : "them");

  return (
    <div className="page">
      <header className="top">
        <div>
          <h1>Gloss</h1>
          <p className="tagline">Possible readings, not answers.</p>
        </div>
        <div className="top-right">
          <span className="tag">Research prototype. Runs on this computer.</span>
          <button className="link" onClick={() => setShowSettings((s) => !s)} aria-expanded={showSettings}>
            Settings
          </button>
        </div>
      </header>

      {showSettings && (
        <section className="card settings" aria-label="Settings">
          <label>
            Model
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {[model, ...models.filter((m) => m !== model)].map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </label>
          <label>
            Local model server
            <input value={base} onChange={(e) => setBase(e.target.value)} />
          </label>
          <div className="row">
            <span>{saved} saved {saved === 1 ? "entry" : "entries"} on this computer.</span>
            <button onClick={() => exportEntries()} disabled={!saved}>
              Export
            </button>
            <button
              onClick={() => {
                if (confirm("Delete all saved entries from this computer?")) {
                  clearEntries();
                  setSaved(0);
                }
              }}
              disabled={!saved}
            >
              Delete all
            </button>
          </div>
        </section>
      )}

      <div className="mode" role="tablist" aria-label="What do you want to read?">
        <button role="tab" aria-selected={isThem} onClick={() => setMode("them")}>
          What did they mean?
        </button>
        <button role="tab" aria-selected={!isThem} onClick={() => setMode("me")}>
          How might my message read?
        </button>
      </div>

      <main className="grid">
        <section className="card form" aria-label="Your situation">
          <div className="pair">
            <label>
              {isThem ? "Who said it" : "Who it is for"}
              <input value={person} onChange={(e) => setPerson(e.target.value)} placeholder="A name or initial" />
            </label>
            <label>
              They are my
              <select value={relation} onChange={(e) => setRelation(e.target.value)}>
                {RELATIONS.map((r) => (
                  <option key={r} value={r}>
                    {r || "(skip)"}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Where
              <select value={channel} onChange={(e) => setChannel(e.target.value)}>
                {CHANNELS.map((c) => (
                  <option key={c} value={c}>
                    {c || "(skip)"}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <label>
            What came before <span className="hint">optional: earlier messages, or what was happening</span>
            <textarea rows={4} value={context} onChange={(e) => setContext(e.target.value)} />
          </label>
          <label>
            {isThem ? `What ${who} said` : "My message"}
            <textarea
              rows={3}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run();
              }}
            />
          </label>
          {!isThem && (
            <label>
              What I mean by it <span className="hint">optional</span>
              <input value={intended} onChange={(e) => setIntended(e.target.value)} />
            </label>
          )}
          <div className="row">
            <button className="primary" onClick={run} disabled={!text.trim() || !!busy}>
              {busy ? (busy === "writing" ? "Writing possible readings..." : "Weighing them...") : "Read it"}
            </button>
            {busy && (
              <button className="link" onClick={() => abort.current?.abort()}>
                Stop
              </button>
            )}
          </div>
          <p className="privacy">Names and words stay on this computer. Use initials if you prefer.</p>
        </section>

        <section className="card result" aria-live="polite" aria-label="Readings">
          {error && <p className="error">{error}</p>}
          {!result && !error && !busy && (
            <p className="empty">
              {isThem
                ? "Paste what someone said. Gloss offers a few ways it could be meant, including that they meant it exactly as said."
                : "Paste a message you plan to send. Gloss shows how it might come across, so you can decide whether to change anything. You do not have to."}
            </p>
          )}
          {busy && <p className="empty">Thinking it through. This can take up to a minute on a laptop.</p>}
          {result && (
            <>
              <div className={`clarity clarity-${result.clarity}`}>
                <strong>{CLARITY_TEXT[result.clarity].word}</strong>
                <span>{CLARITY_TEXT[result.clarity].detail}</span>
              </div>
              <p className="literal">
                <span className="label">The words say</span>
                {result.literal}
              </p>
              <ol className="readings">
                {result.readings.map((r, i) => (
                  <li key={i}>
                    <div className="reading-head">
                      <span className="likelihood">{result.abstain ? "Possible" : likelihood(r.p)}</span>
                      <span className="bar" aria-hidden="true">
                        <span style={{ width: `${Math.round(r.p * 100)}%` }} />
                      </span>
                      {r.literal && <span className="face">{isThem ? "Taken at face value" : "As you mean it"}</span>}
                    </div>
                    <p>{r.meaning}</p>
                    {r.cues.length > 0 && <p className="cues">Based on: {r.cues.join("; ")}</p>}
                  </li>
                ))}
              </ol>
              {result.check && (
                <p className="check">
                  <span className="label">{isThem ? "To check, you could ask" : "If you want, you could add"}</span>
                  {result.check}
                </p>
              )}
              <div className="feedback">
                <span>Was this helpful?</span>
                {(["right", "partly", "wrong", "unsure"] as Verdict[]).map((v) => (
                  <button key={v} aria-pressed={verdict === v} onClick={() => give(v)}>
                    {{ right: "Yes", partly: "Partly", wrong: "No", unsure: "Not sure" }[v]}
                  </button>
                ))}
                <input
                  className="note"
                  placeholder="Anything to add? (optional, saved only here)"
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                  onBlur={() => verdict && give(verdict)}
                />
              </div>
              <p className="meta">
                {result.model}, {(result.ms / 1000).toFixed(0)} s. Gloss can be wrong. You know the people involved; it does not.
              </p>
            </>
          )}
        </section>
      </main>
    </div>
  );
}
