import { useEffect, useRef, useState } from "react";
import { DEFAULT_BASE, listModels } from "./engine/llm.ts";
import {
  addOwnReading,
  reflect,
  styleNote,
  waysToReply,
  writeCheck,
  type Draft,
  type Reply,
  type ReflectInput,
  type ReflectResult,
} from "./engine/reflect.ts";
import { EVIDENCE } from "./evidence.ts";
import { addEntry, clearEntries, exportEntries, loadEntries, type Verdict } from "./log.ts";

const RELATIONS = ["", "friend", "colleague", "manager", "partner", "family member", "someone I just met", "teacher or tutor"];
const CHANNELS = ["", "text message", "email", "work chat (Slack, Teams)", "in person", "video call", "phone call", "social media"];

const CLARITY_TEXT = {
  clear: { word: "Fairly clear", detail: "One reading stands out." },
  some: { word: "Somewhat open", detail: "One reading leads, but others are possible." },
  high: { word: "Hard to tell", detail: "The words alone do not settle it. Asking is reasonable." },
} as const;

// In "my message" mode the question is whether it will read as meant.
const CLARITY_TEXT_ME = {
  clear: { word: "Fairly clear", detail: "Most people would read it one way." },
  some: { word: "Somewhat open", detail: "One reading leads, but people could take it differently." },
  high: { word: "Open to different readings", detail: "People could take this in more than one way." },
} as const;

// Private notes about people, keyed by the name the user typed. They never
// leave this computer and are only added as context when that name is used.
function loadPeople(): Record<string, string> {
  try {
    const raw = localStorage.getItem("cue.people") ?? "{}";
    return JSON.parse(raw) as Record<string, string>;
  } catch {
    return {};
  }
}

function savePerson(name: string, note: string) {
  const key = name.trim().toLowerCase();
  if (!key) return;
  const all = loadPeople();
  if (note.trim()) all[key] = note.trim();
  else delete all[key];
  store("people", JSON.stringify(all));
}

function likelihood(p: number): string {
  if (p >= 0.6) return "Likely";
  if (p >= 0.25) return "Possible";
  return "Less likely";
}

function readStored(key: string, fallback: string): string {
  try {
    return localStorage.getItem(`cue.${key}`) ?? fallback;
  } catch {
    return fallback;
  }
}

function store(key: string, value: string) {
  try {
    localStorage.setItem(`cue.${key}`, value);
  } catch {
    /* storage unavailable; settings just will not persist */
  }
}

type Tab = "them" | "me" | "style";

function About({ onClose }: { onClose: () => void }) {
  return (
    <section className="card about" aria-label="About Cue">
      <h2>About Cue</h2>
      <ul>
        <li>Cue offers a few possible readings of a message or moment, and says how clear or open it seems. It does not know the people involved. You do.</li>
        <li>It always keeps the plain reading in view, because many messages mean exactly what they say.</li>
        <li>It can also show how your own message might come across. You do not have to change anything. Different people communicate differently, and that goes both ways.</li>
        <li>Everything runs on this computer. Nothing you type is sent anywhere.</li>
        <li>It is a research prototype, and it can be wrong. It is not a medical or support service.</li>
      </ul>
      <details className="evidence">
        <summary>How good is it?</summary>
        <ul>
          {EVIDENCE.points.map((x) => (
            <li key={x.head}>
              <strong>{x.head}</strong> {x.body}
            </li>
          ))}
        </ul>
        <p className="meta">
          Tested with {EVIDENCE.model} on public research items, {EVIDENCE.updated}. Not yet tested with autistic adults' own messages; that is what the study is for.
        </p>
      </details>
      <button className="primary" onClick={onClose}>
        Got it
      </button>
    </section>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("them");
  const mode = tab === "me" ? "me" : "them";
  const [showAbout, setShowAbout] = useState(() => readStored("seenAbout", "") !== "1");
  const [style, setStyle] = useState("");
  const [note, setNoteText] = useState("");
  const [person, setPerson] = useState("");
  const [relation, setRelation] = useState("");
  const [channel, setChannel] = useState("");
  const [context, setContext] = useState("");
  const [aboutMe, setAboutMe] = useState("");
  const [text, setText] = useState("");
  const [intended, setIntended] = useState("");
  const [about, setAbout] = useState("");
  const [own, setOwn] = useState("");
  const [extra, setExtra] = useState("");
  const [readFirst, setReadFirst] = useState(() => readStored("readFirst", "") === "1");
  const [replies, setReplies] = useState<Reply[] | null>(null);
  const [busy, setBusy] = useState<null | "writing" | "weighing" | "replies" | "check">(null);
  const [result, setResult] = useState<ReflectResult | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [lastInput, setLastInput] = useState<ReflectInput | null>(null);
  const [error, setError] = useState("");
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const [comment, setComment] = useState("");
  const [base, setBase] = useState(() => readStored("base", DEFAULT_BASE));
  const [model, setModel] = useState(() => readStored("model", "qwen3:8b"));
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
      .catch(() => setError("Cue cannot reach the local model. Is Ollama running?"));
  }, [base]);

  useEffect(() => store("model", model), [model]);
  useEffect(() => store("base", base), [base]);
  useEffect(() => store("readFirst", readFirst ? "1" : ""), [readFirst]);
  // Fill the person note when a known name is typed. Only replace text that
  // was loaded from a note, never what the user is typing.
  const loadedAbout = useRef("");
  useEffect(() => {
    const note = loadPeople()[person.trim().toLowerCase()] ?? "";
    setAbout((cur) => (note || cur === loadedAbout.current ? note : cur));
    loadedAbout.current = note;
  }, [person]);

  async function run() {
    if (!text.trim() || busy) return;
    const input: ReflectInput = {
      mode,
      text,
      context,
      aboutMe: aboutMe.trim() || undefined,
      person,
      relation,
      channel,
      about,
      intended: mode === "me" ? intended : undefined,
      own: mode === "them" && readFirst ? own : undefined,
    };
    savePerson(person, about);
    abort.current = new AbortController();
    setError("");
    setResult(null);
    setVerdict(null);
    setComment("");
    setReplies(null);
    setExtra("");
    setDraft(null);
    try {
      const r = await reflect(base, model, input, abort.current.signal, (step, d) => {
        setBusy(step);
        if (d) setDraft(d);
      });
      setResult(r);
      setLastInput(input);
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(String((e as Error).message ?? e));
    } finally {
      setBusy(null);
    }
  }

  async function runStyle() {
    if (!style.trim() || busy) return;
    abort.current = new AbortController();
    setError("");
    setNoteText("");
    setBusy("writing");
    try {
      setNoteText(await styleNote(base, model, { style, relation, channel }, abort.current.signal));
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(String((e as Error).message ?? e));
    } finally {
      setBusy(null);
    }
  }

  async function runExtra() {
    if (!extra.trim() || !result || !lastInput || busy) return;
    abort.current = new AbortController();
    setBusy("weighing");
    try {
      const r = await addOwnReading(base, lastInput, result, extra, abort.current.signal);
      setResult(r);
      setLastInput({ ...lastInput, own: extra });
      setExtra("");
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(String((e as Error).message ?? e));
    } finally {
      setBusy(null);
    }
  }

  async function runCheck() {
    if (!result || !lastInput || busy || result.check) return;
    abort.current = new AbortController();
    setBusy("check");
    try {
      const check = await writeCheck(base, model, lastInput, result, abort.current.signal);
      setResult({ ...result, check });
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(String((e as Error).message ?? e));
    } finally {
      setBusy(null);
    }
  }

  async function runReplies() {
    if (!result || !lastInput || busy) return;
    abort.current = new AbortController();
    setBusy("replies");
    try {
      setReplies(await waysToReply(base, model, lastInput, result, abort.current.signal));
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(String((e as Error).message ?? e));
    } finally {
      setBusy(null);
    }
  }

  function give(v: Verdict) {
    if (!result || !lastInput) return;
    setVerdict(v);
    setSaved(addEntry({ at: new Date().toISOString(), input: lastInput, result, verdict: v, note: comment }));
  }

  const isThem = mode === "them";
  const who = person.trim() || (isThem ? "they" : "them");

  return (
    <div className="page">
      <header className="top">
        <div>
          <h1>Cue</h1>
          <p className="tagline">Possible readings, not answers.</p>
        </div>
        <div className="top-right">
          <span className="tag">Research prototype. Runs on this computer.</span>
          <button className="link" onClick={() => setShowAbout(true)}>
            About
          </button>
          <button className="link" onClick={() => setShowSettings((s) => !s)} aria-expanded={showSettings}>
            Settings
          </button>
        </div>
      </header>

      {showAbout && (
        <About
          onClose={() => {
            setShowAbout(false);
            store("seenAbout", "1");
          }}
        />
      )}

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
          <label className="check-row">
            <input type="checkbox" checked={readFirst} onChange={(e) => setReadFirst(e.target.checked)} />
            Let me write my own reading first, before Cue shows its readings
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
        <button role="tab" aria-selected={tab === "them"} onClick={() => setTab("them")}>
          What did they mean?
        </button>
        <button role="tab" aria-selected={tab === "me"} onClick={() => setTab("me")}>
          How might my message read?
        </button>
        <button role="tab" aria-selected={tab === "style"} onClick={() => setTab("style")}>
          Explain my style
        </button>
      </div>

      {tab === "style" ? (
        <main className="grid">
          <section className="card form" aria-label="Your communication style">
            <label>
              How do you communicate? <span className="hint">in your own words</span>
              <textarea
                id="style"
                rows={5}
                value={style}
                onChange={(e) => setStyle(e.target.value)}
                placeholder="For example: I write short messages and get to the point. I don't use many emojis. If I don't reply straight away, I'm usually just busy."
              />
            </label>
            <div className="pair two">
              <label>
                For my
                <select value={relation} onChange={(e) => setRelation(e.target.value)}>
                  {RELATIONS.map((r) => (
                    <option key={r} value={r}>
                      {r || "(anyone)"}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Where
                <select value={channel} onChange={(e) => setChannel(e.target.value)}>
                  {CHANNELS.map((c) => (
                    <option key={c} value={c}>
                      {c || "(anywhere)"}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="row">
              <button className="primary" onClick={runStyle} disabled={!style.trim() || !!busy}>
                {busy ? "Writing..." : "Write a style note"}
              </button>
            </div>
            <p className="privacy">Optional. Some people find that a short note about how they communicate heads off misunderstandings. Sharing it is up to you.</p>
          </section>
          <section className="card result" aria-live="polite" aria-label="Style note">
            {error && <p className="error">{error}</p>}
            {!note && !error && <p className="empty">Your note appears here. Edit it freely before using it.</p>}
            {note && (
              <>
                <span className="label">A note you could share</span>
                <textarea id="note-out" rows={5} value={note} onChange={(e) => setNoteText(e.target.value)} />
                <div className="row">
                  <button
                    onClick={() => {
                      navigator.clipboard?.writeText(note).catch(() => undefined);
                    }}
                  >
                    Copy
                  </button>
                </div>
              </>
            )}
          </section>
        </main>
      ) : (
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
          {person.trim() && (
            <label>
              What you know about {person.trim()} <span className="hint">optional, remembered for this name on this computer only</span>
              <input
                value={about}
                onChange={(e) => setAbout(e.target.value)}
                placeholder="For example: always brief on Slack; we get on well"
              />
            </label>
          )}
          <label>
            What came before <span className="hint">optional: earlier messages, or what was happening</span>
            <textarea rows={4} value={context} onChange={(e) => setContext(e.target.value)} />
          </label>
          <label>
            About you <span className="hint">optional: used for replies, not to weigh what they meant</span>
            <input value={aboutMe} onChange={(e) => setAboutMe(e.target.value)} placeholder="For example: I am autistic; I often get anxious about messages" />
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
          {isThem && readFirst && (
            <label>
              Your reading first <span className="hint">optional: what do you think {who} meant?</span>
              <input value={own} onChange={(e) => setOwn(e.target.value)} />
            </label>
          )}
          <div className="row">
            <button className="primary" onClick={run} disabled={!text.trim() || !!busy}>
              {busy === "writing" ? "Writing possible readings..." : busy === "weighing" ? "Weighing them..." : "Read it"}
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
                ? "Paste what someone said. Cue offers a few ways it could be meant, including that they meant it exactly as said."
                : "Paste a message you plan to send. Cue shows how it might come across, so you can decide whether to change anything. You do not have to."}
            </p>
          )}
          {busy && !result && !draft?.meanings.length && (
            <p className="empty">Thinking it through. This can take up to a minute on a laptop.</p>
          )}
          {busy && !result && draft && draft.meanings.length > 0 && (
            <div className="draft">
              {draft.literal && (
                <p className="literal">
                  <span className="label">The words say</span>
                  {draft.literal}
                </p>
              )}
              <ol className="readings">
                {draft.meanings.map((m, i) => (
                  <li key={i}>
                    <div className="reading-head">
                      <span className="likelihood pending">{busy === "weighing" ? "Weighing..." : "Not weighed yet"}</span>
                      {i === 0 && <span className="face">{isThem ? "Taken at face value" : "As you mean it"}</span>}
                    </div>
                    <p>{m}</p>
                  </li>
                ))}
              </ol>
            </div>
          )}
          {result && (
            <>
              {result.safety?.distress && (
                <p className="safety" role="note">
                  <span className="label">Before the readings</span>
                  {result.safety.distress}
                </p>
              )}
              {result.safety?.scam && (
                <p className="safety" role="note">
                  <span className="label">Be careful</span>
                  {result.safety.scam}
                </p>
              )}
              <div className={`clarity clarity-${result.clarity}`}>
                <strong>{(isThem ? CLARITY_TEXT : CLARITY_TEXT_ME)[result.clarity].word}</strong>
                <span>{(isThem ? CLARITY_TEXT : CLARITY_TEXT_ME)[result.clarity].detail}</span>
              </div>
              <p className="literal">
                <span className="label">The words say</span>
                {result.literal}
              </p>
              <ol className="readings">
                {result.readings.filter((r) => !r.hidden).map((r, i) => (
                  <li key={i}>
                    <div className="reading-head">
                      <span className="likelihood">{result.abstain ? "Possible" : likelihood(r.p)}</span>
                      <span className="bar" aria-hidden="true">
                        <span style={{ width: `${Math.round(r.p * 100)}%` }} />
                      </span>
                      {r.literal && <span className="face">{isThem ? "Taken at face value" : "As you mean it"}</span>}
                      {r.own && <span className="face own">Your reading</span>}
                    </div>
                    <p>{r.meaning}</p>
                    {r.cues.length > 0 && <p className="cues">Based on: {r.cues.join("; ")}</p>}
                  </li>
                ))}
              </ol>
              {isThem && (
                <div className="row add-own">
                  <input
                    aria-label="Add your own reading"
                    placeholder={result.readings.some((r) => r.own) ? "Try a different reading of your own" : "None of these? Add your own reading"}
                    value={extra}
                    onChange={(e) => setExtra(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") runExtra();
                    }}
                  />
                  <button onClick={runExtra} disabled={!extra.trim() || !!busy}>
                    {busy === "weighing" ? "Weighing..." : "Weigh it"}
                  </button>
                </div>
              )}
              {result.boundary && (
                <p className="boundary">
                  <span className="label">Worth remembering</span>
                  {result.boundary}
                </p>
              )}
              {result.check ? (
                <p className="check">
                  <span className="label">{isThem ? "To check, you could ask" : "If you want, you could add"}</span>
                  {result.check}
                </p>
              ) : (
                <div className="row">
                  <button onClick={runCheck} disabled={!!busy}>
                    {busy === "check"
                      ? "Writing..."
                      : isThem
                        ? "A question you could ask (optional)"
                        : "A sentence you could add (optional)"}
                  </button>
                </div>
              )}
              {isThem && !replies && (
                <div className="row">
                  <button onClick={runReplies} disabled={!!busy}>
                    {busy === "replies" ? "Thinking of replies..." : "Ideas for replying (optional)"}
                  </button>
                </div>
              )}
              {replies && (
                <div className="replies">
                  <span className="label">Ways you could reply, if you want to</span>
                  <ul>
                    {replies.map((x, i) => (
                      <li key={i}>
                        <p>{x.reply}</p>
                        <p className="cues">How it might land: {x.might_land}</p>
                      </li>
                    ))}
                  </ul>
                  <p className="meta">These are options, not advice. Your own words are fine.</p>
                </div>
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
                  value={comment}
                  onChange={(e) => setComment(e.target.value)}
                  onBlur={() => verdict && give(verdict)}
                />
              </div>
              <p className="meta">
                {result.model}, {(result.ms / 1000).toFixed(0)} s. Cue can be wrong. You know the people involved; it does not.
              </p>
            </>
          )}
        </section>
      </main>
      )}
    </div>
  );
}
