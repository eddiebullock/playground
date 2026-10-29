import type { EmotionRead } from "../analysis/emotion.ts";

type Props = {
  emotion: EmotionRead;
  hasFace: boolean;
  workplace: boolean;
};

export function EmotionTab({ emotion, hasFace, workplace }: Props) {
  if (workplace) {
    return (
      <section className="panel" aria-label="Emotion">
        <h2>Face shape</h2>
        <p className="warn">
          Emotion inference from faces is off at work and in education (EU AI
          Act). Switch that off in the header if this is personal use.
        </p>
      </section>
    );
  }
  return (
    <section className="panel" aria-label="Emotion">
      <h2>Face shape</h2>
      <p className="lede">A guess from mouth and brow shape. It is not what they feel.</p>
      <p className="status">{hasFace ? emotion.hedge : "No face in the picture yet."}</p>
      {hasFace && emotion.cues.length > 0 && (
        <ul className="cues">
          {emotion.cues.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      )}
      {hasFace && <p className="detail">How clear this shape is: {emotion.confidence}.</p>}
      <p className="fine">A face is not a feeling. Context, words and history matter more.</p>
    </section>
  );
}
