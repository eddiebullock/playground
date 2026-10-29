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
        <h2>Emotion</h2>
        <p className="warn">
          Face labels are off at work and in education (EU AI Act). Switch that
          off in the header if this is personal use.
        </p>
      </section>
    );
  }
  return (
    <section className="panel" aria-label="Emotion">
      <h2>Emotion</h2>
      <p className="word">{hasFace ? emotion.label : "None"}</p>
      <p className="fine">Guess from the face. Not a feeling.</p>
    </section>
  );
}
