# Gloss: master plan

Working name kept from the October concept scenes. A gloss is a note in the
margin that explains what a passage means. That is the product.

Owner: Eddie Bullock (ARC, Cambridge). Plan written 2026-10-24 by Claude,
working autonomously from: the first-year report, the PREC application, the
`social_ai` concept prototype, the `es-vlm` work, three literature/tech/
regulation reviews (`docs/research/`), and experiment E1. The focus-group
transcripts (sessions 2-4) were read on 2026-10-25. The themes and the
decisions they drive are in `docs/research/04_focus_groups.md`. They support
the order below and add five features (section 3).

---

## 1. The thesis in one paragraph

The hard part of AI social support for autistic adults is not recognition
accuracy. It is three other things: (1) **calibration**: saying "possibly"
when a signal is genuinely ambiguous, and staying quiet when it does not know;
(2) **direction**: treating a misunderstanding as a two-way translation problem
(double empathy) rather than correcting the autistic person; (3) **timing**:
most of the cost autistic adults describe comes before and after an
interaction (preparation, rumination, re-reading a message), not only in the
live moment. A tool that is calibrated, two-way and available before and after
the interaction addresses a gap nobody has filled. Every published device
(Superpower Glass, Brain Power, SocialMind) prompts live, labels faces with
basic emotions, and gives single confident answers. Mostly it was built for
children and judged by parents.

## 2. Recommendation: change the order, keep the destination

Proposed in the report: VLM reads social cues from video, later in a wearable.
Recommended instead: **language first, after the interaction first, faces last**.
The wearable stays the long-term destination.

| Reason | Evidence |
|---|---|
| Science | Facial configurations do not reliably signal emotion without context (Barrett et al. 2019). Your Study 1 found models strongest on prototypical states and weakest on the complex, low-intensity states that matter. Language and context carry most of the signal for sarcasm, indirect requests and implicature. |
| Law | EU AI Act Art. 5(1)(f) bans inferring emotion from biometric data (face, voice) at work and in education. Inference from text is outside the ban. A workplace video tool that reads colleagues' faces cannot be sold in the EU unless it is a CE-marked medical device. |
| Ethics / PREC | Live capture of other people (bystanders) is the hardest ethics problem. A tool that works on what the user pastes or summarises avoids most of it. |
| Community | Autistic critics reject tools that train toward neurotypical norms (Noora backlash, Spiel et al. 2019, Keyes 2020). Autistic adults already use ChatGPT and Goblin Tools to translate tone. They want it non-judgemental and private, and they fear "automated masking" and over-reliance. Our focus groups said the same: local processing, user control, choice between support and learning, and never "stop talking about your interest" (`04_focus_groups.md` T1, T3, T5, T9). |
| Evidence gap | No controlled study of message-tone interpretation for autistic adults. No tool that expresses calibrated uncertainty. No published LLM feasibility trial in autistic adults. Each one is a paper. |
| Feasibility | A reflective interpreter has no latency requirement, works on any laptop, and can be in participants' hands within weeks. |

What stays: multimodal VLM work (Studies 2-3) feeds the live mode and the
glasses. Emteq's inward-facing sensing (the user's own arousal/load, relevant
to alexithymia) is a better fit for the wearable than reading other people's
faces, and it raises fewer legal problems.

## 3. Product: three modes, built in this order

1. **Reflect** (build now, the Study 5 core). Paste a message, an email, a chat
   thread, or a short description of what happened. Gloss returns 2-4 possible
   readings with how likely each seems, the cues each rests on, an honest
   ambiguity rating ("most people would be unsure too"), and a way to check
   ("you could ask: ..."). It does not rewrite the user's voice unless asked.
   Added after the focus groups (`04_focus_groups.md`): an optional "your
   reading first" step; "add your own reading", which Gloss weighs without
   knowing it is the user's; private notes about a person, used as context;
   opt-in ideas for replying with how each might land; and a reminder that
   you can say no when someone pushes past a refusal.
2. **Two-way** (build now, small). "How might they read *my* message?" The same
   engine run on the user's own draft. It explains conventions as conventions,
   not rules, and it suggests a sentence to explain your own style if you want
   one. This is the double-empathy feature, and no competitor has it.
3. **Live** (build second). A Chrome extension for web video calls. Tab audio is
   the other person and the mic is the user. Captions, idiom/sarcasm glosses at
   turn ends, a "floor is open" turn-taking cue, and an on-demand recap. Faces
   are off by default and disabled in work settings for EU users. Used in the
   lab only with consenting confederates until ethics allows more.
4. **Wearable** (Study 6, with Emteq). The same event bus, with phone-hosted
   models. Add inward sensing of the user's own state. The focus groups
   preferred a phone, an earpiece, or a ring or wristband haptic to glasses.
   Live cues default to a tap or one word, with detail after the moment.

Fixed design rules (from the concept scenes, the research and calibrated deferral):
- Hedged language ("possibly", "may"). Colour is always paired with a word. A
  quiet mode, and an explainer card the user can show to others.
- The tool never tells the user how to sit, look or seem. No eye-contact
  coaching. No masking scripts unless the user explicitly asks. Never
  suggests talking less about an interest (focus groups T9). A user can opt in
  to a turn-taking reminder for themselves; it is never a default.
- Not a lie detector. It reads what most people would take from the words, and
  treats kind white lies as kindness (T2).
- It abstains when it is unsure, and abstaining counts as a feature in evaluation.
- Local-first. Raw audio and video are never stored. Data belongs to the user
  and is never reportable to an employer.
- A stereotype audit: disclosing autism must not shift advice towards
  avoidance (Wohn et al. CHI 2026). This is tested in evaluation.

## 4. Research programme (what each step proves)

| ID | Question | Status |
|---|---|---|
| E1 | Can free on-device LLMs read non-literal language at human level, and does their confidence track their own errors and human disagreement? (Hu et al. 2023 items with 374-person human data; MUStARD sarcasm) | Running. First signal: 4B models answering directly are well below humans, and their errors are mostly **literal readings**. |
| E2 | Scaling and reasoning: what model size and mode reaches human level with calibrated deferral? Sets the minimum hardware for Study 5. | Next (8B local; larger needs CSD3 with Eddie's go-ahead). |
| E3 | Selective prediction: with an abstain rule, what accuracy does the tool reach on the readings it does show, and at what coverage? This is the "calibrated deferral" headline. | From E1/E2 data. |
| E4 | Product-level eval: does Gloss Reflect's top reading match the intended meaning, and does its ambiguity rating match human disagreement? Stereotype audit. | Harness built 2026-10-25 (`app/scripts/eval.ts`); 8B run in progress. |
| E5 | Multimodal sarcasm: does prosody add to text (MUStARD audio)? Decides whether Live needs audio models. | Needs 1.4 GB of video and disk space. |
| E6 | Live latency on the M3: VAD, ASR and interpretation at turn ends inside a browser extension. | After Reflect. |
| E7 | Autistic-rater ground truth: do autistic and non-autistic raters disagree on items, and does the model align with either? (Online survey, needs ethics.) | Study 5 add-on. |

Paper candidates: (a) E1-E3: "Calibrated pragmatic inference in on-device
LLMs", a direct sequel to Study 1 that is product-relevant; (b) Study 5
feasibility; (c) E7 with the double-empathy framing.

## 5. Study 5: feasibility and acceptability (draft protocol in `docs/study5_protocol.md`)

Mixed methods, single arm, 6 weeks of real-world use of Reflect and Two-way,
N = 36-45 enrolled autistic adults (UK, 18+, self-identified or diagnosed). An
optional lab session tries Live with a consenting confederate. Outcomes are
organised by Bowen 2009. Acceptability uses the TFA questionnaire and TFA
interviews. Also SUS, a diary after each interaction (effort 0-10, was the
reading right, any distress), opt-in usage logs, and pre/post CAT-Q, UCLA-m
and ASQoL. Pre-registered traffic-light progression criteria. An autistic
advisory group paid at the NIHR rate (GBP 27.50/h). PREC route. No NHS
recruitment, to avoid HRA.

## 6. Spin-out track (in parallel, low effort now)

1. Get the signed iCASE collaboration agreement with Emteq and check foreground
   IP with the ROO contracts team **before** writing product code meant for a
   company. Keep the architecture independent of Emteq.
2. Intended-purpose wording: universal design ("a communication assistant that
   offers possible readings of tone and intent"). No claims to treat, reduce
   symptoms or "compensate for" a condition, because UK MDR makes software
   "intended for compensation for a handicap" a medical device. Get MHRA advice
   before any launch.
3. Routes: ICURe (next cycle), Founders at Cambridge, Accelerate Cambridge
   (deadline 15 Jan 2027), SVI seed (up to GBP 20k), RAEng Enterprise
   Fellowship after the viva. Cambridge Enterprise fast-track equity is 5% for
   unpatented "Other" software.
4. Moat: autistic co-design, calibrated uncertainty with evidence, privacy, and
   two-way framing. General LLMs do tone for free, so a "tone checker" alone is
   a commodity.

## 7. Autonomy protocol (how Claude works without Eddie)

- **Loop**: research -> hypothesis -> smallest experiment -> analyse -> record
  in `docs/DECISIONS.md` -> build -> evaluate against the experiments -> repeat.
- **Decide alone**: choice of experiments, prompts, analyses, UI, code
  structure, which public datasets to use, local model downloads within the
  disk budget.
- **Stop and ask Eddie**: spending money, cluster jobs (the runbook requires an
  explicit yes per command), anything touching licensed data (Mindreading,
  EU-Emotions), anything sent outside the machine, contacting people,
  committing to the main playground repo, deleting files Claude did not create.
- **Resources**: M3, 16 GB RAM, about 9 GB free disk (check `df -h` before
  each download; delete models no longer needed). No paid APIs.
- **Record keeping**: every run writes JSONL plus a summary. Every decision
  gets a dated line with its evidence. Negative results are kept.
- **Quality bar**: no claim without a number and a CI. Human baselines
  wherever they exist. Held-out items are never used for prompt tuning.

## 8. Sequence

Done on 2026-10-24: E1/E2/E3 (4B and 8B, direct, stance, calibration), Reflect
v0, Study 5 protocol draft. Done on 2026-10-25: focus groups read and folded
in; the features they asked for; E4 harness.

Next, in order:
1. E4 product eval on 8B: does the full pipeline get the intended meaning, and
   does its clarity track human disagreement? Fix what it shows.
2. E4 stereotype audit (disclosure must not shift readings).
3. Latency on the M3 (the focus groups asked for speed; Reflect's 8B pipeline
   takes about a minute).
4. An "How good is it?" panel in the app with the E1/E4 numbers (T1: trust
   needs evidence).
5. Brief and summary for Eddie.

## 9. Decisions needed from Eddie

1. Do you accept the order: language first, reflective first, faces last? (Recommended.)
2. Model hosting for Study 5. If local 8B models are not good enough, the
   options are: (a) participants need 16 GB laptops; (b) a university-hosted
   open model (needs IT/HPC agreement); (c) a frontier API under a university
   data agreement (costs money, so it needs your approval and a budget).
3. ~~Focus-group transcripts~~: read on 2026-10-25.
4. Locate the iCASE agreement.
5. Should Gloss seek review or endorsement from an autistic-led organisation
   (for example the NAS) before Study 5? The focus groups named this as a
   condition of trust (T1). It means contacting people, so it is your call.
6. The `gloss` folder now sits inside `playground/social_ai` but is still its
   own git repository with no remote. Merge it into the playground repo with
   history (`git subtree`), or give it its own GitHub repository? Nothing has
   been pushed.
