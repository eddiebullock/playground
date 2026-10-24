# Study 5 draft protocol: feasibility and acceptability of Gloss

Draft for Eddie and the supervisors. Built on `docs/research/03_study_spinout.md`.
Items marked **[FG]** wait on the focus-group findings (Study 4). Items marked
**[E2]** wait on the model-hosting decision.

## Title

Augmenting social interaction with AI for autistic adults: a mixed-methods
feasibility and acceptability study of a calibrated, two-way communication
assistant.

## Objectives (Bowen et al. 2009 focus areas)

1. **Acceptability** (primary). How acceptable is Gloss to autistic adults,
   across the seven TFA constructs: affective attitude, burden, ethicality,
   intervention coherence, opportunity costs, perceived effectiveness and
   self-efficacy?
2. **Demand.** How often, and for which situations, do people use it?
3. **Practicality.** Does it run on participants' own computers, at what
   latency, and how often does it fail?
4. **Implementation.** Which features do people use (Reflect, Two-way, Live),
   and does use persist or fade over 6 weeks?
5. **Limited efficacy signal** (not powered). Pre/post change in reported
   effort of interactions, camouflaging (CAT-Q) and loneliness (UCLA-m);
   per-use perceived accuracy.
6. **Safety and harms.** Distress, over-reliance, readings experienced as
   wrong or hurtful, and stereotype-driven advice.
7. **Calibration in the wild.** When Gloss says "fairly clear", do users rate
   the reading right more often than when it says "hard to tell"? This is the
   field test of calibrated deferral.

## Design

Single arm, mixed methods, 6 weeks of real-world use, with one optional lab
session. If resources allow, add a waitlist arm of equal size (24-35 per arm)
so that the outcome measures' variance can be estimated against a control.

## Participants

- Autistic adults aged 18 or over, UK resident, formal diagnosis or
  self-identification. Self-identified people are included because late
  diagnosis is common and the tool is not diagnosis-gated. Analyses are
  reported both ways.
- Has a laptop or desktop that can run the tool **[E2: minimum spec]**.
- Recruited through the ARC database, Autistica's network and autistic-led
  organisations. **No NHS recruitment**, so the study stays with PREC and does
  not need HRA review.
- N = 36-45 enrolled, so that about 30 completers estimate a 70% engagement
  rate within about +/-16 percentage points. Expected attrition is 20-25%
  (Molehill Mountain retained 76%).

## Intervention

Gloss Reflect and Two-way, installed locally. Everything runs on the
participant's computer. No raw audio or video is captured. Participants paste
or type the messages or summaries they choose, and are told to use initials
for other people. Optional: one lab session with Gloss Live on a video call
with a consenting researcher or confederate. **[FG]** decides which features
and wording ship.

## Measures

| When | Measure |
|---|---|
| Baseline | Demographics; AQ-10 or RAADS-14 (descriptive only); CAT-Q; UCLA-m; ASQoL; a list of social situations ranked by frequency and distress (this fills the prevalence gap found in the literature); prospective TFA |
| During use | Event diary after any interaction where Gloss was used (in-app, about 30 s): effort of the interaction 0-10; was the reading right / partly / wrong / not sure; did you act on it; any distress. Usage logs stay on the device, and the participant exports them and chooses what to share. |
| Week 3 | Short check-in call or message (participant chooses the mode) |
| Week 6 | Retrospective TFA; SUS (all-positive version); UEQ-S; CAT-Q; UCLA-m; ASQoL; semi-structured exit interview structured by the TFA, with questions sent in advance and a choice of video, audio, text or email |

## Pre-registered progression criteria (Avery et al. 2017 traffic light)

| Domain | Green | Amber | Red |
|---|---|---|---|
| Recruitment within window | 100% | 50-99% | below 50% |
| Engagement (used it in 4 or more of 6 weeks) | 70% or more | 50-69% | below 50% |
| Retention to week-6 measures | 75% or more | 60-74% | below 60% |
| TFA general acceptability (rated acceptable) | 70% or more | 50-69% | below 50% |
| Perceived accuracy (right + partly, per use) | 70% or more | 50-69% | below 50% |
| Serious harm attributable to the tool | none | - | any |

## Analysis

Descriptive statistics with 95% CIs for the feasibility proportions. Pre/post
change with CIs, not significance tests. Calibration-in-the-wild: the
proportion rated right within each clarity band, compared across bands. TFA
framework analysis of interviews, with an autistic co-researcher as second
coder. A mixed-effects model of diary effort by Gloss use is exploratory.

## Involvement

Autistic advisory group (4-6 people, drawn from the focus groups where
willing), paid GBP 27.50/hour (NIHR rate). They co-design the wording of
outputs, the diary items and the progression thresholds, and review the
interview guide. Credited as co-authors where their contribution meets
authorship criteria.

## Ethics and data

- PREC. DPIA through the University Information Compliance Office.
  Lawful basis: public task, plus Art. 9(2)(j) research.
- Third parties: no recording. Participants enter only their own summaries or
  messages, and are asked to redact names. No raw media is stored. Diaries are
  exported by the participant.
- Risk: a reading could cause distress, for example by suggesting someone was
  unkind. Mitigations: the literal reading is always shown; hedged wording;
  the "Hard to tell" state; a signposting card; the researcher reviews any
  diary entry flagged as distressing.
- Intended-purpose statement: Gloss is a general communication assistant. It
  makes no claims to treat, diagnose or compensate for any condition. This
  keeps the study out of the medical-device clinical-investigation route
  (`03_study_spinout.md` section 2.6).

## Timeline (indicative)

| Month | Step |
|---|---|
| 0-1 | Fold in the focus-group findings; advisory group; finalise model hosting [E2] |
| 1-2 | PREC submission; DPIA; pilot with 3-5 advisory members |
| 3-5 | Recruit and run in rolling cohorts |
| 6 | Analysis; progression decision; Study 6 design with Emteq |
